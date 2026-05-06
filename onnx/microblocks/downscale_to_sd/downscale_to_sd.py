from microblocks.base import BuildResult, MicroblockBase, ApplierPassthrough
import onnx.helper as oh
from onnx import TensorProto

class DownscaleToSDBase(ApplierPassthrough, MicroblockBase):
    """
    Microblock that adaptively downscales any input resolution to roughly
    SD size using an integer ratio, with padding alignment for forthcoming
    algorithm stages.
    
    How it works:
    1. Extract N, C, H, W from input tensor shape
    2. Calculate integer ratio: ratio = max(ceil(H/target_H), ceil(W/target_W), 1)
    3. Compute output_H = H // ratio, output_W = W // ratio
    4. Align to padding boundary: padded_H = ceil(output_H / align) * align
    5. Resize to [N, C, padded_H, padded_W]
    
    Padding alignment ensures the output tensor dimensions are multiples
    of the alignment value (default 16), which is required by many ISP
    algorithm stages (AE, AWB, demosaic, etc.).
    
    Examples (original resolution, before bayer2cfa, alignment=16):
        4K  (3840x2160) → ratio=6 → 640x360 → 640x368
        FHD (1920x1080) → ratio=3 → 640x360 → 640x368
        HD  (1280x720)  → ratio=2 → 640x360 → 640x368
        SD  (720x480)   → ratio=1 → 720x480 → 720x480 (already aligned)
    
    Uses ONNX Resize operator with nearest-neighbor mode for speed.
    """
    name = "downscale_to_sd"
    version = "v0"
    target_height = 480
    target_width = 640
    alignment = 16  # padding alignment for forthcoming algo stages
    dim = 4

    def _build(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages and len(prev_stages) > 0 else stage
        input_image = f"{upstream}.applier"
        out_name = f"{stage}.applier"

        # Shape indices
        idx_0 = f"{stage}.idx_0"
        idx_1 = f"{stage}.idx_1"
        idx_2 = f"{stage}.idx_2"
        idx_3 = f"{stage}.idx_3"

        # Input shape
        shape = f"{stage}.shape"
        n_dim = f"{stage}.n"
        c_dim = f"{stage}.c"
        h_dim = f"{stage}.h"
        w_dim = f"{stage}.w"

        # Ratio calculation
        t_h = f"{stage}.t_h"
        t_w = f"{stage}.t_w"
        h_div = f"{stage}.h_div"
        w_div = f"{stage}.w_div"
        ratio = f"{stage}.ratio"
        one_const = f"{stage}.one"
        one_int = f"{stage}.one_int"
        zero_int = f"{stage}.zero_int"

        # Downscaled output dimensions (before padding)
        out_h = f"{stage}.out_h"
        out_w = f"{stage}.out_w"

        # Padding alignment
        align_val = f"{stage}.align_val"
        align_m1 = f"{stage}.align_m1"
        h_pad = f"{stage}.h_pad"
        w_pad = f"{stage}.w_pad"
        h_add = f"{stage}.h_add"
        w_add = f"{stage}.w_add"
        h_div_align = f"{stage}.h_div_align"
        w_div_align = f"{stage}.w_div_align"

        # Final sizes
        sizes = f"{stage}.sizes"
        sizes_int64 = f"{stage}.sizes_int64"

        inits = [
            oh.make_tensor(idx_0, TensorProto.INT64, [], [0]),
            oh.make_tensor(idx_1, TensorProto.INT64, [], [1]),
            oh.make_tensor(idx_2, TensorProto.INT64, [], [2]),
            oh.make_tensor(idx_3, TensorProto.INT64, [], [3]),
            oh.make_tensor(t_h, TensorProto.INT64, [], [self.target_height]),
            oh.make_tensor(t_w, TensorProto.INT64, [], [self.target_width]),
            oh.make_tensor(one_const, TensorProto.INT64, [], [1]),
            oh.make_tensor(one_int, TensorProto.INT64, [], [1]),
            oh.make_tensor(zero_int, TensorProto.INT64, [], [0]),
            oh.make_tensor(align_val, TensorProto.INT64, [], [self.alignment]),
            oh.make_tensor(align_m1, TensorProto.INT64, [], [self.alignment - 1]),
        ]

        nodes = [
            # 1. Extract N, C, H, W from shape
            oh.make_node('Shape', inputs=[input_image], outputs=[shape], name=f"{stage}_shape"),
            oh.make_node('Gather', inputs=[shape, idx_0], outputs=[n_dim], name=f"{stage}_gather_n"),
            oh.make_node('Gather', inputs=[shape, idx_1], outputs=[c_dim], name=f"{stage}_gather_c"),
            oh.make_node('Gather', inputs=[shape, idx_2], outputs=[h_dim], name=f"{stage}_gather_h"),
            oh.make_node('Gather', inputs=[shape, idx_3], outputs=[w_dim], name=f"{stage}_gather_w"),

            # 2. Calculate ceil division for ratio:
            #    h_div = (H + target_H - 1) // target_H
            #    w_div = (W + target_W - 1) // target_W
            oh.make_node('Add', inputs=[h_dim, t_h], outputs=[f"{stage}.h_plus_t"],
                        name=f"{stage}_add_h_t"),
            oh.make_node('Add', inputs=[w_dim, t_w], outputs=[f"{stage}.w_plus_t"],
                        name=f"{stage}_add_w_t"),
            oh.make_node('Sub', inputs=[f"{stage}.h_plus_t", one_const], outputs=[h_div],
                        name=f"{stage}_sub_h"),
            oh.make_node('Sub', inputs=[f"{stage}.w_plus_t", one_const], outputs=[w_div],
                        name=f"{stage}_sub_w"),
            oh.make_node('Div', inputs=[h_div, t_h], outputs=[f"{stage}.h_ratio"],
                        name=f"{stage}_div_h"),
            oh.make_node('Div', inputs=[w_div, t_w], outputs=[f"{stage}.w_ratio"],
                        name=f"{stage}_div_w"),

            # 3. ratio = max(h_ratio, w_ratio, 1)
            oh.make_node('Max', inputs=[f"{stage}.h_ratio", f"{stage}.w_ratio"], outputs=[ratio],
                        name=f"{stage}_max_hw"),
            oh.make_node('Max', inputs=[ratio, one_int], outputs=[ratio],
                        name=f"{stage}_max_one"),

            # 4. Downscaled output: output_H = H // ratio, output_W = W // ratio
            oh.make_node('Div', inputs=[h_dim, ratio], outputs=[out_h], name=f"{stage}_div_out_h"),
            oh.make_node('Div', inputs=[w_dim, ratio], outputs=[out_w], name=f"{stage}_div_out_w"),

            # 5. Align to padding boundary:
            #    padded_H = ((output_H + alignment - 1) // alignment) * alignment
            oh.make_node('Add', inputs=[out_h, align_m1], outputs=[h_add],
                        name=f"{stage}_add_h_align"),
            oh.make_node('Add', inputs=[out_w, align_m1], outputs=[w_add],
                        name=f"{stage}_add_w_align"),
            oh.make_node('Div', inputs=[h_add, align_val], outputs=[h_div_align],
                        name=f"{stage}_div_h_align"),
            oh.make_node('Div', inputs=[w_add, align_val], outputs=[w_div_align],
                        name=f"{stage}_div_w_align"),
            oh.make_node('Mul', inputs=[h_div_align, align_val], outputs=[h_pad],
                        name=f"{stage}_mul_h_pad"),
            oh.make_node('Mul', inputs=[w_div_align, align_val], outputs=[w_pad],
                        name=f"{stage}_mul_w_pad"),

            # 6. Concatenate sizes: [N, C, padded_H, padded_W]
            oh.make_node('Concat', inputs=[n_dim, c_dim, h_pad, w_pad], outputs=[sizes],
                        name=f"{stage}_concat_sizes", axis=0),

            # 7. Cast sizes to int64 (Resize expects INT64 for sizes)
            oh.make_node('Cast', inputs=[sizes], outputs=[sizes_int64],
                        name=f"{stage}_cast_sizes", to=TensorProto.INT64),
        ]

        # 8. Resize with nearest neighbor for speed
        roi = f"{stage}.roi"
        nodes.append(
            oh.make_node('Resize', inputs=[input_image, roi, zero_int, sizes_int64],
                        outputs=[out_name], name=f"{stage}_resize",
                        mode='nearest', coordinate_transformation_mode='asymmetric',
                        nearest_mode='floor')
        )
        inits.append(oh.make_tensor(roi, TensorProto.FLOAT, [0], []))

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", self.dim, "H", "w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", self.dim, "H", "W"]),
        ]

        outputs = {"applier": {"name": out_name, "type": TensorProto.FLOAT, "shape": ["n", self.dim, "H", "W"]}}
        return BuildResult(outputs, nodes, inits, vis) \
            .appendInput(input_image, type=TensorProto.FLOAT)

    def build_algo(self, stage: str, prev_stages=None):
        return self._build(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return super().build_test_algo(stage, prev_stages)


# === Subclasses with different targets ===

class DownscaleToSD(DownscaleToSDBase):
    """SD target 640x480, alignment 16."""
    target_height = 480
    target_width = 640
    alignment = 16
    version = "v1"


class DownscaleToSD3CH(DownscaleToSDBase):
    """3-channel variant, SD target 640x480."""
    target_height = 480
    target_width = 640
    alignment = 16
    dim = 3
    version = "v2"


class DownscaleToQVGA(DownscaleToSDBase):
    """QVGA target 320x240, alignment 16."""
    target_height = 240
    target_width = 320
    alignment = 16
    version = "v3"


class DownscaleToQVGA3CH(DownscaleToSDBase):
    """3-channel variant, QVGA target 320x240."""
    target_height = 240
    target_width = 320
    alignment = 16
    dim = 3
    version = "v4"


class DownscaleToSDAlign32(DownscaleToSDBase):
    """SD target 640x480, alignment 32."""
    target_height = 480
    target_width = 640
    alignment = 32
    version = "v5"


class DownscaleToSD3CHAlign32(DownscaleToSDBase):
    """3-channel variant, SD target 640x480, alignment 32."""
    target_height = 480
    target_width = 640
    alignment = 32
    dim = 3
    version = "v6"