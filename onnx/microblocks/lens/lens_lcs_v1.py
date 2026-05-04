from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class LensLCSV1(MicroblockBase):
    """
    LensLCSV1 (v1)
    --------------
    Adaptive lens shading correction block with coefficient resizing.

    Inputs (external):
        - prev_stage.applier : upstream image tensor [n,3,h,w]
        - lcs_coeffs : full-resolution correction coefficients [H,W]

    Outputs:
        - applier : corrected image tensor [n,3,h*,w*]
        - lcs_coeffs_resized : resized coefficient map [h*,w*]
        - lcs_coeffs_out : identity copy of original coeffs [H,W]
    """
    name = "lens_lcs_v1"
    family = "lens_lcs_v1"
    version = "v1"

    def build_algo(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        input_image = f"{upstream}.applier"
        lcs_coeffs  = f"{stage}.lcs_coeffs"

        # Constants as rank-1 tensors
        one_n, one_c = f"{stage}.one_n", f"{stage}.one_c"
        h_factor, w_factor = f"{stage}.h_factor", f"{stage}.w_factor"
        inits += [
            oh.make_tensor(one_n, TensorProto.FLOAT, [1], [1.0]),
            oh.make_tensor(one_c, TensorProto.FLOAT, [1], [1.0]),
            oh.make_tensor(h_factor, TensorProto.FLOAT, [1], [0.5]),
            oh.make_tensor(w_factor, TensorProto.FLOAT, [1], [0.5]),
        ]

        # Concat into [1,1,0.5,0.5]
        scales = f"{stage}.scales"
        nodes.append(
            oh.make_node(
                "Concat",
                inputs=[one_n, one_c, h_factor, w_factor],
                outputs=[scales],
                name=f"{stage}.concat_scales",
                axis=0,
            )
        )
        vis.append(oh.make_tensor_value_info(scales, TensorProto.FLOAT, [4]))

        roi = f"{stage}.roi_empty"
        inits.append(oh.make_tensor(roi, TensorProto.FLOAT, [0], []))

        # Resize using scales only (no sizes)
        lcs_resized = f"{stage}.lcs_coeffs_resized"
        nodes.append(
            oh.make_node(
                "Resize",
                inputs=[lcs_coeffs, roi, scales],  # only data + scales
                outputs=[lcs_resized],
                name=f"{stage}.resize_lcs",
                mode="linear",
            )
        )
        vis.append(oh.make_tensor_value_info(lcs_resized, TensorProto.FLOAT, [1, 1, "h*", "w*"]))

        # Identity for original coeffs
        lcs_coeffs_out = f"{stage}.lcs_coeffs_out"
        nodes.append(
            oh.make_node("Identity", inputs=[lcs_coeffs], outputs=[lcs_coeffs_out], name=f"{stage}.identity_lcs")
        )
        vis.append(oh.make_tensor_value_info(lcs_coeffs_out, TensorProto.FLOAT, [1,1, "H", "W"]))

        # Apply correction
        applier = f"{stage}.applier"
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_resized], outputs=[applier], name=f"{stage}.algo_mul_apply")
        )
        vis.append(oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]))

        outputs = {
            "applier": {"name": applier},
            "lcs_coeffs_resized": {"name": lcs_resized},
            "lcs_coeffs": {"name": lcs_coeffs_out},
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, ['n','c',"h","w"], type=TensorProto.FLOAT)
        result.appendInput(lcs_coeffs, [1, 1, 'h', 'w'], type=TensorProto.FLOAT)
        return result

    def build_applier(self, stage: str, prev_stages=None):
        vis, nodes, inits = [], [], []
        upstream = prev_stages[0] if prev_stages else stage

        input_image = f"{upstream}.applier"
        lcs_coeffs  = f"{stage}.lcs_coeffs"
        applier     = f"{stage}.applier"

        # Apply correction directly
        nodes.append(
            oh.make_node("Mul", inputs=[input_image, lcs_coeffs], outputs=[applier], name=f"{stage}.mul_apply")
        )

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
            oh.make_tensor_value_info(lcs_coeffs, TensorProto.FLOAT, [1, 1, "h", "w"]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, ["n", 3, "h*", "w*"]),
        ]

        outputs = {"applier": {"name": applier}}

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, desc=['n', 'c', "h", "w"])
        result.appendInput(lcs_coeffs, type=TensorProto.FLOAT, desc=[1, 1, "h", "w"])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        H, W = 1080, 1920
        VALUE = 0.5

        coeffs_in   = f"{stage}.lcs_coeffs_in"
        coeffs_out  = f"{stage}.lcs_coeffs"
        coeffs_resz = f"{stage}.lcs_coeffs_resized"

        # Initializer: tensor filled with 0.5
        coeffs_data = [VALUE] * (H * W)
        inits = [
            oh.make_tensor(coeffs_in, TensorProto.FLOAT, [1, 1, 'H', 'W'], coeffs_data),
        ]

        # Identity node to expose original coeffs
        id_node = oh.make_node("Identity", inputs=[coeffs_in], outputs=[coeffs_out], name=f"{stage}_coeffs_id")

        # Scales constant for Resize: shrink both dims by half
        scales_name = f"{stage}.scales"
        scales_init = oh.make_tensor(scales_name, TensorProto.FLOAT, [4], [1, 1, 0.5, 0.5])
        inits.append(scales_init)

        # Explicit empty ROI
        roi = f"{stage}.roi_empty"
        inits += [oh.make_tensor(roi, TensorProto.FLOAT, [0], [])]

        # Resize node
        resize_node = oh.make_node(
            "Resize",
            inputs=[coeffs_in, roi, scales_name],  # roi left empty
            outputs=[coeffs_resz],
            name=f"{stage}_resize_lcs",
            mode="linear",
        )

        nodes = [id_node, resize_node]

        vis = [
            oh.make_tensor_value_info(coeffs_in,   TensorProto.FLOAT, [1,1,'H', 'W']),
            oh.make_tensor_value_info(coeffs_out,  TensorProto.FLOAT, [1,1,'H', 'W']),
            oh.make_tensor_value_info(coeffs_resz, TensorProto.FLOAT, [1,1, 'H // 2', 'W // 2']),
        ]

        outputs = {
            "lcs_coeffs":         {"name": coeffs_out},
            "lcs_coeffs_resized": {"name": coeffs_resz},
        }

        return BuildResult(outputs, nodes, inits, vis)