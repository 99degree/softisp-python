from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class CropWidthFixedBase(MicroblockBase):
    """
    Microblock that crops the width dimension of an [N,C,H,W] tensor
    to remove stride/padding. The input tensor has stride (width=stride),
    and we want to crop it to the actual width (width=width).
    
    For example:
    - Input: [N,C,H,2048] with stride=2
    - Output: [N,C,H,1024] (2048/2 = 1024)
    
    The stride is passed as a parameter, and the crop width is calculated
    as input_width / stride.
    """
    name = "stride_remove_crop"
    version = "v0"
    
    # Default stride (can be overridden in subclasses)
    stride = 2
    dim = 4

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Names
        input_shape = f"{stage}.input_shape"
        input_width = f"{stage}.input_width"
        stride_val = f"{stage}.stride_val"
        crop_width = f"{stage}.crop_width"
        starts = f"{stage}.crop_starts"
        ends   = f"{stage}.crop_ends"
        axes   = f"{stage}.crop_axes"

        # Initializers
        stride_const = oh.make_tensor(stride_val, TensorProto.INT64, [], [self.stride])
        axes_const   = oh.make_tensor(axes,   TensorProto.INT64, [], [3])  # slice width axis

        # Nodes
        # Get input shape
        nodes = [
            oh.make_node('Shape', inputs=[input_image], outputs=[input_shape],
                        name=f"{stage}_shape")
        ]
        
        # Extract width (index 3)
        width_index = f"{stage}.width_index"
        inits = [oh.make_tensor(width_index, TensorProto.INT64, [], [3])]
        nodes.append(
            oh.make_node('Gather', inputs=[input_shape, width_index], outputs=[input_width],
                        name=f"{stage}_gather_width")
        )
        
        # Calculate crop width: input_width / stride
        nodes.append(
            oh.make_node('Div', inputs=[input_width, stride_const], outputs=[crop_width],
                        name=f"{stage}_div_crop_width")
        )
        
        # Create starts (0) and ends (crop_width)
        starts_const = oh.make_tensor(starts, TensorProto.INT64, [], [0])
        inits.extend([stride_const, starts_const, axes_const])
        
        nodes.append(
            oh.make_node('Slice', inputs=[input_image, starts, crop_width, axes],
                        outputs=[out_name],
                        name=f"{stage}_slice")
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", self.dim, "H", "w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", self.dim, "H", "W"]),
        ]

        outputs = {"applier": {"name": out_name, "type": TensorProto.FLOAT, "shape":["n", self.dim, "H", "W"] }}

        return BuildResult(outputs, nodes, inits, vis) \
            .appendInput(input_image, type=TensorProto.FLOAT)

    def build_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        # For testing, reuse applier logic
        return super().build_test_algo(stage, prev_stages)


class CropWidth3CH(CropWidthFixedBase):
    """
    3-channel variant for RGB/YUV images.
    """
    dim = 3
    name = "stride_remove_crop"
    version = "v1"


class CropWidthStride2(CropWidthFixedBase):
    """
    Variant with stride=2 (most common case).
    Input width is divided by 2 to get output width.
    """
    stride = 2
    name = "stride_remove_crop"
    version = "v2"


class CropWidth3CHStride2(CropWidthFixedBase):
    """
    3-channel variant with stride=2.
    """
    dim = 3
    stride = 2
    name = "stride_remove_crop"
    version = "v3"


class CropWidthStride4(CropWidthFixedBase):
    """
    Variant with stride=4.
    Input width is divided by 4 to get output width.
    """
    stride = 4
    name = "stride_remove_crop"
    version = "v4"


class CropWidth3CHStride4(CropWidthFixedBase):
    """
    3-channel variant with stride=4.
    """
    dim = 3
    stride = 4
    name = "stride_remove_crop"
    version = "v5"
