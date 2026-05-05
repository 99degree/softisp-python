from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class CropWidthFixedBase(MicroblockBase):
    """
    Microblock that removes DMA stride/padding from the width dimension
    of an [N,C,H,W] tensor.
    
    The input tensor has stride (width includes padding for DMA alignment).
    The actual image width comes from image_desc stage. After bayer2cfa
    (steps=[2,2]), dimensions are halved, so target = image_desc.width / 2.
    
    Inputs (two prev_stages):
        prev_stages[0]: Image tensor source (e.g., bayernorm)
        prev_stages[1]: Metadata source providing width (e.g., image_desc)
    
    Example:
        4K:  image_desc.width=3840, after bayer2cfa stride=1920, target=1920
        FHD: image_desc.width=1920, after bayer2cfa stride=1024, target=960
    """
    name = "stride_remove_crop"
    version = "v0"
    dim = 4

    def build_applier(self, stage: str, prev_stages=None):
        upstream_image = prev_stages[0] if prev_stages and len(prev_stages) > 0 else stage
        upstream_meta = prev_stages[1] if prev_stages and len(prev_stages) > 1 else stage
        
        input_image = f"{upstream_image}.applier"
        input_width = f"{upstream_meta}.width"  # actual image width
        out_name    = f"{stage}.applier"

        half = f"{stage}.half"
        target_width = f"{stage}.target_width"
        starts = f"{stage}.starts"
        axes   = f"{stage}.axes"

        inits = [
            oh.make_tensor(half, TensorProto.INT64, [], [2]),
            oh.make_tensor(starts, TensorProto.INT64, [], [0]),
            oh.make_tensor(axes, TensorProto.INT64, [], [3]),
        ]

        nodes = [
            oh.make_node('Div', inputs=[input_width, half], outputs=[target_width],
                        name=f"{stage}_div_target"),
            oh.make_node('Slice', inputs=[input_image, starts, target_width, axes],
                        outputs=[out_name], name=f"{stage}_slice"),
        ]

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", self.dim, "H", "w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", self.dim, "H", "W"]),
        ]

        outputs = {"applier": {"name": out_name, "type": TensorProto.FLOAT, "shape":["n", self.dim, "H", "W"]}}
        return BuildResult(outputs, nodes, inits, vis) \
            .appendInput(input_image, type=TensorProto.FLOAT) \
            .appendInput(input_width, type=TensorProto.INT64, shape=[1])

    def build_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class CropWidth3CH(CropWidthFixedBase):
    dim = 3
    name = "stride_remove_crop_3ch"
    version = "v1"