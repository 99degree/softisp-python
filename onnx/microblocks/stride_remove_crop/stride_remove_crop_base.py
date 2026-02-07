from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class CropWidthFixedBase(MicroblockBase):
    """
    Microblock that crops the width dimension of an [N,C,H,W] tensor
    to a fixed width value. N, C, H remain unchanged.
    """
    name = "stride_remove_crop"
    version = "v0"

    def __init__(self, fixed_width: int = 1920, dim = 4):
        super().__init__()
        self.fixed_width = fixed_width
        self.dim = dim

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        out_name    = f"{stage}.applier"

        # Names
        starts = f"{stage}.crop_starts"
        ends   = f"{stage}.crop_ends"
        axes   = f"{stage}.crop_axes"

        # Initializers
        starts_const = oh.make_tensor(starts, TensorProto.INT64, [1], [0])                 # start at 0
        ends_const   = oh.make_tensor(ends,   TensorProto.INT64, [1], [self.fixed_width])  # fixed width
        axes_const   = oh.make_tensor(axes,   TensorProto.INT64, [1], [3])                 # slice width axis

        # Nodes
        slice_node = oh.make_node(
            "Slice",
            inputs=[input_image, starts, ends, axes],
            outputs=[out_name],
            name=f"{stage}_slice"
        )

        inits = [starts_const, ends_const, axes_const]

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", self.dim, "H", "w"]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", self.dim, "H", "W"]),
        ]

        outputs = {"applier": {"name": out_name, "type": TensorProto.FLOAT, "shape":["n", self.dim, "H", "W"] }}

        return BuildResult(outputs, [slice_node], inits, vis) \
            .appendInput(input_image, type=TensorProto.FLOAT)

    def build_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        # For testing, reuse applier logic
        return super().build_test_algo(stage, prev_stages)


class CropWidth3CH(CropWidthFixedBase):
    dim = 3
    name = "stride_remove_crop"
    version = "v1"

    def __init__(self, fixed_width: int = 1920, dim: int = 3):
        super().__init__(fixed_width=fixed_width, dim=dim)
