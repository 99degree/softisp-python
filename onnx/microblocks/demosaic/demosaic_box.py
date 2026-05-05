# demosaic_box_v1.py
# Lightweight demosaic applier using box/nearest or linear interpolation.
# Compatible with SoftISP microblocks framework.
from __future__ import annotations
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import BuildResult
from .demosaic_base import DemosaicBase
import numpy as np


class DemosaicBoxV0(DemosaicBase):
    name = "demosaic_box"
    version = "v0"
    provides = ["applier"]

    def __init__(self, version: str = "v0"):
        super().__init__(version=version)

    def build_algo(self, stage, prev_stages=None):
        return super().build_algo(stage, prev_stages)

    def build_test_algo(self, stage, prev_stages=None):
        return super().build_test_algo(stage, prev_stages)

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None, mode: str = "linear"):
        """
        Build a lightweight ONNX demosaic applier.
        - Input: upstream CFA tensor [N,4,H,W]
        - Output: RGB tensor [N,3,H_even,W_even]
        - mode: "nearest" for box/nearest, "linear" for bilinear interpolation
        """
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"
        out = f"{stage}.applier"

        nodes, inits, vis = [], [], []

        # constants
        inits.append(oh.make_tensor(f"{stage}.roi_empty", TensorProto.FLOAT, [0], []))
        inits.append(oh.make_tensor(f"{stage}.scales_half2full", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
        inits.append(oh.make_tensor(f"{stage}.steps_122", TensorProto.INT64, [4], [1, 1, 2, 2]))
        inits.append(oh.make_tensor(f"{stage}.starts_R", TensorProto.INT64, [4], [0, 0, 0, 0]))
        inits.append(oh.make_tensor(f"{stage}.starts_G0", TensorProto.INT64, [4], [0, 0, 0, 1]))
        inits.append(oh.make_tensor(f"{stage}.starts_G1", TensorProto.INT64, [4], [0, 0, 1, 0]))
        inits.append(oh.make_tensor(f"{stage}.starts_B", TensorProto.INT64, [4], [0, 0, 1, 1]))
        inits.append(oh.make_tensor(f"{stage}.slice_axes", TensorProto.INT64, [4], [0, 1, 2, 3]))
        inits.append(oh.make_tensor(f"{stage}.slice_starts", TensorProto.INT64, [4], [0, 0, 0, 0]))

        # crop to even H/W
        nodes.append(oh.make_node("Shape", [cfa4], ["shape_raw"], name=f"{stage}_shape"))
        inits.append(oh.make_tensor(f"{stage}.idx0", TensorProto.INT64, [1], [0]))
        inits.append(oh.make_tensor(f"{stage}.idx1", TensorProto.INT64, [1], [1]))
        inits.append(oh.make_tensor(f"{stage}.idx2", TensorProto.INT64, [1], [2]))
        inits.append(oh.make_tensor(f"{stage}.idx3", TensorProto.INT64, [1], [3]))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx0"], ["N_dim"], axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx1"], ["C_dim"], axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx2"], ["H_dim"], axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx3"], ["W_dim"], axis=0))
        # floor even dims
        inits.append(oh.make_tensor(f"{stage}.two_f", TensorProto.FLOAT, [1], [2.0]))
        inits.append(oh.make_tensor(f"{stage}.two_i", TensorProto.INT64, [1], [2]))
        nodes.append(oh.make_node("Cast", ["H_dim"], ["H_f"], to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Cast", ["W_dim"], ["W_f"], to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Div", ["H_f", f"{stage}.two_f"], ["H_div2"]))
        nodes.append(oh.make_node("Div", ["W_f", f"{stage}.two_f"], ["W_div2"]))
        nodes.append(oh.make_node("Floor", ["H_div2"], ["Hc_f"]))
        nodes.append(oh.make_node("Floor", ["W_div2"], ["Wc_f"]))
        nodes.append(oh.make_node("Cast", ["Hc_f"], ["Hc"], to=TensorProto.INT64))
        nodes.append(oh.make_node("Cast", ["Wc_f"], ["Wc"], to=TensorProto.INT64))
        nodes.append(oh.make_node("Mul", ["Hc", f"{stage}.two_i"], ["H_even"]))
        nodes.append(oh.make_node("Mul", ["Wc", f"{stage}.two_i"], ["W_even"]))
        nodes.append(oh.make_node("Concat", ["N_dim", "C_dim", "H_even", "W_even"], ["slice_ends"], axis=0))
        nodes.append(oh.make_node("Slice", [cfa4, f"{stage}.slice_starts", "slice_ends", f"{stage}.slice_axes"], ["cfa_cropped"]))

        # split channels
        nodes.append(oh.make_node("Split", ["cfa_cropped"], [f"{stage}.cR_full", f"{stage}.cG0_full", f"{stage}.cG1_full", f"{stage}.cB_full"], axis=1))

        # half-res extraction
        nodes.append(oh.make_node("Slice", [f"{stage}.cR_full", f"{stage}.starts_R", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.R_half"]))
        nodes.append(oh.make_node("Slice", [f"{stage}.cG0_full", f"{stage}.starts_G0", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.G0_half"]))
        nodes.append(oh.make_node("Slice", [f"{stage}.cG1_full", f"{stage}.starts_G1", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.G1_half"]))
        nodes.append(oh.make_node("Slice", [f"{stage}.cB_full", f"{stage}.starts_B", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.B_half"]))

        # average G
        inits.append(oh.make_tensor(f"{stage}.half_f", TensorProto.FLOAT, [1], [0.5]))
        nodes.append(oh.make_node("Add", [f"{stage}.G0_half", f"{stage}.G1_half"], [f"{stage}.G_sum_half"]))
        nodes.append(oh.make_node("Mul", [f"{stage}.G_sum_half", f"{stage}.half_f"], [f"{stage}.G_half"]))

        # upsample to full res
        nodes.append(oh.make_node("Resize", [f"{stage}.R_half", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.R_full"], mode=mode))
        nodes.append(oh.make_node("Resize", [f"{stage}.G_half", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.G_full"], mode=mode))
        nodes.append(oh.make_node("Resize", [f"{stage}.B_half", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.B_full"], mode=mode))

        # concat to RGB
        nodes.append(oh.make_node("Concat", [f"{stage}.R_full", f"{stage}.G_full", f"{stage}.B_full"], [out], axis=1))

        # BuildResult
        outputs = {"applier": {"name": out, "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]}}
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(cfa4, type=TensorProto.FLOAT, shape=["n", 4, "h", "w"])
        return result

class DemosaicBoxV1(DemosaicBase):
    name = "demosaic_box"
    version = "v1"
    provides = ["applier"]

    def __init__(self, version: str = "v1"):
        super().__init__(version=version)
