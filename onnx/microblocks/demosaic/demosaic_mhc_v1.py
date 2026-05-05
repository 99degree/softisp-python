# demosaic_mhc.py
# Adaptive deterministic MHC demosaic applier (opset 11 compatible)
# Full replacement module: accepts upstream CFA tensor `cfa4` [N,4,H,W]
# - chops odd H/W inside the graph
# - extracts half-res planes via strided Slice
# - derives measured masks from the same channels
# - applies canonical MHC 5x5 kernels
# - provides a reusable dynamic gatekeeping helper `add_safe_mul` that
#   tiles/slices operands at runtime to avoid illegal broadcasting errors
#
# This file returns a BuildResult compatible with the repo's microblocks framework.
from __future__ import annotations
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import BuildResult
from .demosaic_base import DemosaicBase
from microblocks.registry import Registry
import numpy as np
from typing import List


def _make_5x5_init(name: str, mat: List[List[int]]):
    arr = np.array(mat, dtype=np.float32).reshape(1, 1, 5, 5)
    return oh.make_tensor(name, TensorProto.FLOAT, list(arr.shape), arr.flatten().tolist())


def add_safe_mul(nodes: list, inits: list, stage: str, A: str, B: str, out: str):
    """
    Append nodes+inits that compute safe elementwise Mul(A, B) even if channel counts differ.
    - A is treated as the target tensor (we match B to A).
    - B is tiled/sliced to match A's channel count.
    - Appends nodes that produce `out`.
    """
    # 1) shapes
    nodes.append(oh.make_node("Shape", [A], [f"{stage}.shape_{A}"], name=f"{stage}_shape_{A}"))
    nodes.append(oh.make_node("Shape", [B], [f"{stage}.shape_{B}"], name=f"{stage}_shape_{B}"))

    # 2) gather channel dims (index 1)
    gidx_name = f"{stage}.gidx1_{A}_{B}"
    inits.append(oh.make_tensor(gidx_name, TensorProto.INT64, [1], [1]))
    nodes.append(oh.make_node("Gather", [f"{stage}.shape_{A}", gidx_name], [f"{stage}.C_{A}"], name=f"{stage}_gather_C_{A}", axis=0))
    nodes.append(oh.make_node("Gather", [f"{stage}.shape_{B}", gidx_name], [f"{stage}.C_{B}"], name=f"{stage}_gather_C_{B}", axis=0))

    # 3) cast to float and compute ratio = ceil(C_A / C_B)
    nodes.append(oh.make_node("Cast", [f"{stage}.C_{A}"], [f"{stage}.C_{A}_f"], name=f"{stage}_cast_C_{A}_f", to=TensorProto.FLOAT))
    nodes.append(oh.make_node("Cast", [f"{stage}.C_{B}"], [f"{stage}.C_{B}_f"], name=f"{stage}_cast_C_{B}_f", to=TensorProto.FLOAT))
    nodes.append(oh.make_node("Div", [f"{stage}.C_{A}_f", f"{stage}.C_{B}_f"], [f"{stage}.ratio_f_{A}_{B}"], name=f"{stage}_div_ratio_{A}_{B}"))
    nodes.append(oh.make_node("Ceil", [f"{stage}.ratio_f_{A}_{B}"], [f"{stage}.ratio_ceil_{A}_{B}"], name=f"{stage}_ceil_ratio_{A}_{B}"))
    nodes.append(oh.make_node("Cast", [f"{stage}.ratio_ceil_{A}_{B}"], [f"{stage}.ratio_i_{A}_{B}"], name=f"{stage}_cast_ratio_i_{A}_{B}", to=TensorProto.INT64))

    # 4) build repeats vector [1, ratio, 1, 1] and Tile B
    one_i_name = f"{stage}.one_i_tile_{A}_{B}"
    inits.append(oh.make_tensor(one_i_name, TensorProto.INT64, [1], [1]))
    nodes.append(oh.make_node("Concat", [one_i_name, f"{stage}.ratio_i_{A}_{B}", one_i_name, one_i_name], [f"{stage}.repeats_{A}_{B}"], name=f"{stage}_concat_repeats_{A}_{B}", axis=0))
    nodes.append(oh.make_node("Tile", [B, f"{stage}.repeats_{A}_{B}"], [f"{stage}.{B}_tiled"], name=f"{stage}_tile_{B}"))

    # 5) slice tiled to exact C_A channels (defensive)
    nodes.append(oh.make_node("Concat", ["N_dim", f"{stage}.C_{A}", "H_even", "W_even"], [f"{stage}.slice_ends_target_{A}_{B}"], name=f"{stage}_concat_slice_ends_target_{A}_{B}", axis=0))
    axes_name = f"{stage}.slice_axes_full_{A}_{B}"
    starts_name = f"{stage}.slice_starts_zero_{A}_{B}"
    inits.append(oh.make_tensor(axes_name, TensorProto.INT64, [4], [0, 1, 2, 3]))
    inits.append(oh.make_tensor(starts_name, TensorProto.INT64, [4], [0, 0, 0, 0]))
    nodes.append(oh.make_node("Slice", [f"{stage}.{B}_tiled", starts_name, f"{stage}.slice_ends_target_{A}_{B}", axes_name], [f"{stage}.{B}_matched"], name=f"{stage}_slice_{B}_matched"))

    # 6) final Mul
    nodes.append(oh.make_node("Mul", [A, f"{stage}.{B}_matched"], [out], name=f"{stage}_mul_safe_{A}_{B}"))


class DemosaicMHCV1(DemosaicBase):
    name = "demosaic_mhc"
    version = "v1"
    provides = ["applier"]

    def __init__(self, version: str = "v1"):
        super().__init__(version=version)

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_algo(self, stage, prev_stages=None):
        return super().build_algo(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Build an ONNX applier subgraph that:
          - expects upstream CFA tensor `cfa4 = f"{upstream}.applier"` with shape [N,4,H,W]
          - chops odd H/W inside the graph
          - extracts half-res planes via strided Slice
          - derives measured masks from the same channels
          - applies canonical MHC kernels
          - uses add_safe_mul for robust elementwise ops
        Returns a BuildResult with output mapping "applier".
        """
        upstream = prev_stages[0] if prev_stages else stage
        cfa4 = f"{upstream}.applier"   # expected input: [N,4,H,W]
        out = f"{stage}.applier"       # output: [N,3,H_even,W_even]

        nodes = []
        inits = []
        vis = []

        # -------------------------
        # Basic constants (initializers)
        # -------------------------
        inits.append(oh.make_tensor(f"{stage}.one_f", TensorProto.FLOAT, [1], [1.0]))
        inits.append(oh.make_tensor(f"{stage}.two_f", TensorProto.FLOAT, [1], [2.0]))
        inits.append(oh.make_tensor(f"{stage}.two_i", TensorProto.INT64, [1], [2]))
        inits.append(oh.make_tensor(f"{stage}.roi_empty", TensorProto.FLOAT, [0], []))
        inits.append(oh.make_tensor(f"{stage}.scales_half2full", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
        inits.append(oh.make_tensor(f"{stage}.half_f", TensorProto.FLOAT, [1], [0.5]))
        inits.append(oh.make_tensor(f"{stage}.zero_f", TensorProto.FLOAT, [1], [0.0]))

        # steps for strided slice (1,1,2,2)
        inits.append(oh.make_tensor(f"{stage}.steps_122", TensorProto.INT64, [4], [1, 1, 2, 2]))

        # starts for per-channel half-res extraction
        inits.append(oh.make_tensor(f"{stage}.starts_R", TensorProto.INT64, [4], [0, 0, 0, 0]))
        inits.append(oh.make_tensor(f"{stage}.starts_Gtr", TensorProto.INT64, [4], [0, 0, 0, 1]))
        inits.append(oh.make_tensor(f"{stage}.starts_Gbl", TensorProto.INT64, [4], [0, 0, 1, 0]))
        inits.append(oh.make_tensor(f"{stage}.starts_B", TensorProto.INT64, [4], [0, 0, 1, 1]))

        # const 3 for reshape target
        inits.append(oh.make_tensor(f"{stage}.const_3", TensorProto.INT64, [1], [3]))

        # -------------------------
        # Canonical MHC 5x5 kernels (initializers)
        # -------------------------
        K_g = [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ]
        K_r_b = [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 5, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ]
        K_rg_h = [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        inits.extend([
            _make_5x5_init(f"{stage}.K_mhc_g_r", K_g),
            _make_5x5_init(f"{stage}.K_mhc_g_b", K_g),
            _make_5x5_init(f"{stage}.K_mhc_r_b", K_r_b),
            _make_5x5_init(f"{stage}.K_mhc_b_r", K_r_b),
            _make_5x5_init(f"{stage}.K_mhc_r_g_h", K_rg_h),
            _make_5x5_init(f"{stage}.K_mhc_r_g_v", K_rg_h),
            _make_5x5_init(f"{stage}.K_mhc_b_g_h", K_rg_h),
            _make_5x5_init(f"{stage}.K_mhc_b_g_v", K_rg_h),
        ])

        # -------------------------
        # Sobel kernels (initializers)
        # -------------------------
        sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
        inits.append(oh.make_tensor(f"{stage}.sobel_h", TensorProto.FLOAT, list(sobel_h.shape), sobel_h.flatten().tolist()))
        inits.append(oh.make_tensor(f"{stage}.sobel_v", TensorProto.FLOAT, list(sobel_v.shape), sobel_v.flatten().tolist()))

        # -------------------------
        # Dynamic crop: compute H_even, W_even and slice cfa4 to even dims
        # -------------------------
        nodes.append(oh.make_node("Shape", [cfa4], ["shape_raw"], name=f"{stage}_shape"))

        # gather dims
        inits.append(oh.make_tensor(f"{stage}.idx0", TensorProto.INT64, [1], [0]))
        inits.append(oh.make_tensor(f"{stage}.idx1", TensorProto.INT64, [1], [1]))
        inits.append(oh.make_tensor(f"{stage}.idx2", TensorProto.INT64, [1], [2]))
        inits.append(oh.make_tensor(f"{stage}.idx3", TensorProto.INT64, [1], [3]))

        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx0"], ["N_dim"], name=f"{stage}_gather_N", axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx1"], ["C_dim"], name=f"{stage}_gather_C", axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx2"], ["H_dim"], name=f"{stage}_gather_H", axis=0))
        nodes.append(oh.make_node("Gather", ["shape_raw", f"{stage}.idx3"], ["W_dim"], name=f"{stage}_gather_W", axis=0))

        # Hc = floor(H/2), Wc = floor(W/2)
        nodes.append(oh.make_node("Cast", ["H_dim"], ["H_f"], name=f"{stage}_cast_H_f", to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Cast", ["W_dim"], ["W_f"], name=f"{stage}_cast_W_f", to=TensorProto.FLOAT))
        nodes.append(oh.make_node("Div", ["H_f", f"{stage}.two_f"], ["H_div2"], name=f"{stage}_div_H2"))
        nodes.append(oh.make_node("Div", ["W_f", f"{stage}.two_f"], ["W_div2"], name=f"{stage}_div_W2"))
        nodes.append(oh.make_node("Floor", ["H_div2"], ["Hc_f"], name=f"{stage}_floor_Hc"))
        nodes.append(oh.make_node("Floor", ["W_div2"], ["Wc_f"], name=f"{stage}_floor_Wc"))
        nodes.append(oh.make_node("Cast", ["Hc_f"], ["Hc"], name=f"{stage}_cast_Hc", to=TensorProto.INT64))
        nodes.append(oh.make_node("Cast", ["Wc_f"], ["Wc"], name=f"{stage}_cast_Wc", to=TensorProto.INT64))
        nodes.append(oh.make_node("Mul", ["Hc", f"{stage}.two_i"], ["H_even"], name=f"{stage}_mul_Heven"))
        nodes.append(oh.make_node("Mul", ["Wc", f"{stage}.two_i"], ["W_even"], name=f"{stage}_mul_Weven"))

        # slice_ends = Concat([N_dim, C_dim, H_even, W_even])
        nodes.append(oh.make_node("Concat", ["N_dim", "C_dim", "H_even", "W_even"], ["slice_ends"], name=f"{stage}_concat_slice_ends", axis=0))

        # starts and axes initializers for Slice
        inits.append(oh.make_tensor(f"{stage}.slice_starts", TensorProto.INT64, [4], [0, 0, 0, 0]))
        inits.append(oh.make_tensor(f"{stage}.slice_axes", TensorProto.INT64, [4], [0, 1, 2, 3]))

        # cfa_cropped = Slice(cfa4, starts, slice_ends, axes)
        nodes.append(oh.make_node("Slice", [cfa4, f"{stage}.slice_starts", "slice_ends", f"{stage}.slice_axes"], ["cfa_cropped"], name=f"{stage}_slice_crop"))

        # -------------------------
        # Split channels from cfa_cropped (N,4,H_even,W_even)
        # -------------------------
        nodes.append(oh.make_node("Split", ["cfa_cropped"], [f"{stage}.cR_full", f"{stage}.cG0_full", f"{stage}.cG1_full", f"{stage}.cB_full"], name=f"{stage}_split_cfa", axis=1))

        # -------------------------
        # Half-res extraction via strided Slice (steps=[1,1,2,2])
        # -------------------------
        nodes.append(oh.make_node("Slice", [f"{stage}.cR_full", f"{stage}.starts_R", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.R_half"], name=f"{stage}_slice_R_half"))
        nodes.append(oh.make_node("Slice", [f"{stage}.cG0_full", f"{stage}.starts_Gtr", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.G0_half"], name=f"{stage}_slice_G0_half"))
        nodes.append(oh.make_node("Slice", [f"{stage}.cG1_full", f"{stage}.starts_Gbl", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.G1_half"], name=f"{stage}_slice_G1_half"))
        nodes.append(oh.make_node("Slice", [f"{stage}.cB_full", f"{stage}.starts_B", "slice_ends", f"{stage}.slice_axes", f"{stage}.steps_122"], [f"{stage}.B_half"], name=f"{stage}_slice_B_half"))

        # -------------------------
        # Build coarse RGB: G_coarse = 0.5*(G0_half + G1_half) and concat
        # -------------------------
        nodes.append(oh.make_node("Add", [f"{stage}.G0_half", f"{stage}.G1_half"], [f"{stage}.G_sum_half"], name=f"{stage}_add_G_half"))
        # safe multiply by scalar uses normal Mul (scalar broadcast is allowed)
        nodes.append(oh.make_node("Mul", [f"{stage}.G_sum_half", f"{stage}.half_f"], [f"{stage}.G_coarse"], name=f"{stage}_mul_G_half"))
        nodes.append(oh.make_node("Concat", [f"{stage}.R_half", f"{stage}.G_coarse", f"{stage}.B_half"], [f"{stage}.raw_rgb_coarse"], name=f"{stage}_concat_raw_coarse", axis=1))

        # -------------------------
        # Build measured masks from the full-res channels (derive boolean then cast to float)
        # -------------------------
        nodes.append(oh.make_node("Greater", [f"{stage}.cR_full", f"{stage}.zero_f"], [f"{stage}.maskR_bool"], name=f"{stage}_greater_R"))
        nodes.append(oh.make_node("Cast", [f"{stage}.maskR_bool"], [f"{stage}.maskR_f"], name=f"{stage}_cast_maskR", to=TensorProto.FLOAT))

        nodes.append(oh.make_node("Greater", [f"{stage}.cG0_full", f"{stage}.zero_f"], [f"{stage}.maskG0_bool"], name=f"{stage}_greater_G0"))
        nodes.append(oh.make_node("Cast", [f"{stage}.maskG0_bool"], [f"{stage}.maskG0_f"], name=f"{stage}_cast_maskG0", to=TensorProto.FLOAT))

        nodes.append(oh.make_node("Greater", [f"{stage}.cG1_full", f"{stage}.zero_f"], [f"{stage}.maskG1_bool"], name=f"{stage}_greater_G1"))
        nodes.append(oh.make_node("Cast", [f"{stage}.maskG1_bool"], [f"{stage}.maskG1_f"], name=f"{stage}_cast_maskG1", to=TensorProto.FLOAT))

        nodes.append(oh.make_node("Greater", [f"{stage}.cB_full", f"{stage}.zero_f"], [f"{stage}.maskB_bool"], name=f"{stage}_greater_B"))
        nodes.append(oh.make_node("Cast", [f"{stage}.maskB_bool"], [f"{stage}.maskB_f"], name=f"{stage}_cast_maskB", to=TensorProto.FLOAT))

        # Combined G mask
        nodes.append(oh.make_node("Add", [f"{stage}.maskG0_f", f"{stage}.maskG1_f"], [f"{stage}.maskG_f"], name=f"{stage}_add_maskG"))

        # -------------------------
        # Upsample half->full for kernels that operate on full grid
        # -------------------------
        nodes.append(oh.make_node("Resize", [f"{stage}.raw_rgb_coarse", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.raw_rgb_full"], name=f"{stage}_resize_raw_full", mode="linear"))
        nodes.append(oh.make_node("Resize", [f"{stage}.G_coarse", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.G_full"], name=f"{stage}_resize_G_full", mode="linear"))

        # Upsample masks to full-res (nearest)
        nodes.append(oh.make_node("Resize", [f"{stage}.maskR_f", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.maskR_full"], name=f"{stage}_resize_maskR_full", mode="nearest"))
        nodes.append(oh.make_node("Resize", [f"{stage}.maskG_f", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.maskG_full"], name=f"{stage}_resize_maskG_full", mode="nearest"))
        nodes.append(oh.make_node("Resize", [f"{stage}.maskB_f", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.maskB_full"], name=f"{stage}_resize_maskB_full", mode="nearest"))

        # -------------------------
        # Split raw_rgb_full into R_full/G_full_up/B_full for cross convs
        # -------------------------
        nodes.append(oh.make_node("Split", [f"{stage}.raw_rgb_full"], [f"{stage}.R_full", f"{stage}.G_full_up", f"{stage}.B_full"], name=f"{stage}_split_raw_full", axis=1))

        # -------------------------
        # MHC canonical convolutions on full grid
        # -------------------------
        nodes.append(oh.make_node("Conv", [f"{stage}.G_full", f"{stage}.K_mhc_g_r"], [f"{stage}.g2r"], name=f"{stage}_conv_g2r", pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Conv", [f"{stage}.G_full", f"{stage}.K_mhc_g_b"], [f"{stage}.g2b"], name=f"{stage}_conv_g2b", pads=[2, 2, 2, 2]))

        nodes.append(oh.make_node("Conv", [f"{stage}.R_full", f"{stage}.K_mhc_r_b"], [f"{stage}.r2b"], name=f"{stage}_conv_r2b", pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Conv", [f"{stage}.B_full", f"{stage}.K_mhc_b_r"], [f"{stage}.b2r"], name=f"{stage}_conv_b2r", pads=[2, 2, 2, 2]))

        nodes.append(oh.make_node("Conv", [f"{stage}.R_full", f"{stage}.K_mhc_r_g_h"], [f"{stage}.r_g_h"], name=f"{stage}_conv_r_g_h", pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Conv", [f"{stage}.R_full", f"{stage}.K_mhc_r_g_v"], [f"{stage}.r_g_v"], name=f"{stage}_conv_r_g_v", pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Conv", [f"{stage}.B_full", f"{stage}.K_mhc_b_g_h"], [f"{stage}.b_g_h"], name=f"{stage}_conv_b_g_h", pads=[2, 2, 2, 2]))
        nodes.append(oh.make_node("Conv", [f"{stage}.B_full", f"{stage}.K_mhc_b_g_v"], [f"{stage}.b_g_v"], name=f"{stage}_conv_b_g_v", pads=[2, 2, 2, 2]))

        # -------------------------
        # Directional weights from G gradients and blending
        # -------------------------
        nodes.append(oh.make_node("Conv", [f"{stage}.G_full", f"{stage}.sobel_h"], [f"{stage}.grad_h"], name=f"{stage}_conv_grad_h", pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Conv", [f"{stage}.G_full", f"{stage}.sobel_v"], [f"{stage}.grad_v"], name=f"{stage}_conv_grad_v", pads=[1, 1, 1, 1]))
        nodes.append(oh.make_node("Abs", [f"{stage}.grad_h"], [f"{stage}.mag_h"], name=f"{stage}_abs_h"))
        nodes.append(oh.make_node("Abs", [f"{stage}.grad_v"], [f"{stage}.mag_v"], name=f"{stage}_abs_v"))
        nodes.append(oh.make_node("Concat", [f"{stage}.mag_h", f"{stage}.mag_v"], [f"{stage}.mag_hv"], name=f"{stage}_concat_mag", axis=1))

        # smoothing kernel for directional weights
        smooth_k2 = np.ones((2, 1, 3, 3), dtype=np.float32) / 9.0
        inits.append(oh.make_tensor(f"{stage}.smooth_k2", TensorProto.FLOAT, list(smooth_k2.shape), smooth_k2.flatten().tolist()))
        nodes.append(oh.make_node("Conv", [f"{stage}.mag_hv", f"{stage}.smooth_k2"], [f"{stage}.dir_smooth"], name=f"{stage}_conv_dir_smooth", pads=[1, 1, 1, 1], group=2))
        nodes.append(oh.make_node("Softmax", [f"{stage}.dir_smooth"], [f"{stage}.weight_hv"], name=f"{stage}_softmax_dir", axis=1))
        nodes.append(oh.make_node("Resize", [f"{stage}.weight_hv", f"{stage}.roi_empty", f"{stage}.scales_half2full"], [f"{stage}.weight_hv_full"], name=f"{stage}_resize_weight_full", mode="linear"))
        nodes.append(oh.make_node("Split", [f"{stage}.weight_hv_full"], [f"{stage}.weight_h_full", f"{stage}.weight_v_full"], name=f"{stage}_split_weight_full", axis=1))

        # candidate stacks
        nodes.append(oh.make_node("Concat", [f"{stage}.r_g_h", f"{stage}.G_full", f"{stage}.b_g_h"], [f"{stage}.cand_h"], name=f"{stage}_concat_cand_h", axis=1))
        nodes.append(oh.make_node("Concat", [f"{stage}.r_g_v", f"{stage}.G_full", f"{stage}.b_g_v"], [f"{stage}.cand_v"], name=f"{stage}_concat_cand_v", axis=1))

        nodes.append(oh.make_node("Concat", [f"{stage}.weight_h_full", f"{stage}.weight_h_full", f"{stage}.weight_h_full"], [f"{stage}.w_h_3"], name=f"{stage}_concat_wh", axis=1))
        nodes.append(oh.make_node("Concat", [f"{stage}.weight_v_full", f"{stage}.weight_v_full", f"{stage}.weight_v_full"], [f"{stage}.w_v_3"], name=f"{stage}_concat_wv", axis=1))

        # -------------------------
        # Safe multiplications for horizontal and vertical candidates
        # -------------------------
        add_safe_mul(nodes, inits, stage, f"{stage}.cand_h", f"{stage}.w_h_3", f"{stage}.part_h")
        add_safe_mul(nodes, inits, stage, f"{stage}.cand_v", f"{stage}.w_v_3", f"{stage}.part_v")

        # blended_rgb = part_h + part_v
        nodes.append(oh.make_node("Add", [f"{stage}.part_h", f"{stage}.part_v"], [f"{stage}.blended_rgb"], name=f"{stage}_add_blend"))

        # -------------------------
        # Preserve measured samples exactly
        # Build mask_known_full = concat([maskR_full, maskG_full, maskB_full]) -> [N,3,H,W]
        # -------------------------
        nodes.append(oh.make_node("Concat", [f"{stage}.maskR_full", f"{stage}.maskG_full", f"{stage}.maskB_full"], [f"{stage}.mask_known_full_raw"], name=f"{stage}_concat_mask_known_raw", axis=1))
        nodes.append(oh.make_node("Cast", [f"{stage}.mask_known_full_raw"], [f"{stage}.mask_known_full"], name=f"{stage}_cast_mask_known", to=TensorProto.FLOAT))

        # inv_mask = 1 - mask_known_full
        nodes.append(oh.make_node("Sub", [f"{stage}.one_f", f"{stage}.mask_known_full"], [f"{stage}.inv_mask"], name=f"{stage}_sub_inv_mask"))

        # base_nearest = concat([cR_full, (cG0_full+cG1_full)/2, cB_full])
        nodes.append(oh.make_node("Add", [f"{stage}.cG0_full", f"{stage}.cG1_full"], [f"{stage}.cG_sum_full"], name=f"{stage}_add_cG_full"))
        nodes.append(oh.make_node("Mul", [f"{stage}.cG_sum_full", f"{stage}.half_f"], [f"{stage}.cG_full_avg"], name=f"{stage}_mul_cG_avg"))
        nodes.append(oh.make_node("Concat", [f"{stage}.cR_full", f"{stage}.cG_full_avg", f"{stage}.cB_full"], [f"{stage}.base_nearest"], name=f"{stage}_concat_base_nearest", axis=1))

        # Ensure mask_known_full matches blended_rgb channels (dynamic)
        add_safe_mul(nodes, inits, stage, f"{stage}.blended_rgb", f"{stage}.inv_mask", f"{stage}.blended_masked")
        add_safe_mul(nodes, inits, stage, f"{stage}.base_nearest", f"{stage}.mask_known_full", f"{stage}.base_masked")

        # final add
        nodes.append(oh.make_node("Add", [f"{stage}.blended_masked", f"{stage}.base_masked"], [out], name=f"{stage}_add_final"))

        # -------------------------
        # BuildResult and inputs/outputs
        # -------------------------
        outputs = {"applier": {"name": out, "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]}}
        result = BuildResult(outputs, nodes, inits, vis)

        # append expected inputs (keeps same calling convention)
        result.appendInput(cfa4, type=TensorProto.FLOAT, shape=["n", 4, "h", "w"])

        return result
