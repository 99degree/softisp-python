# demosaic_mhc_adaptive_opset11_fixed.py
# Adaptive deterministic MHC-like pipeline (no external padding).
# Input: raw_bayer [N,1,H,W] (single-channel RGGB Bayer)
# Output: final_rgb [N,3,H_even,W_even] where H_even = floor(H/2)*2, W_even = floor(W/2)*2
# Opset: 11, IR: 11
# Requires: numpy, onnx, onnxruntime
# pip install numpy onnx onnxruntime

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto, shape_inference, OperatorSetIdProto
import onnxruntime as ort

# -------------------------
# Test harness parameters (change to test odd/even sizes)
# -------------------------
H_full_in = 1289  # example odd height
W_full_in = 1280  # example even width
N = 1

# -------------------------
# Create a synthetic Bayer input (single-channel RGGB)
# -------------------------
def make_rggb_bayer(h, w):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = 0.5 * (yy + xx)
    img = np.zeros((h, w), dtype=np.float32)
    img[0::2, 0::2] = base[0::2, 0::2]   # R (top-left)
    img[0::2, 1::2] = base[0::2, 1::2]   # G (top-right)
    img[1::2, 0::2] = base[1::2, 0::2]   # G (bottom-left)
    img[1::2, 1::2] = base[1::2, 1::2]   # B (bottom-right)
    return img

bayer_full = make_rggb_bayer(H_full_in, W_full_in).astype(np.float32)[None, None, :, :]  # [N,1,H,W]

# -------------------------
# Build ONNX graph: initializers first (CRITICAL)
# -------------------------
nodes = []
initializers = []

# Basic float/int constants
one_f = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one_f")
two_f = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="two_f")
initializers.extend([one_f, two_f])

one_i = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one_i")
two_i = numpy_helper.from_array(np.array([2], dtype=np.int64), name="two_i")
initializers.extend([one_i, two_i])

# roi_empty initializer (empty float tensor) used by Resize (must exist before Resize nodes)
roi_empty = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi_empty")
initializers.append(roi_empty)

# Slice helpers for dynamic cropping (starts/axes)
starts_init = numpy_helper.from_array(np.array([0,0,0,0], dtype=np.int64), name="slice_starts")
axes_init = numpy_helper.from_array(np.array([0,1,2,3], dtype=np.int64), name="slice_axes")
initializers.extend([starts_init, axes_init])

# Index initializers for Gather
idx0 = numpy_helper.from_array(np.array([0], dtype=np.int64), name="idx0")
idx1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="idx1")
idx2 = numpy_helper.from_array(np.array([2], dtype=np.int64), name="idx2")
idx3 = numpy_helper.from_array(np.array([3], dtype=np.int64), name="idx3")
initializers.extend([idx0, idx1, idx2, idx3])

# smoothing kernel repeated for grouped smoothing (2 channels)
smooth_k = np.ones((2,1,3,3), dtype=np.float32) / 9.0
init_smooth2 = numpy_helper.from_array(smooth_k, name="smooth_k2")
initializers.append(init_smooth2)

# 5x5 analytic directional kernels per packed channel (grouped convs)
K_h = np.zeros((12,1,5,5), dtype=np.float32)
K_v = np.zeros((12,1,5,5), dtype=np.float32)
center_row = np.array([0.02, 0.08, 0.6, 0.08, 0.02], dtype=np.float32)
for i in range(12):
    K_h[i,0,2,:] = center_row
    K_v[i,0,:,2] = center_row
for k in (K_h, K_v):
    s = k.sum(axis=(2,3), keepdims=True)
    s[s == 0] = 1.0
    k[:] = k / s
init_K_h = numpy_helper.from_array(K_h.astype(np.float32), name="K_h")
init_K_v = numpy_helper.from_array(K_v.astype(np.float32), name="K_v")
initializers.extend([init_K_h, init_K_v])

# Cross-channel grouped conv kernels: groups=3, in_per_group=1, out_channels=9
out_channels = 9
in_per_group = 1
kh, kw = 3, 3
K_cross = np.zeros((out_channels, in_per_group, kh, kw), dtype=np.float32)
delta_kernel = np.array([[0.0, -0.2, 0.0], [-0.2, 1.0, -0.2], [0.0, -0.2, 0.0]], dtype=np.float32)
for oc in range(out_channels):
    K_cross[oc, 0, :, :] = delta_kernel
init_K_cross = numpy_helper.from_array(K_cross.astype(np.float32), name="K_cross")
initializers.append(init_K_cross)

# separable refinement kernels (3 output channels)
sep_h = np.array([0.25, 0.5, 0.25], dtype=np.float32).reshape(1,1,1,3)
sep_v = sep_h.reshape(1,1,3,1)
init_sep_h = numpy_helper.from_array(np.tile(sep_h, (3,1,1,1)), name="sep_h")
init_sep_v = numpy_helper.from_array(np.tile(sep_v, (3,1,1,1)), name="sep_v")
initializers.extend([init_sep_h, init_sep_v])

# phase 2x2 one-hot RGGB as initializer (NHWC: [1,2,2,4])
phase_nhwc = np.zeros((1,2,2,4), dtype=np.float32)
phase_nhwc[0,0,0,0] = 1.0  # p00 R
phase_nhwc[0,0,1,1] = 1.0  # p01 G
phase_nhwc[0,1,0,2] = 1.0  # p10 G
phase_nhwc[0,1,1,3] = 1.0  # p11 B
init_phase = numpy_helper.from_array(phase_nhwc, name="phase_nhwc_const")
initializers.append(init_phase)

# Sobel kernels for gradients (must be initializers before nodes that use them)
sobel_h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32).reshape(1,1,3,3)
sobel_v = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32).reshape(1,1,3,3)
init_sobel_h = numpy_helper.from_array(sobel_h, name="sobel_h")
init_sobel_v = numpy_helper.from_array(sobel_v, name="sobel_v")
initializers.extend([init_sobel_h, init_sobel_v])

# scales_full constant (2x up)
scales_full_const = numpy_helper.from_array(np.array([1.0,1.0,2.0,2.0], dtype=np.float32), name="scales_full_const")
initializers.append(scales_full_const)

# const 3 for target shape
const_3 = numpy_helper.from_array(np.array([3], dtype=np.int64), name="const_3")
initializers.append(const_3)

# half scalar for averaging greens
half_f = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half_f")
initializers.append(half_f)

# -------------------------
# Graph input (dynamic H,W)
# -------------------------
input_bayer = helper.make_tensor_value_info("raw_bayer", TensorProto.FLOAT, ["N", 1, "H", "W"])

# -------------------------
# Dynamic cropping fragment (compute Hc,Wc and crop to even dims)
# -------------------------
nodes.append(helper.make_node("Shape", ["raw_bayer"], ["shape_raw"], name="shape_raw"))
nodes.append(helper.make_node("Gather", ["shape_raw", idx0.name], ["N_dim"], name="gather_N", axis=0))
nodes.append(helper.make_node("Gather", ["shape_raw", idx1.name], ["C_dim"], name="gather_C", axis=0))
nodes.append(helper.make_node("Gather", ["shape_raw", idx2.name], ["H_dim"], name="gather_H", axis=0))
nodes.append(helper.make_node("Gather", ["shape_raw", idx3.name], ["W_dim"], name="gather_W", axis=0))

nodes.append(helper.make_node("Cast", ["H_dim"], ["H_f"], name="cast_H_f", to=TensorProto.FLOAT))
nodes.append(helper.make_node("Cast", ["W_dim"], ["W_f"], name="cast_W_f", to=TensorProto.FLOAT))

nodes.append(helper.make_node("Div", ["H_f", two_f.name], ["H_div2"], name="div_H2"))
nodes.append(helper.make_node("Div", ["W_f", two_f.name], ["W_div2"], name="div_W2"))
nodes.append(helper.make_node("Floor", ["H_div2"], ["Hc_f"], name="floor_Hc"))
nodes.append(helper.make_node("Floor", ["W_div2"], ["Wc_f"], name="floor_Wc"))

nodes.append(helper.make_node("Cast", ["Hc_f"], ["Hc"], name="cast_Hc_i", to=TensorProto.INT64))
nodes.append(helper.make_node("Cast", ["Wc_f"], ["Wc"], name="cast_Wc_i", to=TensorProto.INT64))

nodes.append(helper.make_node("Mul", ["Hc", two_i.name], ["H_even"], name="mul_Heven"))
nodes.append(helper.make_node("Mul", ["Wc", two_i.name], ["W_even"], name="mul_Weven"))

nodes.append(helper.make_node("Concat", ["N_dim","C_dim","H_even","W_even"], ["slice_ends"], name="concat_slice_ends", axis=0))
nodes.append(helper.make_node("Slice", ["raw_bayer", starts_init.name, "slice_ends", axes_init.name], ["bayer_cropped"], name="slice_crop"))

# dynamic scales for phase resize: scales_phase = [1,1, Hc/2, Wc/2]
nodes.append(helper.make_node("Cast", ["Hc"], ["Hc_f2"], name="cast_Hc_f2", to=TensorProto.FLOAT))
nodes.append(helper.make_node("Cast", ["Wc"], ["Wc_f2"], name="cast_Wc_f2", to=TensorProto.FLOAT))
nodes.append(helper.make_node("Div", ["Hc_f2", two_f.name], ["sc_h_phase"], name="div_Hc2"))
nodes.append(helper.make_node("Div", ["Wc_f2", two_f.name], ["sc_w_phase"], name="div_Wc2"))
nodes.append(helper.make_node("Concat", ["one_f","one_f","sc_h_phase","sc_w_phase"], ["scales_phase_dyn"], name="concat_scales_phase", axis=0))

# dynamic target shape for Reshape: [N,3,H_even,W_even]
nodes.append(helper.make_node("Concat", ["N_dim", "const_3", "H_even", "W_even"], ["target_shape_dyn"], name="concat_target_shape", axis=0))

# -------------------------
# Extract coarse sub-samples via strided Slice (step=2)
# -------------------------
steps_122 = numpy_helper.from_array(np.array([1,1,2,2], dtype=np.int64), name="steps_122")
initializers.append(steps_122)

starts_R = numpy_helper.from_array(np.array([0,0,0,0], dtype=np.int64), name="starts_R")
starts_Gtr = numpy_helper.from_array(np.array([0,0,0,1], dtype=np.int64), name="starts_Gtr")
starts_Gbl = numpy_helper.from_array(np.array([0,0,1,0], dtype=np.int64), name="starts_Gbl")
starts_B = numpy_helper.from_array(np.array([0,0,1,1], dtype=np.int64), name="starts_B")
initializers.extend([starts_R, starts_Gtr, starts_Gbl, starts_B])

nodes.append(helper.make_node("Slice", ["bayer_cropped", starts_R.name, "slice_ends", axes_init.name, steps_122.name], ["R_coarse_raw"], name="slice_R"))
nodes.append(helper.make_node("Slice", ["bayer_cropped", starts_Gtr.name, "slice_ends", axes_init.name, steps_122.name], ["G_tr_raw"], name="slice_Gtr"))
nodes.append(helper.make_node("Slice", ["bayer_cropped", starts_Gbl.name, "slice_ends", axes_init.name, steps_122.name], ["G_bl_raw"], name="slice_Gbl"))
nodes.append(helper.make_node("Slice", ["bayer_cropped", starts_B.name, "slice_ends", axes_init.name, steps_122.name], ["B_coarse_raw"], name="slice_B"))

# G_coarse = 0.5*(G_tr + G_bl)
nodes.append(helper.make_node("Add", ["G_tr_raw", "G_bl_raw"], ["G_sum"], name="add_Gs"))
nodes.append(helper.make_node("Mul", ["G_sum", half_f.name], ["G_coarse"], name="mul_Ghalf"))

# concat raw_rgb_coarse = [R,G,B]
nodes.append(helper.make_node("Concat", ["R_coarse_raw","G_coarse","B_coarse_raw"], ["raw_rgb_coarse"], name="concat_raw_coarse", axis=1))

# -------------------------
# Phase masks: transpose and resize to coarse using dynamic scales
# -------------------------
nodes.append(helper.make_node("Transpose", ["phase_nhwc_const"], ["phase_nchw"], name="transpose_phase", perm=[0,3,1,2]))
nodes.append(helper.make_node("Split", ["phase_nchw"], ["p00_2x2","p01_2x2","p10_2x2","p11_2x2"],
                              name="split_phase", axis=1, split=[1,1,1,1]))
nodes.append(helper.make_node("Resize", ["p00_2x2", "roi_empty", "scales_phase_dyn"], ["p00_coarse"], name="resize_p00_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_2x2", "roi_empty", "scales_phase_dyn"], ["p01_coarse"], name="resize_p01_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_2x2", "roi_empty", "scales_phase_dyn"], ["p10_coarse"], name="resize_p10_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_2x2", "roi_empty", "scales_phase_dyn"], ["p11_coarse"], name="resize_p11_to_coarse", mode="nearest"))

# -------------------------
# Pack masked per-phase channels: multiply raw_rgb_coarse channels by phase masks and concat -> [N,12,Hc,Wc]
# -------------------------
nodes.append(helper.make_node("Split", ["raw_rgb_coarse"], ["raw_R","raw_G","raw_B"], name="split_raw_rgb_coarse", axis=1, split=[1,1,1]))

mul_outputs = []
for c in ["R","G","B"]:
    for p in ["p00_coarse","p01_coarse","p10_coarse","p11_coarse"]:
        out = f"m_{c}_{p}"
        nodes.append(helper.make_node("Mul", [f"raw_{c}", p], [out], name=f"mul_{c}_{p}"))
        mul_outputs.append(out)
nodes.append(helper.make_node("Concat", mul_outputs, ["packed_masked"], name="concat_packed", axis=1))

# -------------------------
# Directional grouped 5x5 convs (groups=12)
# -------------------------
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_h.name], ["conv_packed_h"], name="conv_packed_h",
                              pads=[2,2,2,2], strides=[1,1], group=12))
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_v.name], ["conv_packed_v"], name="conv_packed_v",
                              pads=[2,2,2,2], strides=[1,1], group=12))

nodes.append(helper.make_node("Split", ["conv_packed_h"], ["R_block_h","G_block_h","B_block_h"], name="split_h", axis=1, split=[4,4,4]))
nodes.append(helper.make_node("Split", ["conv_packed_v"], ["R_block_v","G_block_v","B_block_v"], name="split_v", axis=1, split=[4,4,4]))

nodes.append(helper.make_node("DepthToSpace", ["R_block_h"], ["up_R_h"], name="d2s_R_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_h"], ["up_G_h"], name="d2s_G_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_h"], ["up_B_h"], name="d2s_B_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["R_block_v"], ["up_R_v"], name="d2s_R_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_v"], ["up_G_v"], name="d2s_G_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_v"], ["up_B_v"], name="d2s_B_v", blocksize=2))

# -------------------------
# Directional weights from green gradients (coarse)
# -------------------------
nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_h.name], ["grad_h"], name="conv_grad_h", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_v.name], ["grad_v"], name="conv_grad_v", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Abs", ["grad_h"], ["mag_h"], name="abs_h"))
nodes.append(helper.make_node("Abs", ["grad_v"], ["mag_v"], name="abs_v"))
nodes.append(helper.make_node("Concat", ["mag_h", "mag_v"], ["mag_hv"], name="concat_mag", axis=1))
nodes.append(helper.make_node("Conv", ["mag_hv", init_smooth2.name], ["dir_smooth"], name="conv_dir_smooth", pads=[1,1,1,1], strides=[1,1], group=2))
nodes.append(helper.make_node("Softmax", ["dir_smooth"], ["weight_hv"], name="softmax_dir", axis=1))

nodes.append(helper.make_node("Resize", ["weight_hv", "roi_empty", scales_full_const.name], ["weight_hv_full"], name="resize_weight_full", mode="linear"))
nodes.append(helper.make_node("Split", ["weight_hv_full"], ["weight_h_full","weight_v_full"], name="split_weight_full", axis=1, split=[1,1]))

# -------------------------
# Build up_h_rgb and up_v_rgb and blend
# -------------------------
nodes.append(helper.make_node("Concat", ["up_R_h","up_G_h","up_B_h"], ["up_h_rgb"], name="concat_up_h", axis=1))
nodes.append(helper.make_node("Concat", ["up_R_v","up_G_v","up_B_v"], ["up_v_rgb"], name="concat_up_v", axis=1))
nodes.append(helper.make_node("Concat", ["weight_h_full","weight_h_full","weight_h_full"], ["w_h_3"], name="concat_wh", axis=1))
nodes.append(helper.make_node("Concat", ["weight_v_full","weight_v_full","weight_v_full"], ["w_v_3"], name="concat_wv", axis=1))
nodes.append(helper.make_node("Mul", ["up_h_rgb", "w_h_3"], ["part_h"], name="mul_up_h"))
nodes.append(helper.make_node("Mul", ["up_v_rgb", "w_v_3"], ["part_v"], name="mul_up_v"))
nodes.append(helper.make_node("Add", ["part_h", "part_v"], ["blended_rgb"], name="add_blend"))

# -------------------------
# Cross-channel grouped conv -> cross_out (groups=3)
# -------------------------
nodes.append(helper.make_node("Conv", ["raw_rgb_coarse", init_K_cross.name], ["cross_out"], name="conv_cross", pads=[1,1,1,1], strides=[1,1], group=3))
nodes.append(helper.make_node("Split", ["cross_out"], ["c0","c1","c2","c3","c4","c5","c6","c7","c8"], name="split_cross", axis=1, split=[1]*9))

nodes.append(helper.make_node("Resize", ["c0", "roi_empty", scales_full_const.name], ["delta_R_full"], name="resize_delta_R", mode="linear"))
nodes.append(helper.make_node("Resize", ["c6", "roi_empty", scales_full_const.name], ["delta_B_full"], name="resize_delta_B", mode="linear"))
nodes.append(helper.make_node("Resize", ["c2", "roi_empty", scales_full_const.name], ["conf_R_full"], name="resize_conf_R", mode="linear"))
nodes.append(helper.make_node("Resize", ["c8", "roi_empty", scales_full_const.name], ["conf_B_full"], name="resize_conf_B", mode="linear"))
nodes.append(helper.make_node("Sigmoid", ["conf_R_full"], ["conf_R_full_s"], name="sig_conf_R"))
nodes.append(helper.make_node("Sigmoid", ["conf_B_full"], ["conf_B_full_s"], name="sig_conf_B"))

# -------------------------
# Apply deltas and confidences to blended channels
# -------------------------
nodes.append(helper.make_node("Split", ["blended_rgb"], ["blend_R","blend_G","blend_B"], name="split_blend", axis=1, split=[1,1,1]))
nodes.append(helper.make_node("Add", ["blend_R", "delta_R_full"], ["corr_R_raw"], name="add_delta_R"))
nodes.append(helper.make_node("Add", ["blend_B", "delta_B_full"], ["corr_B_raw"], name="add_delta_B"))

nodes.append(helper.make_node("Sub", ["one_f", "conf_R_full_s"], ["inv_conf_R"], name="sub_inv_conf_R"))
nodes.append(helper.make_node("Mul", ["corr_R_raw", "conf_R_full_s"], ["corr_R_p"], name="mul_corrR_conf"))
nodes.append(helper.make_node("Mul", ["blend_R", "inv_conf_R"], ["blend_R_p"], name="mul_blendR_invconf"))
nodes.append(helper.make_node("Add", ["corr_R_p", "blend_R_p"], ["corr_R"], name="add_corrR"))

nodes.append(helper.make_node("Sub", ["one_f", "conf_B_full_s"], ["inv_conf_B"], name="sub_inv_conf_B"))
nodes.append(helper.make_node("Mul", ["corr_B_raw", "conf_B_full_s"], ["corr_B_p"], name="mul_corrB_conf"))
nodes.append(helper.make_node("Mul", ["blend_B", "inv_conf_B"], ["blend_B_p"], name="mul_blendB_invconf"))
nodes.append(helper.make_node("Add", ["corr_B_p", "blend_B_p"], ["corr_B"], name="add_corrB"))

nodes.append(helper.make_node("Concat", ["corr_R","blend_G","corr_B"], ["candidate_rgb"], name="concat_candidate", axis=1))

# -------------------------
# Build mask_known_full from phase masks at full res (nearest)
# -------------------------
nodes.append(helper.make_node("Resize", ["p00_coarse", "roi_empty", scales_full_const.name], ["p00_full"], name="resize_p00_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_coarse", "roi_empty", scales_full_const.name], ["p01_full"], name="resize_p01_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_coarse", "roi_empty", scales_full_const.name], ["p10_full"], name="resize_p10_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_coarse", "roi_empty", scales_full_const.name], ["p11_full"], name="resize_p11_full", mode="nearest"))

nodes.append(helper.make_node("Add", ["p01_full", "p10_full"], ["G_full"], name="add_g_full"))
nodes.append(helper.make_node("Identity", ["p00_full"], ["R_full"], name="id_r_full"))
nodes.append(helper.make_node("Identity", ["p11_full"], ["B_full"], name="id_b_full"))
nodes.append(helper.make_node("Concat", ["R_full","G_full","B_full"], ["mask_known_full_raw"], name="concat_mask_known_raw", axis=1))
nodes.append(helper.make_node("Cast", ["mask_known_full_raw"], ["mask_known_full"], name="cast_mask_float", to=TensorProto.FLOAT))

# -------------------------
# Resize raw_rgb_coarse -> base_nearest (nearest) to full resolution
# -------------------------
nodes.append(helper.make_node("Resize", ["raw_rgb_coarse", "roi_empty", scales_full_const.name], ["base_nearest_raw"], name="resize_base", mode="nearest"))

# Reshape candidate and base to dynamic target_shape_dyn
nodes.append(helper.make_node("Reshape", ["candidate_rgb", "target_shape_dyn"], ["candidate_rgb_r"], name="reshape_candidate"))
nodes.append(helper.make_node("Reshape", ["base_nearest_raw", "target_shape_dyn"], ["base_nearest"], name="reshape_base"))

# -------------------------
# Masked full-res separable refinement applied only to non-measured pixels
# -------------------------
nodes.append(helper.make_node("Conv", ["candidate_rgb_r", init_sep_h.name], ["sep_h_out"], name="sep_h", pads=[0,1,0,1], strides=[1,1], group=3))
nodes.append(helper.make_node("Conv", ["sep_h_out", init_sep_v.name], ["refined_rgb"], name="sep_v", pads=[1,0,1,0], strides=[1,1], group=3))

nodes.append(helper.make_node("Reshape", ["mask_known_full", "target_shape_dyn"], ["mask_known_full_r"], name="reshape_mask"))
nodes.append(helper.make_node("Sub", ["one_f", "mask_known_full_r"], ["inv_mask"], name="sub_inv_mask"))
nodes.append(helper.make_node("Mul", ["refined_rgb", "inv_mask"], ["refined_masked"], name="mul_refined_mask"))
nodes.append(helper.make_node("Mul", ["base_nearest", "mask_known_full_r"], ["base_masked"], name="mul_base_mask"))
nodes.append(helper.make_node("Add", ["refined_masked", "base_masked"], ["final_rgb"], name="add_final"))

# -------------------------
# Output
# -------------------------
output_info = helper.make_tensor_value_info("final_rgb", TensorProto.FLOAT, ["N", 3, "H_even", "W_even"])

# Build graph
graph = helper.make_graph(
    nodes,
    "demosaic_mhc_adaptive_opset11_fixed",
    inputs=[input_bayer],
    outputs=[output_info],
    initializer=initializers
)

# Create model and set opset_import to 11 and IR version to 11
model = helper.make_model(graph, producer_name="demosaic_mhc_adaptive_fixed")
if len(model.opset_import) == 0:
    ops = OperatorSetIdProto()
    ops.version = 11
    model.opset_import.extend([ops])
else:
    model.opset_import[0].version = 11
model.ir_version = 11

# Infer shapes and validate
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

# Save model
onnx.save(model, "demosaic_mhc_adaptive_opset11_fixed.onnx")
print("Saved ONNX model: demosaic_mhc_adaptive_opset11_fixed.onnx")

# -------------------------
# Run the model with onnxruntime (test)
# -------------------------
sess = ort.InferenceSession("demosaic_mhc_adaptive_opset11_fixed.onnx", providers=["CPUExecutionProvider"])

inputs = {
    "raw_bayer": bayer_full.astype(np.float32)
}

out = sess.run(["final_rgb"], inputs)
final_rgb = out[0]  # [N,3,H_even,W_even]

print("final_rgb.shape:", final_rgb.shape)
print("min/max final:", final_rgb.min(), final_rgb.max())
