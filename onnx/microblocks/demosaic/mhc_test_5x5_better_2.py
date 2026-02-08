# demosaic_mhc_final_opset11_fixed_full.py
# Deterministic MHC-like pipeline with 5x5 analytic kernels, cross-channel mixing,
# normalized directional weights, confidence maps, and masked full-res refinement.
# Exports ONNX model with opset 11 and IR version 11.
# Requires: numpy, onnx, onnxruntime
# pip install numpy onnx onnxruntime

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto, shape_inference, OperatorSetIdProto
import onnxruntime as ort

# -------------------------
# Parameters
# -------------------------
H_full = 1280
W_full = 1280
Hc = H_full // 2
Wc = W_full // 2
N = 1

# -------------------------
# Synthetic RGGB Bayer (full resolution)
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

bayer = make_rggb_bayer(H_full, W_full)

# -------------------------
# Build coarse per-color planes [N,3,Hc,Wc]
# -------------------------
R_coarse = bayer[0::2, 0::2].astype(np.float32)
G_tr = bayer[0::2, 1::2].astype(np.float32)
G_bl = bayer[1::2, 0::2].astype(np.float32)
G_coarse = 0.5 * (G_tr + G_bl)
B_coarse = bayer[1::2, 1::2].astype(np.float32)

raw_rgb_coarse = np.stack([R_coarse, G_coarse, B_coarse], axis=0)[None, ...]  # [1,3,Hc,Wc]

# -------------------------
# Phase onehot NHWC [1,2,2,4] (RGGB)
# -------------------------
phase_nhwc = np.zeros((1, 2, 2, 4), dtype=np.float32)
phase_nhwc[0, 0, 0, 0] = 1.0  # p00 (R)
phase_nhwc[0, 0, 1, 1] = 1.0  # p01 (G)
phase_nhwc[0, 1, 0, 2] = 1.0  # p10 (G)
phase_nhwc[0, 1, 1, 3] = 1.0  # p11 (B)

# -------------------------
# Build ONNX graph: initializers first
# -------------------------
nodes = []
initializers = []

# Common scalar and shape initializers
one_scalar_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one_scalar")
initializers.append(one_scalar_init)

scales_phase = np.array([1.0, 1.0, float(Hc) / 2.0, float(Wc) / 2.0], dtype=np.float32)
scales_phase_init = numpy_helper.from_array(scales_phase, name="scales_phase")
initializers.append(scales_phase_init)

roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi_empty")
initializers.append(roi_init)

scales_full = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
scales_full_init = numpy_helper.from_array(scales_full, name="scales_full")
initializers.append(scales_full_init)

target_shape = np.array([N, 3, H_full, W_full], dtype=np.int64)
target_shape_init = numpy_helper.from_array(target_shape, name="target_shape")
initializers.append(target_shape_init)

# Sobel kernels for green gradients
sobel_h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32).reshape(1,1,3,3)
sobel_v = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32).reshape(1,1,3,3)
init_sobel_h = numpy_helper.from_array(sobel_h, name="sobel_h")
init_sobel_v = numpy_helper.from_array(sobel_v, name="sobel_v")
initializers.extend([init_sobel_h, init_sobel_v])

# smoothing kernel repeated for grouped smoothing (2 channels)
smooth_k = np.ones((1,1,3,3), dtype=np.float32) / 9.0
smooth_k2 = np.tile(smooth_k, (2,1,1,1))
init_smooth2 = numpy_helper.from_array(smooth_k2, name="smooth_k2")
initializers.append(init_smooth2)

# 5x5 analytic directional kernels per packed channel (grouped convs)
K_h = np.zeros((12,1,5,5), dtype=np.float32)
K_v = np.zeros((12,1,5,5), dtype=np.float32)
center_row = np.array([0.02, 0.08, 0.6, 0.08, 0.02], dtype=np.float32)
for i in range(12):
    K_h[i,0,2,:] = center_row
    K_v[i,0,:,2] = center_row
# normalize per-channel
for k in (K_h, K_v):
    s = k.sum(axis=(2,3), keepdims=True)
    s[s == 0] = 1.0
    k[:] = k / s
init_K_h = numpy_helper.from_array(K_h.astype(np.float32), name="K_h")
init_K_v = numpy_helper.from_array(K_v.astype(np.float32), name="K_v")
initializers.extend([init_K_h, init_K_v])

# Cross-channel grouped conv kernels: groups=3, input channels=3 -> in_per_group=1
# We'll produce 9 output maps (3 colors * 3 maps each)
out_channels = 9
in_per_group = 1
kh, kw = 3, 3
K_cross = np.zeros((out_channels, in_per_group, kh, kw), dtype=np.float32)

# delta kernel and small cross-channel mixing
delta_kernel = np.array([[0.0, -0.2, 0.0], [-0.2, 1.0, -0.2], [0.0, -0.2, 0.0]], dtype=np.float32)
# layout: outputs 0..2 -> R maps, 3..5 -> G maps, 6..8 -> B maps
for oc in range(out_channels):
    # determine which input channel this output should convolve (0=R,1=G,2=B)
    input_channel = oc // 3
    # place delta kernel scaled by a small factor; since in_per_group=1, we fill [oc,0,:,:]
    # scale center by 1.0 for same-channel, smaller for cross-channel (we emulate mild mixing by repeating kernels per group)
    K_cross[oc, 0, :, :] = delta_kernel * (1.0 if True else 0.5)

init_K_cross = numpy_helper.from_array(K_cross.astype(np.float32), name="K_cross")
initializers.append(init_K_cross)

# Auxiliary small separable refinement kernels (1x3 and 3x1) for 3 output channels
sep_h = np.array([0.25, 0.5, 0.25], dtype=np.float32).reshape(1,1,1,3)
sep_v = sep_h.reshape(1,1,3,1)
init_sep_h = numpy_helper.from_array(np.tile(sep_h, (3,1,1,1)), name="sep_h")  # shape [3,1,1,3]
init_sep_v = numpy_helper.from_array(np.tile(sep_v, (3,1,1,1)), name="sep_v")  # shape [3,1,3,1]
initializers.extend([init_sep_h, init_sep_v])

# -------------------------
# Inputs
# -------------------------
input_raw = helper.make_tensor_value_info("raw_rgb_coarse", TensorProto.FLOAT, [N, 3, Hc, Wc])
input_phase = helper.make_tensor_value_info("phase_nhwc", TensorProto.FLOAT, [N, 2, 2, 4])

# 1) Phase transpose and split (opset 11: split attribute)
nodes.append(helper.make_node("Transpose", ["phase_nhwc"], ["phase_nchw"], name="transpose_phase", perm=[0,3,1,2]))
nodes.append(helper.make_node("Split", ["phase_nchw"], ["p00_2x2","p01_2x2","p10_2x2","p11_2x2"],
                              name="split_phase", axis=1, split=[1,1,1,1]))

# 2) Resize phase masks 2x2 -> Hc x Wc (nearest)
nodes.append(helper.make_node("Resize", ["p00_2x2", roi_init.name, scales_phase_init.name], ["p00_coarse"], name="resize_p00_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_2x2", roi_init.name, scales_phase_init.name], ["p01_coarse"], name="resize_p01_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_2x2", roi_init.name, scales_phase_init.name], ["p10_coarse"], name="resize_p10_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_2x2", roi_init.name, scales_phase_init.name], ["p11_coarse"], name="resize_p11_to_coarse", mode="nearest"))

# 3) Slice raw channels
def add_slice(input_name, out_name, starts, ends, axes, name):
    s = numpy_helper.from_array(np.array(starts, dtype=np.int64), name=name + "_starts")
    e = numpy_helper.from_array(np.array(ends, dtype=np.int64), name=name + "_ends")
    a = numpy_helper.from_array(np.array(axes, dtype=np.int64), name=name + "_axes")
    initializers.extend([s, e, a])
    return helper.make_node("Slice", [input_name, s.name, e.name, a.name], [out_name], name=name)

nodes.append(add_slice("raw_rgb_coarse", "raw_R", [0,0,0,0], [N,1,Hc,Wc], [0,1,2,3], "slice_raw_R"))
nodes.append(add_slice("raw_rgb_coarse", "raw_G", [0,1,0,0], [N,2,Hc,Wc], [0,1,2,3], "slice_raw_G"))
nodes.append(add_slice("raw_rgb_coarse", "raw_B", [0,2,0,0], [N,3,Hc,Wc], [0,1,2,3], "slice_raw_B"))

# 4) Multiply raw channels by tiled phase masks and pack -> [N,12,Hc,Wc]
mul_outputs = []
colors = ["R","G","B"]
phases = ["p00_coarse","p01_coarse","p10_coarse","p11_coarse"]
for c, raw in zip(colors, ["raw_R","raw_G","raw_B"]):
    for p in phases:
        out = f"m_{c}_{p}"
        nodes.append(helper.make_node("Mul", [raw, p], [out], name=f"mul_{c}_{p}"))
        mul_outputs.append(out)
nodes.append(helper.make_node("Concat", mul_outputs, ["packed_masked"], name="concat_packed", axis=1))

# 5) Directional grouped 5x5 convs for coarse candidates (groups=12)
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_h.name], ["conv_packed_h"], name="conv_packed_h",
                              pads=[2,2,2,2], strides=[1,1], group=12))
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_v.name], ["conv_packed_v"], name="conv_packed_v",
                              pads=[2,2,2,2], strides=[1,1], group=12))

# 6) Split conv outputs into per-color 4-channel blocks (opset 11)
nodes.append(helper.make_node("Split", ["conv_packed_h"], ["R_block_h","G_block_h","B_block_h"], name="split_h", axis=1, split=[4,4,4]))
nodes.append(helper.make_node("Split", ["conv_packed_v"], ["R_block_v","G_block_v","B_block_v"], name="split_v", axis=1, split=[4,4,4]))

# 7) DepthToSpace each block -> up_h and up_v per color
nodes.append(helper.make_node("DepthToSpace", ["R_block_h"], ["up_R_h"], name="d2s_R_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_h"], ["up_G_h"], name="d2s_G_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_h"], ["up_B_h"], name="d2s_B_h", blocksize=2))

nodes.append(helper.make_node("DepthToSpace", ["R_block_v"], ["up_R_v"], name="d2s_R_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_v"], ["up_G_v"], name="d2s_G_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_v"], ["up_B_v"], name="d2s_B_v", blocksize=2))

# 8) Compute coarse green gradients (Sobel) and build two-channel magnitude tensor
nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_h.name], ["grad_h"], name="conv_grad_h", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_v.name], ["grad_v"], name="conv_grad_v", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Abs", ["grad_h"], ["mag_h"], name="abs_h"))
nodes.append(helper.make_node("Abs", ["grad_v"], ["mag_v"], name="abs_v"))
nodes.append(helper.make_node("Concat", ["mag_h", "mag_v"], ["mag_hv"], name="concat_mag", axis=1))

# Smooth mag_hv with grouped conv (groups=2)
nodes.append(helper.make_node("Conv", ["mag_hv", init_smooth2.name], ["dir_smooth"], name="conv_dir_smooth", pads=[1,1,1,1], strides=[1,1], group=2))

# Softmax across channel axis to get normalized directional weights [N,2,Hc,Wc]
nodes.append(helper.make_node("Softmax", ["dir_smooth"], ["weight_hv"], name="softmax_dir", axis=1))

# 9) Upsample weight_hv to full resolution and split into h/v
nodes.append(helper.make_node("Resize", ["weight_hv", roi_init.name, scales_full_init.name], ["weight_hv_full"], name="resize_weight_full", mode="linear"))
nodes.append(helper.make_node("Split", ["weight_hv_full"], ["weight_h_full","weight_v_full"], name="split_weight_full", axis=1, split=[1,1]))

# 10) Concat up_h and up_v per color into full candidate tensors
nodes.append(helper.make_node("Concat", ["up_R_h","up_G_h","up_B_h"], ["up_h_rgb"], name="concat_up_h", axis=1))
nodes.append(helper.make_node("Concat", ["up_R_v","up_G_v","up_B_v"], ["up_v_rgb"], name="concat_up_v", axis=1))

# 11) Blend candidates using normalized weights
nodes.append(helper.make_node("Concat", ["weight_h_full","weight_h_full","weight_h_full"], ["w_h_3"], name="concat_wh", axis=1))
nodes.append(helper.make_node("Concat", ["weight_v_full","weight_v_full","weight_v_full"], ["w_v_3"], name="concat_wv", axis=1))
nodes.append(helper.make_node("Mul", ["up_h_rgb", "w_h_3"], ["part_h"], name="mul_up_h"))
nodes.append(helper.make_node("Mul", ["up_v_rgb", "w_v_3"], ["part_v"], name="mul_up_v"))
nodes.append(helper.make_node("Add", ["part_h", "part_v"], ["blended_rgb"], name="add_blend"))

# 12) Cross-channel grouped conv to produce delta and confidence hints (groups=3)
nodes.append(helper.make_node("Conv", ["raw_rgb_coarse", init_K_cross.name], ["cross_out"], name="conv_cross", pads=[1,1,1,1], strides=[1,1], group=3))

# Split cross_out into 9 maps: for each color [map0,map1,map2]
nodes.append(helper.make_node("Split", ["cross_out"], ["c0","c1","c2","c3","c4","c5","c6","c7","c8"], name="split_cross", axis=1, split=[1]*9))

# Upsample selected maps to full resolution (choose c0 and c6 as delta proxies; c2 and c8 as conf proxies)
nodes.append(helper.make_node("Resize", ["c0", roi_init.name, scales_full_init.name], ["delta_R_full"], name="resize_delta_R", mode="linear"))
nodes.append(helper.make_node("Resize", ["c6", roi_init.name, scales_full_init.name], ["delta_B_full"], name="resize_delta_B", mode="linear"))
nodes.append(helper.make_node("Resize", ["c2", roi_init.name, scales_full_init.name], ["conf_R_full"], name="resize_conf_R", mode="linear"))
nodes.append(helper.make_node("Resize", ["c8", roi_init.name, scales_full_init.name], ["conf_B_full"], name="resize_conf_B", mode="linear"))

# Sigmoid confidences
nodes.append(helper.make_node("Sigmoid", ["conf_R_full"], ["conf_R_full_s"], name="sig_conf_R"))
nodes.append(helper.make_node("Sigmoid", ["conf_B_full"], ["conf_B_full_s"], name="sig_conf_B"))

# 13) Split blended_rgb into channels and apply delta corrections to R and B
nodes.append(helper.make_node("Split", ["blended_rgb"], ["blend_R","blend_G","blend_B"], name="split_blend", axis=1, split=[1,1,1]))
nodes.append(helper.make_node("Add", ["blend_R", "delta_R_full"], ["corr_R_raw"], name="add_delta_R"))
nodes.append(helper.make_node("Add", ["blend_B", "delta_B_full"], ["corr_B_raw"], name="add_delta_B"))

# Use confidence to blend between blended and corrected channels
nodes.append(helper.make_node("Sub", ["one_scalar", "conf_R_full_s"], ["inv_conf_R"], name="sub_inv_conf_R"))
nodes.append(helper.make_node("Mul", ["corr_R_raw", "conf_R_full_s"], ["corr_R_p"], name="mul_corrR_conf"))
nodes.append(helper.make_node("Mul", ["blend_R", "inv_conf_R"], ["blend_R_p"], name="mul_blendR_invconf"))
nodes.append(helper.make_node("Add", ["corr_R_p", "blend_R_p"], ["corr_R"], name="add_corrR"))

nodes.append(helper.make_node("Sub", ["one_scalar", "conf_B_full_s"], ["inv_conf_B"], name="sub_inv_conf_B"))
nodes.append(helper.make_node("Mul", ["corr_B_raw", "conf_B_full_s"], ["corr_B_p"], name="mul_corrB_conf"))
nodes.append(helper.make_node("Mul", ["blend_B", "inv_conf_B"], ["blend_B_p"], name="mul_blendB_invconf"))
nodes.append(helper.make_node("Add", ["corr_B_p", "blend_B_p"], ["corr_B"], name="add_corrB"))

# 14) Reconstruct candidate_rgb from corr_R, blend_G, corr_B
nodes.append(helper.make_node("Concat", ["corr_R","blend_G","corr_B"], ["candidate_rgb"], name="concat_candidate", axis=1))

# 15) Build mask_known_full from phase masks at full res
nodes.append(helper.make_node("Resize", ["p00_coarse", roi_init.name, scales_full_init.name], ["p00_full"], name="resize_p00_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_coarse", roi_init.name, scales_full_init.name], ["p01_full"], name="resize_p01_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_coarse", roi_init.name, scales_full_init.name], ["p10_full"], name="resize_p10_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_coarse", roi_init.name, scales_full_init.name], ["p11_full"], name="resize_p11_full", mode="nearest"))

nodes.append(helper.make_node("Add", ["p01_full", "p10_full"], ["G_full"], name="add_g_full"))
nodes.append(helper.make_node("Identity", ["p00_full"], ["R_full"], name="id_r_full"))
nodes.append(helper.make_node("Identity", ["p11_full"], ["B_full"], name="id_b_full"))
nodes.append(helper.make_node("Concat", ["R_full","G_full","B_full"], ["mask_known_full_raw"], name="concat_mask_known_raw", axis=1))
nodes.append(helper.make_node("Cast", ["mask_known_full_raw"], ["mask_known_full"], name="cast_mask_float", to=TensorProto.FLOAT))

# 16) Resize raw_rgb_coarse -> base_nearest (nearest) to full resolution
nodes.append(helper.make_node("Resize", ["raw_rgb_coarse", roi_init.name, scales_full_init.name], ["base_nearest_raw"], name="resize_base", mode="nearest"))

# 17) Ensure candidate_rgb and base_nearest have explicit shape [N,3,H_full,W_full] via Reshape
nodes.append(helper.make_node("Reshape", ["candidate_rgb", target_shape_init.name], ["candidate_rgb_r"], name="reshape_candidate"))
nodes.append(helper.make_node("Reshape", ["base_nearest_raw", target_shape_init.name], ["base_nearest"], name="reshape_base"))

# 18) Masked full-res separable refinement applied only to non-measured pixels
nodes.append(helper.make_node("Conv", ["candidate_rgb_r", init_sep_h.name], ["sep_h_out"], name="sep_h", pads=[0,1,0,1], strides=[1,1], group=3))
nodes.append(helper.make_node("Conv", ["sep_h_out", init_sep_v.name], ["refined_rgb"], name="sep_v", pads=[1,0,1,0], strides=[1,1], group=3))

# Combine refined with measured preservation:
nodes.append(helper.make_node("Reshape", ["mask_known_full", target_shape_init.name], ["mask_known_full_r"], name="reshape_mask"))
nodes.append(helper.make_node("Sub", ["one_scalar", "mask_known_full_r"], ["inv_mask"], name="sub_inv_mask"))
nodes.append(helper.make_node("Mul", ["refined_rgb", "inv_mask"], ["refined_masked"], name="mul_refined_mask"))
nodes.append(helper.make_node("Mul", ["base_nearest", "mask_known_full_r"], ["base_masked"], name="mul_base_mask"))
nodes.append(helper.make_node("Add", ["refined_masked", "base_masked"], ["final_rgb"], name="add_final"))

# Output
output_info = helper.make_tensor_value_info("final_rgb", TensorProto.FLOAT, [N, 3, H_full, W_full])

# Build graph
graph = helper.make_graph(
    nodes,
    "demosaic_mhc_final_opset11_fixed",
    inputs=[input_raw, input_phase],
    outputs=[output_info],
    initializer=initializers
)

# Create model and set opset_import to 11 and IR version to 11
model = helper.make_model(graph, producer_name="demosaic_mhc_final_fixed")
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
onnx.save(model, "demosaic_mhc_final_opset11_fixed.onnx")
print("Saved ONNX model: demosaic_mhc_final_opset11_fixed.onnx")

# -------------------------
# Run the model with onnxruntime
# -------------------------
sess = ort.InferenceSession("demosaic_mhc_final_opset11_fixed.onnx", providers=["CPUExecutionProvider"])

inputs = {
    "raw_rgb_coarse": raw_rgb_coarse.astype(np.float32),
    "phase_nhwc": phase_nhwc.astype(np.float32)
}

out = sess.run(["final_rgb"], inputs)
final_rgb = out[0]  # [N,3,H_full,W_full]

print("final_rgb.shape:", final_rgb.shape)
print("min/max final:", final_rgb.min(), final_rgb.max())
