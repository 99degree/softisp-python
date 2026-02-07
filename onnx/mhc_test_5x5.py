# demosaic_mhc_like_coarse3x3_opset16_ir11.py
# Deterministic MHC-like coarse pipeline (no training).
# Opset: 16 (node signatures use split as input), IR version set to 11 per user request.
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
H_full = 128
W_full = 128
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
# Build ONNX graph nodes and initializers
# -------------------------
nodes = []
initializers = []

# Inputs
input_raw = helper.make_tensor_value_info("raw_rgb_coarse", TensorProto.FLOAT, [N, 3, Hc, Wc])
input_phase = helper.make_tensor_value_info("phase_nhwc", TensorProto.FLOAT, [N, 2, 2, 4])

# 1) Phase transpose
nodes.append(helper.make_node("Transpose", ["phase_nhwc"], ["phase_nchw"], name="transpose_phase", perm=[0,3,1,2]))

# For opset 16 Split expects 'split' as an input tensor; create split initializers
split_4 = numpy_helper.from_array(np.array([1,1,1,1], dtype=np.int64), name="split_4")
split_3x4 = numpy_helper.from_array(np.array([4,4,4], dtype=np.int64), name="split_3x4")
split_6 = numpy_helper.from_array(np.array([1,1,1,1,1,1], dtype=np.int64), name="split_6")
split_3 = numpy_helper.from_array(np.array([1,1,1], dtype=np.int64), name="split_3")
initializers.extend([split_4, split_3x4, split_6, split_3])

# 2) Split phase into 4 channels (opset 16: split as input)
nodes.append(helper.make_node("Split", ["phase_nchw", split_4.name], ["p00_2x2","p01_2x2","p10_2x2","p11_2x2"],
                              name="split_phase", axis=1))

# 3) Resize phase masks 2x2 -> Hc x Wc (nearest)
scales_phase = np.array([1.0, 1.0, float(Hc) / 2.0, float(Wc) / 2.0], dtype=np.float32)
scales_phase_init = numpy_helper.from_array(scales_phase, name="scales_phase")
initializers.append(scales_phase_init)
roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi_empty")
initializers.append(roi_init)

nodes.append(helper.make_node("Resize", ["p00_2x2", roi_init.name, scales_phase_init.name], ["p00_coarse"], name="resize_p00_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_2x2", roi_init.name, scales_phase_init.name], ["p01_coarse"], name="resize_p01_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_2x2", roi_init.name, scales_phase_init.name], ["p10_coarse"], name="resize_p10_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_2x2", roi_init.name, scales_phase_init.name], ["p11_coarse"], name="resize_p11_to_coarse", mode="nearest"))

# 4) Slice raw channels (opset 11+ Slice signature: inputs [data, starts, ends, axes])
def add_slice(input_name, out_name, starts, ends, axes, name):
    s = numpy_helper.from_array(np.array(starts, dtype=np.int64), name=name + "_starts")
    e = numpy_helper.from_array(np.array(ends, dtype=np.int64), name=name + "_ends")
    a = numpy_helper.from_array(np.array(axes, dtype=np.int64), name=name + "_axes")
    initializers.extend([s, e, a])
    return helper.make_node("Slice", [input_name, s.name, e.name, a.name], [out_name], name=name)

nodes.append(add_slice("raw_rgb_coarse", "raw_R", [0,0,0,0], [N,1,Hc,Wc], [0,1,2,3], "slice_raw_R"))
nodes.append(add_slice("raw_rgb_coarse", "raw_G", [0,1,0,0], [N,2,Hc,Wc], [0,1,2,3], "slice_raw_G"))
nodes.append(add_slice("raw_rgb_coarse", "raw_B", [0,2,0,0], [N,3,Hc,Wc], [0,1,2,3], "slice_raw_B"))

# 5) Multiply raw channels by tiled phase masks and pack -> [N,12,Hc,Wc]
mul_outputs = []
colors = ["R","G","B"]
phases = ["p00_coarse","p01_coarse","p10_coarse","p11_coarse"]
for c, raw in zip(colors, ["raw_R","raw_G","raw_B"]):
    for p in phases:
        out = f"m_{c}_{p}"
        nodes.append(helper.make_node("Mul", [raw, p], [out], name=f"mul_{c}_{p}"))
        mul_outputs.append(out)
nodes.append(helper.make_node("Concat", mul_outputs, ["packed_masked"], name="concat_packed", axis=1))

# 6) Grouped 5x5 convs for directional candidates (horizontal and vertical)
#    Use per-packed-channel grouped convs: kernel shape [12,1,5,5], groups=12
K_h = np.zeros((12,1,5,5), dtype=np.float32)
K_v = np.zeros((12,1,5,5), dtype=np.float32)
# horizontal: emphasize left/right; vertical: emphasize up/down
for i in range(12):
    row = np.array([0.05, 0.2, 0.4, 0.2, 0.05], dtype=np.float32)
    K_h[i,0,2,:] = row  # center row
    K_v[i,0,:,2] = row  # center column (transpose)

# Normalize kernels so sum ~1
for k in (K_h, K_v):
    s = k.sum(axis=(2,3), keepdims=True)
    s[s == 0] = 1.0
    k[:] = k / s

init_K_h = numpy_helper.from_array(K_h.astype(np.float32), name="K_h")
init_K_v = numpy_helper.from_array(K_v.astype(np.float32), name="K_v")
initializers.extend([init_K_h, init_K_v])

# Conv horizontal and vertical (pads=[2,2,2,2] to preserve Hc,Wc)
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_h.name], ["conv_packed_h"], name="conv_packed_h",
                              pads=[2,2,2,2], strides=[1,1], group=12))
nodes.append(helper.make_node("Conv", ["packed_masked", init_K_v.name], ["conv_packed_v"], name="conv_packed_v",
                              pads=[2,2,2,2], strides=[1,1], group=12))

# 7) Split conv outputs into per-color 4-channel blocks (opset 16: split as input)
nodes.append(helper.make_node("Split", ["conv_packed_h", split_3x4.name], ["R_block_h","G_block_h","B_block_h"], name="split_h", axis=1))
nodes.append(helper.make_node("Split", ["conv_packed_v", split_3x4.name], ["R_block_v","G_block_v","B_block_v"], name="split_v", axis=1))

# 8) DepthToSpace each block -> up_h and up_v per color
nodes.append(helper.make_node("DepthToSpace", ["R_block_h"], ["up_R_h"], name="d2s_R_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_h"], ["up_G_h"], name="d2s_G_h", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_h"], ["up_B_h"], name="d2s_B_h", blocksize=2))

nodes.append(helper.make_node("DepthToSpace", ["R_block_v"], ["up_R_v"], name="d2s_R_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block_v"], ["up_G_v"], name="d2s_G_v", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block_v"], ["up_B_v"], name="d2s_B_v", blocksize=2))

# 9) Compute coarse green gradients (Sobel) to produce directional weight
sobel_h = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32).reshape(1,1,3,3)
sobel_v = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32).reshape(1,1,3,3)
init_sobel_h = numpy_helper.from_array(sobel_h, name="sobel_h")
init_sobel_v = numpy_helper.from_array(sobel_v, name="sobel_v")
initializers.extend([init_sobel_h, init_sobel_v])

nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_h.name], ["grad_h"], name="conv_grad_h", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Conv", ["raw_G", init_sobel_v.name], ["grad_v"], name="conv_grad_v", pads=[1,1,1,1], strides=[1,1]))
nodes.append(helper.make_node("Abs", ["grad_h"], ["mag_h"], name="abs_h"))
nodes.append(helper.make_node("Abs", ["grad_v"], ["mag_v"], name="abs_v"))
nodes.append(helper.make_node("Sub", ["mag_v", "mag_h"], ["dir_score"], name="sub_dir_score"))

# Smooth dir_score with 3x3 average
smooth_k = np.ones((1,1,3,3), dtype=np.float32) / 9.0
init_smooth = numpy_helper.from_array(smooth_k, name="smooth_k")
initializers.append(init_smooth)
nodes.append(helper.make_node("Conv", ["dir_score", init_smooth.name], ["dir_smooth"], name="conv_dir_smooth", pads=[1,1,1,1], strides=[1,1]))

# scale factor alpha and sigmoid -> weight_coarse
alpha = np.array([2.0], dtype=np.float32)
alpha_init = numpy_helper.from_array(alpha, name="alpha")
initializers.append(alpha_init)
nodes.append(helper.make_node("Mul", ["dir_smooth", "alpha"], ["dir_scaled"], name="mul_dir_alpha"))
nodes.append(helper.make_node("Sigmoid", ["dir_scaled"], ["weight_coarse"], name="sigmoid_dir"))

# 10) Upsample weight_coarse to full resolution
scales_full = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
scales_full_init = numpy_helper.from_array(scales_full, name="scales_full")
initializers.append(scales_full_init)
nodes.append(helper.make_node("Resize", ["weight_coarse", roi_init.name, scales_full_init.name], ["weight_full"], name="resize_weight_full", mode="linear"))

# 11) Concat up_h and up_v per color into full candidate tensors
nodes.append(helper.make_node("Concat", ["up_R_h","up_G_h","up_B_h"], ["up_h_rgb"], name="concat_up_h", axis=1))
nodes.append(helper.make_node("Concat", ["up_R_v","up_G_v","up_B_v"], ["up_v_rgb"], name="concat_up_v", axis=1))

# 12) Blend candidates: blended = up_h * w + up_v * (1-w)
nodes.append(helper.make_node("Concat", ["weight_full","weight_full","weight_full"], ["weight_full_3"], name="concat_weight3", axis=1))
one_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one_scalar")
initializers.append(one_init)
nodes.append(helper.make_node("Sub", ["one_scalar", "weight_full_3"], ["inv_weight"], name="sub_inv_weight"))
nodes.append(helper.make_node("Mul", ["up_h_rgb", "weight_full_3"], ["part_h"], name="mul_up_h"))
nodes.append(helper.make_node("Mul", ["up_v_rgb", "inv_weight"], ["part_v"], name="mul_up_v"))
nodes.append(helper.make_node("Add", ["part_h", "part_v"], ["blended_rgb"], name="add_blend"))

# 13) Compute coarse delta maps (ΔR, ΔB) guided by green: grouped conv producing 6 maps (delta/conf pairs)
K_aux = np.zeros((6,1,3,3), dtype=np.float32)
# initialize small confidence in center for each conf map
for i in range(3):
    K_aux[2*i+1,0,1,1] = 0.5
init_K_aux = numpy_helper.from_array(K_aux.astype(np.float32), name="K_aux")
initializers.append(init_K_aux)
nodes.append(helper.make_node("Conv", ["raw_rgb_coarse", init_K_aux.name], ["aux_out"], name="conv_aux", pads=[1,1,1,1], strides=[1,1], group=3))

# Split aux_out into six maps (opset 16: split as input)
nodes.append(helper.make_node("Split", ["aux_out", split_6.name],
                              ["delta_R_coarse","conf_R_coarse","delta_G_coarse","conf_G_coarse","delta_B_coarse","conf_B_coarse"],
                              name="split_aux", axis=1))

# 14) Upsample delta_R_coarse and delta_B_coarse to full resolution
nodes.append(helper.make_node("Resize", ["delta_R_coarse", roi_init.name, scales_full_init.name], ["delta_R_full"], name="resize_delta_R", mode="linear"))
nodes.append(helper.make_node("Resize", ["delta_B_coarse", roi_init.name, scales_full_init.name], ["delta_B_full"], name="resize_delta_B", mode="linear"))

# 15) Split blended_rgb into channels and add delta to R and B
nodes.append(helper.make_node("Split", ["blended_rgb", split_3.name], ["blend_R","blend_G","blend_B"], name="split_blend", axis=1))
nodes.append(helper.make_node("Add", ["blend_R", "delta_R_full"], ["corr_R"], name="add_delta_R"))
nodes.append(helper.make_node("Add", ["blend_B", "delta_B_full"], ["corr_B"], name="add_delta_B"))

# 16) Reconstruct final candidate_rgb
nodes.append(helper.make_node("Concat", ["corr_R","blend_G","corr_B"], ["candidate_rgb"], name="concat_candidate", axis=1))

# 17) Build mask_known_full from phase masks at full res
nodes.append(helper.make_node("Resize", ["p00_coarse", roi_init.name, scales_full_init.name], ["p00_full"], name="resize_p00_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_coarse", roi_init.name, scales_full_init.name], ["p01_full"], name="resize_p01_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_coarse", roi_init.name, scales_full_init.name], ["p10_full"], name="resize_p10_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_coarse", roi_init.name, scales_full_init.name], ["p11_full"], name="resize_p11_full", mode="nearest"))

nodes.append(helper.make_node("Add", ["p01_full", "p10_full"], ["G_full"], name="add_g_full"))
nodes.append(helper.make_node("Identity", ["p00_full"], ["R_full"], name="id_r_full"))
nodes.append(helper.make_node("Identity", ["p11_full"], ["B_full"], name="id_b_full"))
nodes.append(helper.make_node("Concat", ["R_full","G_full","B_full"], ["mask_known_full_raw"], name="concat_mask_known_raw", axis=1))
nodes.append(helper.make_node("Cast", ["mask_known_full_raw"], ["mask_known_full"], name="cast_mask_float", to=TensorProto.FLOAT))

# 18) Resize raw_rgb_coarse -> base_nearest (nearest) to full resolution
nodes.append(helper.make_node("Resize", ["raw_rgb_coarse", roi_init.name, scales_full_init.name], ["base_nearest_raw"], name="resize_base", mode="nearest"))

# 19) Ensure candidate_rgb and base_nearest have explicit shape [N,3,H_full,W_full] via Reshape
target_shape = np.array([N, 3, H_full, W_full], dtype=np.int64)
target_shape_init = numpy_helper.from_array(target_shape, name="target_shape")
initializers.append(target_shape_init)

nodes.append(helper.make_node("Reshape", ["candidate_rgb", target_shape_init.name], ["candidate_rgb_r"], name="reshape_candidate"))
nodes.append(helper.make_node("Reshape", ["base_nearest_raw", target_shape_init.name], ["base_nearest"], name="reshape_base"))

# 20) Final gating: preserve measured samples exactly
nodes.append(helper.make_node("Reshape", ["mask_known_full", target_shape_init.name], ["mask_known_full_r"], name="reshape_mask"))
nodes.append(helper.make_node("Sub", ["one_scalar", "mask_known_full_r"], ["inv_mask"], name="sub_inv_mask"))
nodes.append(helper.make_node("Mul", ["base_nearest", "mask_known_full_r"], ["base_masked"], name="mul_base_mask"))
nodes.append(helper.make_node("Mul", ["candidate_rgb_r", "inv_mask"], ["cand_masked"], name="mul_cand_mask"))
nodes.append(helper.make_node("Add", ["base_masked", "cand_masked"], ["final_rgb"], name="add_final"))

# Output
output_info = helper.make_tensor_value_info("final_rgb", TensorProto.FLOAT, [N, 3, H_full, W_full])

# Build graph
graph = helper.make_graph(
    nodes,
    "demosaic_mhc_like_coarse3x3_opset16_ir11",
    inputs=[input_raw, input_phase],
    outputs=[output_info],
    initializer=initializers
)

# Create model: set opset_import to 16 and IR version to 11 per user request
model = helper.make_model(graph, producer_name="demosaic_mhc_like_opset16_ir11")
# Ensure opset_import is set to 16
if len(model.opset_import) == 0:
    ops = OperatorSetIdProto()
    ops.version = 16
    model.opset_import.extend([ops])
else:
    model.opset_import[0].version = 16

# Force IR version to 11 as requested
model.ir_version = 11

# Infer shapes and validate (checker may warn if runtime expects different IR; user requested IR=11)
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

# Save model
onnx.save(model, "demosaic_mhc_like_coarse3x3_opset16_ir11.onnx")
print("Saved ONNX model: demosaic_mhc_like_coarse3x3_opset16_ir11.onnx")

# -------------------------
# Run the model with onnxruntime
# -------------------------
# Note: If your onnxruntime does not support opset 16 / IR 11 combination, you may need to upgrade runtime.
sess = ort.InferenceSession("demosaic_mhc_like_coarse3x3_opset16_ir11.onnx", providers=["CPUExecutionProvider"])

inputs = {
    "raw_rgb_coarse": raw_rgb_coarse.astype(np.float32),
    "phase_nhwc": phase_nhwc.astype(np.float32)
}

out = sess.run(["final_rgb"], inputs)
final_rgb = out[0]  # [N,3,H_full,W_full]

print("final_rgb.shape:", final_rgb.shape)
print("min/max final:", final_rgb.min(), final_rgb.max())
