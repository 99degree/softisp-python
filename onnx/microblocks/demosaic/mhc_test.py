# demosaic_test_conv_pad_fix.py
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
# Create synthetic RGGB Bayer (full resolution)
# -------------------------
def make_rggb_bayer(h, w):
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = 0.5 * (yy + xx)
    img = np.zeros((h, w), dtype=np.float32)
    img[0::2, 0::2] = base[0::2, 0::2]   # R
    img[0::2, 1::2] = base[0::2, 1::2]   # G (top-right)
    img[1::2, 0::2] = base[1::2, 0::2]   # G (bottom-left)
    img[1::2, 1::2] = base[1::2, 1::2]   # B
    return img

bayer = make_rggb_bayer(H_full, W_full)  # [128,128]

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
# Build RGGB phase onehot NHWC [1,2,2,4]
# Channel order: p00 (TL), p01 (TR), p10 (BL), p11 (BR)
# -------------------------
phase_nhwc = np.zeros((1, 2, 2, 4), dtype=np.float32)
phase_nhwc[0, 0, 0, 0] = 1.0  # p00 top-left (R)
phase_nhwc[0, 0, 1, 1] = 1.0  # p01 top-right (G)
phase_nhwc[0, 1, 0, 2] = 1.0  # p10 bottom-left (G)
phase_nhwc[0, 1, 1, 3] = 1.0  # p11 bottom-right (B)

# -------------------------
# Build ONNX graph
# -------------------------
nodes = []
initializers = []

# Inputs
input_raw = helper.make_tensor_value_info("raw_rgb_coarse", TensorProto.FLOAT, [N, 3, Hc, Wc])
input_phase = helper.make_tensor_value_info("phase_nhwc", TensorProto.FLOAT, [N, 2, 2, 4])

# 1) Transpose phase NHWC -> NCHW (perm [0,3,1,2]) -> [N,4,2,2]
nodes.append(helper.make_node("Transpose", ["phase_nhwc"], ["phase_nchw"], name="transpose_phase", perm=[0,3,1,2]))

# 2) Split into p00,p01,p10,p11 along channel axis (still 2x2 spatial)
nodes.append(helper.make_node("Split", ["phase_nchw"], ["p00_2x2","p01_2x2","p10_2x2","p11_2x2"],
                              name="split_phase", axis=1, split=[1,1,1,1]))

# 3) Tile/Resize phase masks from 2x2 -> Hc x Wc (nearest)
scales_phase = np.array([1.0, 1.0, float(Hc) / 2.0, float(Wc) / 2.0], dtype=np.float32)
scales_phase_init = numpy_helper.from_array(scales_phase, name="scales_phase")
initializers.append(scales_phase_init)

roi_init = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi_empty")
initializers.append(roi_init)

nodes.append(helper.make_node("Resize", ["p00_2x2", roi_init.name, scales_phase_init.name], ["p00_coarse"], name="resize_p00_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_2x2", roi_init.name, scales_phase_init.name], ["p01_coarse"], name="resize_p01_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_2x2", roi_init.name, scales_phase_init.name], ["p10_coarse"], name="resize_p10_to_coarse", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_2x2", roi_init.name, scales_phase_init.name], ["p11_coarse"], name="resize_p11_to_coarse", mode="nearest"))

# 4) Slice raw_rgb_coarse into raw_R/raw_G/raw_B
def add_slice_node(input_name, out_name, starts, ends, axes, name):
    starts_init = numpy_helper.from_array(np.array(starts, dtype=np.int64), name=name + "_starts")
    ends_init = numpy_helper.from_array(np.array(ends, dtype=np.int64), name=name + "_ends")
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name=name + "_axes")
    initializers.extend([starts_init, ends_init, axes_init])
    inputs = [input_name, starts_init.name, ends_init.name, axes_init.name]
    return helper.make_node("Slice", inputs, [out_name], name=name)

nodes.append(add_slice_node("raw_rgb_coarse", "raw_R", [0,0,0,0], [N,1,Hc,Wc], [0,1,2,3], name="slice_raw_R"))
nodes.append(add_slice_node("raw_rgb_coarse", "raw_G", [0,1,0,0], [N,2,Hc,Wc], [0,1,2,3], name="slice_raw_G"))
nodes.append(add_slice_node("raw_rgb_coarse", "raw_B", [0,2,0,0], [N,3,Hc,Wc], [0,1,2,3], name="slice_raw_B"))

# 5) Multiply raw channels by each tiled phase mask (broadcasting)
mul_outputs = []
colors = ["R","G","B"]
phases = ["p00_coarse","p01_coarse","p10_coarse","p11_coarse"]
for c, raw_name in zip(colors, ["raw_R","raw_G","raw_B"]):
    for p in phases:
        out_name = f"m_{c}_{p}"
        nodes.append(helper.make_node("Mul", [raw_name, p], [out_name], name=f"mul_{c}_{p}"))
        mul_outputs.append(out_name)

# 6) Concat packed masked -> [N,12,Hc,Wc]
nodes.append(helper.make_node("Concat", mul_outputs, ["packed_masked"], name="concat_packed", axis=1))

# 7) Create initializer for K_subplanes: shape [12,1,2,2] (simple averaging kernel)
K_subplanes = np.ones((12, 1, 2, 2), dtype=np.float32) * 0.25
init_K = numpy_helper.from_array(K_subplanes, name="K_subplanes")
initializers.append(init_K)

# 8) Conv on packed_masked with groups=12 (each channel convolved independently)
#     IMPORTANT: pads set so output spatial size remains Hc x Wc
conv_node = helper.make_node(
    "Conv",
    inputs=["packed_masked", init_K.name],
    outputs=["conv_packed"],
    name="conv_packed",
    pads=[0,0,1,1],   # pad_top=0, pad_left=0, pad_bottom=1, pad_right=1
    strides=[1,1],
    group=12
)
nodes.append(conv_node)

# 9) Split conv_packed into 3 blocks of 4 channels each
nodes.append(helper.make_node("Split", ["conv_packed"], ["R_block","G_block","B_block"], name="split_conv_packed", axis=1, split=[4,4,4]))

# 10) DepthToSpace per color (blocksize=2)
nodes.append(helper.make_node("DepthToSpace", ["R_block"], ["up_R_raw"], name="depth2space_R", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["G_block"], ["up_G_raw"], name="depth2space_G", blocksize=2))
nodes.append(helper.make_node("DepthToSpace", ["B_block"], ["up_B_raw"], name="depth2space_B", blocksize=2))

# 11) Prepare Resize inputs for full-res (2x up from coarse)
scales_full = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
scales_full_init = numpy_helper.from_array(scales_full, name="scales_full")
initializers.append(scales_full_init)

# 12) Resize coarse phase masks to full resolution (nearest) for mask_known_full
nodes.append(helper.make_node("Resize", ["p00_coarse", roi_init.name, scales_full_init.name], ["p00_full"], name="resize_p00_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p01_coarse", roi_init.name, scales_full_init.name], ["p01_full"], name="resize_p01_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p10_coarse", roi_init.name, scales_full_init.name], ["p10_full"], name="resize_p10_full", mode="nearest"))
nodes.append(helper.make_node("Resize", ["p11_coarse", roi_init.name, scales_full_init.name], ["p11_full"], name="resize_p11_full", mode="nearest"))

# 13) Build per-color known masks: R = p00_full, G = p01_full + p10_full, B = p11_full
nodes.append(helper.make_node("Add", ["p01_full", "p10_full"], ["G_full"], name="add_g_full"))
nodes.append(helper.make_node("Identity", ["p00_full"], ["R_full"], name="id_r_full"))
nodes.append(helper.make_node("Identity", ["p11_full"], ["B_full"], name="id_b_full"))
nodes.append(helper.make_node("Concat", ["R_full","G_full","B_full"], ["mask_known_full_raw"], name="concat_mask_known_raw", axis=1))

# 14) Ensure mask_known_full is float32
nodes.append(helper.make_node("Cast", ["mask_known_full_raw"], ["mask_known_full"], name="cast_mask_float", to=TensorProto.FLOAT))

# 15) Resize raw_rgb_coarse -> base_nearest (nearest) to full resolution
nodes.append(helper.make_node("Resize", ["raw_rgb_coarse", roi_init.name, scales_full_init.name], ["base_nearest_raw"], name="resize_base", mode="nearest"))

# 16) Concat up_R/up_G/up_B -> up_rgb_raw [N,3,2Hc,2Wc]
nodes.append(helper.make_node("Concat", ["up_R_raw","up_G_raw","up_B_raw"], ["up_rgb_raw"], name="concat_up_rgb", axis=1))

# 17) Explicit target shape initializer for Reshape: [N,3,H_full,W_full]
target_shape = np.array([N, 3, H_full, W_full], dtype=np.int64)
target_shape_init = numpy_helper.from_array(target_shape, name="target_shape")
initializers.append(target_shape_init)

# 18) Reshape base_nearest_raw and up_rgb_raw to explicit target shape
nodes.append(helper.make_node("Reshape", ["base_nearest_raw", target_shape_init.name], ["base_nearest"], name="reshape_base"))
nodes.append(helper.make_node("Reshape", ["up_rgb_raw", target_shape_init.name], ["up_rgb"], name="reshape_up"))

# 19) Cast both to float (safety)
nodes.append(helper.make_node("Cast", ["base_nearest"], ["base_nearest_f"], name="cast_base_float", to=TensorProto.FLOAT))
nodes.append(helper.make_node("Cast", ["up_rgb"], ["up_rgb_f"], name="cast_up_float", to=TensorProto.FLOAT))

# 20) Compute diff = base_nearest_f - up_rgb_f
nodes.append(helper.make_node("Sub", ["base_nearest_f", "up_rgb_f"], ["diff_base_up"], name="sub_base_up"))

# 21) masked_diff = diff * mask_known_full
# Ensure mask_known_full has same shape: Reshape mask_known_full to target shape if needed
nodes.append(helper.make_node("Reshape", ["mask_known_full", target_shape_init.name], ["mask_known_full_r"], name="reshape_mask"))
nodes.append(helper.make_node("Mul", ["diff_base_up", "mask_known_full_r"], ["masked_diff"], name="mul_diff_mask"))

# 22) final = up_rgb_f + masked_diff
nodes.append(helper.make_node("Add", ["up_rgb_f", "masked_diff"], ["final_rgb"], name="add_up_masked"))

# Output
output_info = helper.make_tensor_value_info("final_rgb", TensorProto.FLOAT, [N, 3, H_full, W_full])

# Build graph
graph = helper.make_graph(
    nodes,
    "demosaic_2x2_packed_graph_conv_pad_fix",
    inputs=[input_raw, input_phase],
    outputs=[output_info],
    initializer=initializers
)

# Create model and set opset/IR to 11 for compatibility with older runtimes
model = helper.make_model(graph, producer_name="demosaic_test_conv_pad_fix")
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
onnx.save(model, "demosaic_2x2_packed_conv_pad_fix.onnx")
print("Saved ONNX model: demosaic_2x2_packed_conv_pad_fix.onnx")

# -------------------------
# Run the model with onnxruntime
# -------------------------
sess = ort.InferenceSession("demosaic_2x2_packed_conv_pad_fix.onnx", providers=["CPUExecutionProvider"])

inputs = {
    "raw_rgb_coarse": raw_rgb_coarse.astype(np.float32),
    "phase_nhwc": phase_nhwc.astype(np.float32)
}

out = sess.run(["final_rgb"], inputs)
final_rgb = out[0]  # [N,3,128,128]

print("final_rgb.shape:", final_rgb.shape)
print("min/max final:", final_rgb.min(), final_rgb.max())
