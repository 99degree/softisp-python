import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto
import onnxruntime as ort

H, W = 264, 364

# -------------------------------
# Step 1: Build coeff_gen.onnx
# -------------------------------
vcm_input = oh.make_tensor_value_info("vcm", TensorProto.FLOAT, [1])
k1_input  = oh.make_tensor_value_info("k1", TensorProto.FLOAT, [1])
k2_input  = oh.make_tensor_value_info("k2", TensorProto.FLOAT, [1])

yy, xx = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W), indexing="ij")
coords = np.stack([xx,yy], axis=0)[None,...]  # [1,2,H,W]
inits1 = [oh.make_tensor("coords", TensorProto.FLOAT, coords.shape, coords.flatten().tolist())]

nodes1 = []
nodes1.append(oh.make_node("Split", ["coords"], ["x","y"], axis=1))  # split channels

nodes1.append(oh.make_node("Mul", ["x","x"], ["x2"]))
nodes1.append(oh.make_node("Mul", ["y","y"], ["y2"]))
nodes1.append(oh.make_node("Add", ["x2","y2"], ["r2"]))

nodes1.append(oh.make_node("Mul", ["k1","r2"], ["k1r2"]))
nodes1.append(oh.make_node("Mul", ["k2","r2"], ["k2r2"]))
nodes1.append(oh.make_node("Mul", ["k2r2","r2"], ["k2r4"]))
nodes1.append(oh.make_node("Add", ["k1r2","k2r4"], ["poly_term"]))
nodes1.append(oh.make_node("Add", ["poly_term","vcm"], ["scale"]))

nodes1.append(oh.make_node("Mul", ["x","scale"], ["x_prime"]))
nodes1.append(oh.make_node("Mul", ["y","scale"], ["y_prime"]))
nodes1.append(oh.make_node("Concat", ["x_prime","y_prime"], ["gdc_coords"], axis=1))  # [1,2,H,W]

nodes1.append(oh.make_node("Mul", ["scale","scale"], ["lsc_gain"]))  # [1,1,H,W]

outputs1 = [
    oh.make_tensor_value_info("gdc_coords", TensorProto.FLOAT, [1,2,H,W]),
    oh.make_tensor_value_info("lsc_gain", TensorProto.FLOAT, [1,1,H,W])
]

graph1 = oh.make_graph(nodes1,"CoeffGen",
                       inputs=[vcm_input,k1_input,k2_input],
                       outputs=outputs1,
                       initializer=inits1)

model1 = oh.make_model(graph1,
                       opset_imports=[oh.make_opsetid("",16)],
                       ir_version=11)
onnx.save(model1,"coeff_gen.onnx")
print("Saved coeff_gen.onnx")

# -------------------------------
# Step 2: Build apply_grids.onnx
# -------------------------------
img_input = oh.make_tensor_value_info("input_img", TensorProto.FLOAT, [1,3,H,W])
gdc_input = oh.make_tensor_value_info("gdc_coords", TensorProto.FLOAT, [1,2,H,W])
lsc_input = oh.make_tensor_value_info("lsc_gain", TensorProto.FLOAT, [1,1,H,W])

nodes2 = []
# Transpose gdc_coords [1,2,H,W] -> [1,H,W,2] for GridSample
nodes2.append(oh.make_node("Transpose", ["gdc_coords"], ["gdc_coords_t"], perm=[0,2,3,1]))
nodes2.append(oh.make_node("Mul", ["input_img","lsc_gain"], ["lsc_img"]))
nodes2.append(oh.make_node("GridSample", ["input_img","gdc_coords_t"], ["gdc_img"],
                          mode="bilinear", padding_mode="zeros", align_corners=1))

outputs2 = [
    oh.make_tensor_value_info("lsc_img", TensorProto.FLOAT, [1,3,H,W]),
    oh.make_tensor_value_info("gdc_img", TensorProto.FLOAT, [1,3,H,W])
]

graph2 = oh.make_graph(nodes2,"ApplyGrids",
                       inputs=[img_input,gdc_input,lsc_input],
                       outputs=outputs2)

model2 = oh.make_model(graph2,
                       opset_imports=[oh.make_opsetid("",16)],
                       ir_version=11)
onnx.save(model2,"apply_grids.onnx")
print("Saved apply_grids.onnx")

# -------------------------------
# Step 3: Run pipeline
# -------------------------------
sess1 = ort.InferenceSession("coeff_gen.onnx", providers=["CPUExecutionProvider"])
sess2 = ort.InferenceSession("apply_grids.onnx", providers=["CPUExecutionProvider"])

inputs1 = {
    "vcm": np.array([5.0], dtype=np.float32),
    "k1": np.array([0.01], dtype=np.float32),
    "k2": np.array([-0.0001], dtype=np.float32),
}
gdc_coords, lsc_gain = sess1.run(None, inputs1)
print("CoeffGen outputs:", gdc_coords.shape, lsc_gain.shape)

img = np.ones((1,3,H,W), dtype=np.float32)

inputs2 = {
    "input_img": img,
    "gdc_coords": gdc_coords,
    "lsc_gain": lsc_gain,
}
lsc_img, gdc_img = sess2.run(None, inputs2)
print("ApplyGrids outputs:", lsc_img.shape, gdc_img.shape)
