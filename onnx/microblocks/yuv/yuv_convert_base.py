from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class YUVConvertBase(MicroblockBase):
    """
    RGB → YUV conversion microblock.
    Input:  [n,3,h,w] (RGB)
    Output: [n,3,h,w] (YUV)
    Canonical parameter output:
        rgb2yuv_matrix [3,3,1,1] kernel (for Conv)
    Audit-visible optional output:
        rgb2yuv_matrix_raw [3,3] (normalized matrix values)
    """
    name = "yuvconvert_base"
    version = "v0"

    # -------------------------------
    # Algo — produce kernel [3,3,1,1], keep raw [3,3] visible, pass-through image
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Raw normalized matrix [3,3] — distinct intermediate name to avoid rank lock-in
        matrix_raw = f"{stage}.rgb2yuv_matrix.normalized"
        default_matrix = [
            0.299,  0.587,  0.114,
           -0.147, -0.289,  0.436,
            0.615, -0.515, -0.100,
        ]
        inits.append(oh.make_tensor(matrix_raw, TensorProto.FLOAT, [3, 3], default_matrix))
        vis.append(oh.make_tensor_value_info(matrix_raw, TensorProto.FLOAT, [3, 3]))

        # Reshape to Conv kernel [out_c, in_c, kH, kW] = [3,3,1,1]
        shape_3311 = f"{stage}.shape_3311"
        inits.append(oh.make_tensor(shape_3311, TensorProto.INT64, [4], [3, 3, 1, 1]))
        matrix_kernel = f"{stage}.rgb2yuv_matrix"  # canonical output name reserved for kernel
        nodes.append(oh.make_node("Reshape", [matrix_raw, shape_3311], [matrix_kernel], name=f"{stage}.reshape_matrix"))
        vis.append(oh.make_tensor_value_info(matrix_kernel, TensorProto.FLOAT, [3, 3, 1, 1]))

        # Pass-through image (visible for audit)
        out_image = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_image], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(out_image,   TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {
            "rgb2yuv_matrix_raw": {"name": matrix_raw,   "type": TensorProto.FLOAT, "shape": [3, 3]},
            "rgb2yuv_matrix":     {"name": matrix_kernel, "type": TensorProto.FLOAT, "shape": [3, 3, 1, 1]},
            "applier":            {"name": out_image,     "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]},
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])
        return result

    # -------------------------------
    # Applier — consume kernel [3,3,1,1] and apply via 1x1 Conv
    # -------------------------------
    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        kernel = f"{stage}.rgb2yuv_matrix"  # expects [3,3,1,1]
        out_name = f"{stage}.applier"

        # 1x1 Conv applies per-pixel 3x3 transform across channels
        node = oh.make_node(
            "Conv",
            inputs=[input_image, kernel],
            outputs=[out_name],
            name=f"{stage}_yuvconvert_conv",
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(kernel,      TensorProto.FLOAT, [3, 3, 1, 1]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {
            "applier": {"name": out_name, "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]},
        }
        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])
        result.appendInput(kernel,      type=TensorProto.FLOAT, shape=[3, 3, 1, 1])
        return result
