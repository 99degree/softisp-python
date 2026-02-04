from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class CCMBase(MicroblockBase):
    """
    Color Correction Matrix (CCM) microblock.
    Input:  [n,3,h,w]   (RGB)
    Output: [n,3,h,w]   (RGB after CCM)
    Parameter (algo output / external input for applier):
        ccm [3,3,1,1] kernel
    """
    name = "ccm_base"
    version = "v0"

    # -------------------------------
    # Algo — produce CCM kernel [3,3,1,1] and pass-through image
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Identity CCM [3,3] initializer (defaults to identity)
        ccm_raw = f"{stage}.ccm_raw"
        default_matrix = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        inits.append(oh.make_tensor(ccm_raw, TensorProto.FLOAT, [3, 3], default_matrix))
        vis.append(oh.make_tensor_value_info(ccm_raw, TensorProto.FLOAT, [3, 3]))

        # Reshape to Conv kernel [out_c, in_c, kH, kW] = [3,3,1,1]
        shape_3311 = f"{stage}.shape_3311"
        inits.append(oh.make_tensor(shape_3311, TensorProto.INT64, [4], [3, 3, 1, 1]))
        ccm_kernel = f"{stage}.ccm"
        nodes.append(oh.make_node("Reshape", [ccm_raw, shape_3311], [ccm_kernel], name=f"{stage}.reshape_ccm"))
        vis.append(oh.make_tensor_value_info(ccm_kernel, TensorProto.FLOAT, [3, 3, 1, 1]))

        # Pass-through image (kept visible for audit)
        out_image = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_image], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(out_image,   TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {
            "ccm":     {"name": ccm_kernel, "type": TensorProto.FLOAT, "shape": [3, 3, 1, 1]},
            "applier": {"name": out_image,  "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]},
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])
        # ccm_raw is internal; external consumers use the reshaped kernel output
        return result

    # -------------------------------
    # Applier — consume CCM kernel [3,3,1,1] and apply via Conv
    # -------------------------------
    def build_applier(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        ccm_kernel  = f"{stage}.ccm"      # external input [3,3,1,1]
        out_name    = f"{stage}.applier"

        # 1x1 Conv applies per-pixel 3x3 matrix across channels
        nodes.append(
            oh.make_node(
                "Conv",
                inputs=[input_image, ccm_kernel],
                outputs=[out_name],
                name=f"{stage}.conv_ccm",
                strides=[1, 1],
                pads=[0, 0, 0, 0],
            )
        )

        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(ccm_kernel,  TensorProto.FLOAT, [3, 3, 1, 1]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {"applier": {"name": out_name, "type": TensorProto.FLOAT, "shape": ["n", 3, "h", "w"]}}

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])
        result.appendInput(ccm_kernel,  type=TensorProto.FLOAT, shape=[3, 3, 1, 1])
        return result
