from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


class ChromaSubsampleBase(MicroblockBase):
    """
    Chroma subsampling microblock.
    Input:  [n,3,target_h,target_w]   (YUV)
    Output: [n,3,target_h,target_w/2] (YUV 4:2:0)

    Optional parameter:
        subsample_scale [4] (scale factors for N,C,H,W)
    """
    name = "chroma_subsample_base"
    version = "v0"

    # -------------------------------
    # Applier (runtime subsampling)
    # -------------------------------
    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        scale = f"{stage}.subsample_scale"
        out_name = f"{stage}.applier"

        # Apply chroma subsampling using Resize
        node = oh.make_node(
            "Resize",
            inputs=[input_image, "", scale],
            outputs=[out_name],
            name=f"{stage}_chroma",
            mode="nearest"
        )

        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(scale, TensorProto.FLOAT, [4]),
            oh.make_tensor_value_info(out_name, TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {"applier": {"name": out_name, "type":TensorProto.FLOAT, "shape":["n", 3, "h", "w"]}}

        result = BuildResult(outputs, [node], [], vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])  # upstream image only
        result.appendInput(scale, type=TensorProto.FLOAT, shape=[4])
        return result

    # -------------------------------
    # Algo (declare scale + pass-through image)
    # -------------------------------
    def build_algo(self, stage: str, prev_stages=None):
        nodes, inits, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"

        # Internal scale parameter with default
        scale = f"{stage}.subsample_scale_src"
        default_scale = [1.0, 1.0, 1.0, 0.5]
        inits.append(oh.make_tensor(scale, TensorProto.FLOAT, [4], default_scale))
        vis.append(oh.make_tensor_value_info(scale, TensorProto.FLOAT, [4]))

        # Identity to expose scale as visible output
        scale_out = f"{stage}.subsample_scale"
        nodes.append(oh.make_node("Identity", [scale], [scale_out], name=f"{stage}.scale_identity"))
        vis.append(oh.make_tensor_value_info(scale_out, TensorProto.FLOAT, [4]))

        # Pass-through image
        out_name = f"{stage}.applier"
        nodes.append(oh.make_node("Identity", [input_image], [out_name], name=f"{stage}.identity"))
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n", 3, "h", "w"]),
            oh.make_tensor_value_info(out_name,   TensorProto.FLOAT, ["n", 3, "h", "w"]),
        ]

        outputs = {
            "applier":        {"name": out_name, "type":TensorProto.FLOAT, "shape":["n", 3, "h", "w"]},
            "subsample_scale": {"name": scale_out, "type":TensorProto.FLOAT, "shape":[4]},  # visible output
        }

        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n", 3, "h", "w"])  # upstream image only
        # scale is optional: do not appendInput here
        return result
