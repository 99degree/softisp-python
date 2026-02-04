from microblocks.base import BuildResult
import onnx.helper as oh
from microblocks.base import MicroblockBase
from onnx import TensorProto


class AWBBase(MicroblockBase):
    """
    Auto White Balance microblock.
    Input:  [n,3,h,w] (RGB image)
    Output: [n,3,h,w] (RGB after channel gains)
    Needs:  wb_gains [1,3,1,1] (R,G,B multipliers)
    """

    name = "awb_base"
    version = "v0"

    def build_applier(self, stage: str, prev_stages=None):
        inits, nodes, vis = [], [], []
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier"
        wb_gains_in = f"{stage}.wb_gains"   # external input [1,3,1,1]
        wb_gains_expanded = f"{stage}.wb_gains_expanded"
        applier = f"{stage}.applier"

        # Shape of input image
        shape_name = f"{stage}.image_shape"
        nodes.append(
            oh.make_node("Shape", [input_image], [shape_name], name=f"{stage}_shape")
        )

        # Expand gains to match image shape
        nodes.append(
            oh.make_node("Expand", [wb_gains_in, shape_name], [wb_gains_expanded], name=f"{stage}_expand_gains")
        )

        # Multiply with broadcasted gains
        nodes.append(
            oh.make_node(
                "Mul",
                inputs=[input_image, wb_gains_in],
                outputs=[applier],
                name=f"{stage}_wb_gain_internal"
            )
        )

        # ValueInfos for audit clarity
        vis += [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, [1, 3, "h", "w"]),
            oh.make_tensor_value_info(wb_gains_in, TensorProto.FLOAT, [1, 3, 1, 1]),
            oh.make_tensor_value_info(applier, TensorProto.FLOAT, [1, 3, "h", "w"]),
        ]

        outputs = {"applier": {"name": applier, "type": TensorProto.FLOAT, "shape": [1,3,"h","w"]}}

        # Explicit external inputs
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=[1,3,"h","w"])
        result.appendInput(wb_gains_in, shape=[1,3,1,1], type=TensorProto.FLOAT)
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Coordinator: stabilizes wb_gains by clipping the delta
        relative to previous frame gains.
        """
        curr_gains  = f"{stage}.wb_gains"       # algo output (external input)
        prev_gains  = f"{stage}.wb_gains_prev"  # previous stabilized gains
        delta_name  = f"{stage}.delta"
        clipped     = f"{stage}.delta_clipped"
        out_gains   = f"{stage}.wb_gains_out"

        # Nodes
        sub_node = oh.make_node("Sub", [curr_gains, prev_gains], [delta_name], name=f"{stage}_awb_delta")
        clip_node = oh.make_node("Clip", [delta_name, f"{stage}.min_delta", f"{stage}.max_delta"], [clipped], name=f"{stage}_awb_clip")
        add_node = oh.make_node("Add", [prev_gains, clipped], [out_gains], name=f"{stage}_awb_out")

        # Initializers
        delta_init_min = oh.make_tensor(f"{stage}.min_delta", TensorProto.FLOAT, [1], [-0.2])
        delta_init_max = oh.make_tensor(f"{stage}.max_delta", TensorProto.FLOAT, [1], [0.2])

        # ValueInfos
        vis = [
            oh.make_tensor_value_info(curr_gains, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(prev_gains, TensorProto.FLOAT, [3]),
            oh.make_tensor_value_info(out_gains, TensorProto.FLOAT, [3]),
        ]

        outputs = {"wb_gains_out": {"name": out_gains}}

        result = BuildResult(outputs, [sub_node, clip_node, add_node],
                             [delta_init_min, delta_init_max], vis)
        result.appendInput(curr_gains)   # algo output
        result.appendInput(prev_gains)   # previous gains
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        """
        Build a minimal test graph for AWB.
        If prev_stages is empty or invalid, generate the test applier output.
        Otherwise, return an empty BuildResult.
        """
        if not prev_stages or prev_stages[0] == "":
            nodes, inits, vis = [], [], []
            applier_name = ".applier"

            vis.append(
                oh.make_tensor_value_info(applier_name, TensorProto.FLOAT, [1, 3, 60, 80])
            )

            import numpy as np
            test_data = np.random.rand(1, 3, 60, 80).astype(np.float32)
            inits.append(
                oh.make_tensor(applier_name, TensorProto.FLOAT,
                               test_data.shape, test_data.flatten().tolist())
            )

            outputs = {
                "applier": {"name": applier_name,
                            "type": TensorProto.FLOAT,
                            "shape": [1, 3, 60, 80]},
            }

            return BuildResult(outputs, nodes, inits, vis)

        raise NotImplementedError
