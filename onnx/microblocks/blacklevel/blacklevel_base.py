from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto

class BlackLevelBase(MicroblockBase):
    """
    Black level subtraction microblock.
    Input:  [N,C,H,W] (image)
    Output: [N,C,H,W] (image with black level offset removed)
    Needs:  offset [1] (scalar)
    """
    name = 'blacklevel'
    version = 'v0'

    def build_applier(self, stage: str, prev_stages=None):
        upstream = prev_stages[0] if prev_stages else stage

        # Names
        input_image = f'{upstream}.applier'
        offset_name = f'{stage}.offset'
        out_name    = f'{stage}.applier'

        # Node
        sub_node = oh.make_node(
            'Sub',
            inputs=[input_image, offset_name],
            outputs=[out_name],
            name=f'{stage}_sub'
        )

        # ValueInfos
        vis = [
            oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ['N', 4, 'H', 'W']),
            oh.make_tensor_value_info(offset_name, TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(out_name,    TensorProto.FLOAT, ['N', 4, 'H', 'W']),
        ]

        outputs = {'applier': {'name': out_name, 'type':TensorProto.FLOAT, 'shape':['N', 4, 'H', 'W']}}

        # BuildResult + declare all external needs as inputs
        result = BuildResult(outputs, [sub_node], [], vis)
        result.appendInput(input_image, shape=["n",4,"h","w"], type=oh.TensorProto.FLOAT)
        result.appendInput(offset_name, shape=[1], type=oh.TensorProto.FLOAT)
        return result

    def build_algo(self, stage: str, prev_stages=None):
        """
        Algo path mirrors applier for consistency.
        Coordinator supplies offset.
        """
        return self.build_applier(stage, prev_stages=prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        # Names
        offset_in  = f'{stage}.offset_in'
        offset_out = f'{stage}.offset'

        # Initializer: fixed float value 0.01
        inits = [
            oh.make_tensor(offset_in, TensorProto.FLOAT, [1], [0.01]),
        ]

        # Identity node to make it node‑backed
        nodes = [
            oh.make_node('Identity', inputs=[offset_in], outputs=[offset_out], name=f'{stage}_offset_id'),
        ]

        # ValueInfos
        vis = [
            oh.make_tensor_value_info(offset_in,  TensorProto.FLOAT, [1]),
            oh.make_tensor_value_info(offset_out, TensorProto.FLOAT, [1]),
        ]

        outputs = {
            'offset': {'name': offset_out, 'type':TensorProto.FLOAT, 'shape':[1]},
        }

        return BuildResult(outputs, nodes, inits, vis)

