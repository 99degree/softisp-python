from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
import numpy as np

class ImageDescBase(MicroblockBase):
    """
    Pseudo microblock that anchors image metadata.
    Declares image [n,c,h,stride], width, and frame_id.
    Each output is re-emitted via Identity so graph outputs are node-backed.
    """
    name = 'image_desc_base'
    version = 'v0'
    input_format = TensorProto.INT16

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stage)

    def build_applier(self, stage: str, prev_stages=None):
        image_in  = f'{stage}.input.image'
        width_in  = f'{stage}.input.width'
        frame_in  = f'{stage}.input.frame_id'

        image_out = f'{stage}.applier'
        width_out = f'{stage}.width'
        frame_out = f'{stage}.frame_id'

        # Intermediate names
        cast_out     = f"{stage}.cast_float"
        shape_hw     = f"{stage}.shape_hw"
        expand_shape = f"{stage}.expand_shape"

        # Initializer for [1,1]
        const_1_1 = oh.make_tensor(f"{stage}.const_1_1", TensorProto.INT64, [2], [1,1])

        nodes = [
            # Cast to float
            oh.make_node('Cast', inputs=[image_in], outputs=[cast_out],
                         name=f'{stage}_cast_to_float', to=TensorProto.FLOAT),
            # Get shape of input [h,w]
            oh.make_node('Shape', inputs=[image_in], outputs=[shape_hw], name=f'{stage}_shape'),
            # Concat [1,1] with [h,w] → [1,1,h,w]
            oh.make_node('Concat', inputs=[f"{stage}.const_1_1", shape_hw],
                         outputs=[expand_shape], name=f'{stage}_concat', axis=0),
            # Reshape casted image to [1,1,h,w]
            oh.make_node('Reshape', inputs=[cast_out, expand_shape],
                         outputs=[image_out], name=f'{stage}_reshape'),
            # Pass width and frame_id through
            oh.make_node('Identity', inputs=[width_in], outputs=[width_out], name=f'{stage}_width_id'),
            oh.make_node('Identity', inputs=[frame_in], outputs=[frame_out], name=f'{stage}_frame_id'),
        ]

        vis = [
            oh.make_tensor_value_info(image_in,  self.input_format, ['h','w']),  # unknown dims
            oh.make_tensor_value_info(width_in,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(frame_in,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(image_out, TensorProto.FLOAT, [1,1,'h','w']),  # will be [1,1,h,w]
            oh.make_tensor_value_info(width_out, TensorProto.INT64, [1]),
            oh.make_tensor_value_info(frame_out, TensorProto.INT64, [1]),
        ]

        outputs = {
            'applier':  {'name': image_out, 'shape':[1,1,'h','w'], 'type': TensorProto.FLOAT},
            'width':    {'name': width_out, 'shape':[1], 'type': TensorProto.INT64},
            'frame_id': {'name': frame_out, 'shape':[1], 'type': TensorProto.INT64},
        }

        return BuildResult(outputs, nodes, [const_1_1], vis) \
            .appendInput(image_in, type=self.input_format, shape=['h','w']) \
            .appendInput(width_in, type=TensorProto.INT64, shape=[1]) \
            .appendInput(frame_in, type=TensorProto.INT64, shape=[1])

    def build_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        """
        Independent test builder:
        - Generates synthetic Bayer FHD (1080 x 2048 stride, 1920 active)
        - Seeds width=1920 and frame_id=1 as constants
        - Outputs image as int16 with same shape [h,w]
        """
        HEIGHT = 1080 // 2
        WIDTH_ACTIVE = 1920 // 2
        STRIDE = 2048 // 2
        BIT_DEPTH = 10
        FRAME_ID = 1
     
        # Synthetic Bayer (float32 for generation)
        max_val = (1 << BIT_DEPTH) - 1
        base = np.linspace(0, max_val, HEIGHT * WIDTH_ACTIVE, dtype=np.float32).reshape(HEIGHT, WIDTH_ACTIVE)
        noise = np.random.normal(0, 3.0, size=(HEIGHT, WIDTH_ACTIVE)).astype(np.float32)
        raw_active = np.clip(base + noise, 0, max_val).astype(np.float32)
     
        raw = np.zeros((HEIGHT, STRIDE), dtype=np.float32)
        raw[:, :WIDTH_ACTIVE] = raw_active
     
        # Convert to int16 for initializer
        raw_int16 = raw.astype(np.int16)
     
        # Input names (seeded as initializers)
        image_in = f'{stage}.image'
        width_in = f'{stage}.width'
        frame_in = f'{stage}.frame_id'
     
        # Output names (stage-scoped)
        image_out = f'{stage}.input.image'
        width_out = f'{stage}.input.width'
        frame_out = f'{stage}.input.frame_id'
     
        # Initializers (constants + image tensor)
        inits = [
            oh.make_tensor(width_in, TensorProto.INT64, [1], [WIDTH_ACTIVE]),
            oh.make_tensor(frame_in, TensorProto.INT64, [1], [FRAME_ID]),
            oh.make_tensor(image_in, TensorProto.INT16, [HEIGHT, STRIDE], raw_int16.flatten().tolist()),
        ]
     
        # Nodes: just Identity to back outputs
        nodes = [
            oh.make_node('Identity', inputs=[image_in], outputs=[image_out], name=f'{stage}_image_id'),
            oh.make_node('Identity', inputs=[width_in], outputs=[width_out], name=f'{stage}_width_id'),
            oh.make_node('Identity', inputs=[frame_in], outputs=[frame_out], name=f'{stage}_frame_id'),
        ]
     
        vis = [
            oh.make_tensor_value_info(image_in,  TensorProto.INT16, [HEIGHT, STRIDE]),
            oh.make_tensor_value_info(width_in,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(frame_in,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(image_out, TensorProto.INT16, [HEIGHT, STRIDE]),
            oh.make_tensor_value_info(width_out, TensorProto.INT64, [1]),
            oh.make_tensor_value_info(frame_out, TensorProto.INT64, [1]),
        ]
     
        outputs = {
            'applier':  {'name': image_out, 'type':TensorProto.INT16, 'shape':[HEIGHT, STRIDE]},
            'width':    {'name': width_out, 'type':TensorProto.INT64, 'shape':[1]},
            'frame_id': {'name': frame_out, 'type':TensorProto.INT64, 'shape':[1]},
        }
     
        return BuildResult(outputs, nodes, inits, vis)
 
class ImageDescV1(ImageDescBase):
    """
    Pseudo microblock that anchors image metadata.
    Declares image [n,c,h,stride], width, and frame_id.
    Each output is re-emitted via Identity so graph outputs are node-backed.
    """
    name = 'image_desc_v1'
    version = 'v1-float-float'
    input_format = TensorProto.FLOAT
