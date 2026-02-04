from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
import numpy as np
from microblocks.registry import Registry
from .image_desc_base import ImageDescBase

class ImageDescV1(ImageDescBase):
    """
    Microblock that anchors image metadata for RGB/YUV.
    Expects [1,3,h,w] float input, de-normalizes to int16,
    and exposes [h,w,3] plus height/width/frame_id.
    Uses constant-input form of Squeeze for portability.
    """
    name = 'image_desc_v1'
    version = 'v1-int16'
    input_format = TensorProto.FLOAT

    def build_applier(self, stage: str, prev_stages=None):
        if prev_stages is None:
            raise NotImplementedError
        upstream = prev_stages[0] if prev_stages else stage
        image_in  = f'{upstream}.applier'

        frame_in = self.getMapping("image_desc_base", prev_stages).getParam("frame_id")

        image_out = f'{stage}.rgb_out'
        width_out = f'{stage}.width'
        height_out= f'{stage}.height'
        frame_out = f'{stage}.frame_id'

        # Intermediate names
        denorm_out = f"{stage}.denorm"
        cast_out   = f"{stage}.cast_int16"
        squeeze_out= f"{stage}.squeezed"
        hwc_out    = f"{stage}.hwc"
        shape_out  = f"{stage}.shape"
        gather_h   = f"{stage}.gather_h"
        gather_w   = f"{stage}.gather_w"

        # Constants
        scale_const = oh.make_tensor(f"{stage}.scale", TensorProto.FLOAT, [1], [65535.0])  # full 16-bit
        idx_h = oh.make_tensor(f"{stage}.idx_h", TensorProto.INT64, [1], [2])  # h index
        idx_w = oh.make_tensor(f"{stage}.idx_w", TensorProto.INT64, [1], [3])  # w index
        axes0 = oh.make_tensor(f"{stage}.axes0", TensorProto.INT64, [1], [0])  # squeeze batch dim

        nodes = [
            # De-normalize
            oh.make_node('Mul', inputs=[image_in, f"{stage}.scale"], outputs=[denorm_out],
                         name=f'{stage}_denorm'),
            # Cast to int16
            oh.make_node('Cast', inputs=[denorm_out], outputs=[cast_out],
                         name=f'{stage}_cast_int16', to=TensorProto.INT16),
            # Squeeze batch dimension using constant input
            oh.make_node('Squeeze', inputs=[cast_out, f"{stage}.axes0"], outputs=[squeeze_out],
                         name=f'{stage}_squeeze'),
            # Transpose to [h,w,3]
            oh.make_node('Transpose', inputs=[squeeze_out], outputs=[hwc_out],
                         name=f'{stage}_transpose', perm=[1,2,0]),
            # Identity to expose image
            oh.make_node('Identity', inputs=[hwc_out], outputs=[image_out], name=f'{stage}_image_id'),
            # Shape of cast_out [1,3,h,w]
            oh.make_node('Shape', inputs=[cast_out], outputs=[shape_out], name=f'{stage}_shape'),
            # Gather h and w
            oh.make_node('Gather', inputs=[shape_out, f"{stage}.idx_h"], outputs=[gather_h],
                         name=f'{stage}_gather_h', axis=0),
            oh.make_node('Gather', inputs=[shape_out, f"{stage}.idx_w"], outputs=[gather_w],
                         name=f'{stage}_gather_w', axis=0),
            # Identity to expose
            oh.make_node('Identity', inputs=[gather_h], outputs=[height_out], name=f'{stage}_height_id'),
            oh.make_node('Identity', inputs=[gather_w], outputs=[width_out], name=f'{stage}_width_id'),
            # Pass frame_id through
            oh.make_node('Identity', inputs=[frame_in], outputs=[frame_out], name=f'{stage}_frame_id'),
        ]

        vis = [
            oh.make_tensor_value_info(image_in,  TensorProto.FLOAT, [1,3,'h','w']),
            oh.make_tensor_value_info(frame_in,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(image_out, TensorProto.INT16, ['h','w',3]),
            oh.make_tensor_value_info(height_out, TensorProto.INT64, [1]),
            oh.make_tensor_value_info(width_out,  TensorProto.INT64, [1]),
            oh.make_tensor_value_info(frame_out,  TensorProto.INT64, [1]),
        ]

        outputs = {
            'rgb_out':  {'name': image_out, 'shape':['h','w',3], 'type': TensorProto.INT16},
            'height':   {'name': height_out, 'shape':[1], 'type': TensorProto.INT64},
            'width':    {'name': width_out,  'shape':[1], 'type': TensorProto.INT64},
            'frame_id': {'name': frame_out,  'shape':[1], 'type': TensorProto.INT64},
        }

        inits = [scale_const, idx_h, idx_w, axes0]

        return BuildResult(outputs, nodes, inits, vis) \
            .appendInput(image_in, type=TensorProto.FLOAT, shape=[1,3,'h','w']) \
            .appendInput(frame_in, type=TensorProto.INT64, shape=[1])
