# autoexposure.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
import numpy as np


class AutoExposureBase(MicroblockBase):
    """
    Auto Exposure Base Class
    Input:  [n,3,h,w] input RGB image from previous stage (demosaic)
    Output:  auto exposure parameters (exposure value, gain)
    """

    name = "autoexposure"
    version = "v1"

    def build_applier(self, stage, prev_stages=None):
        """
        Build an auto exposure estimator that calculates exposure parameters
        based on image statistics.
        """
        # Get input from previous stage (demosaic)
        upstream = prev_stages[0] if prev_stages and len(prev_stages) > 0 else stage
        input_image = f"{upstream}.applier"
        
        # Output names
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        brightness_name = f"{stage}.brightness"
        ratio_name = f"{stage}.brightness_ratio"
        
        nodes, inits, vis = [], [], []
        
        # Add input value info
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # 1. Calculate mean brightness of the image across all channels
        # First reduce across height and width to get per-channel means
        mean_node = oh.make_node("ReduceMean", inputs=[input_image], outputs=[brightness_name], 
                              axes=[2, 3], keepdims=0)
        nodes.append(mean_node)
        
        # 2. Reshape brightness to add batch dimension back
        brightness_reshape = f"{stage}.brightness_4d"
        reshape_node = oh.make_node("Reshape", inputs=[brightness_name, "one_const"], 
                                   outputs=[brightness_reshape])
        nodes.append(reshape_node)
        inits.append(oh.make_tensor("one_const", TensorProto.INT64, [4], [1, 1, 1, 1]))
        
        # 3. Create target brightness (18% gray card standard)
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [0.18])  # Standard 18% gray target
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], value=target_tensor, 
                                 name=f"{target_name}_const")
        nodes.append(target_node)
        
        # 4. Calculate brightness to target ratio
        ratio_node = oh.make_node("Div", inputs=[brightness_reshape, target_name], outputs=[ratio_name])
        nodes.append(ratio_node)
        
        # 5. Calculate exposure adjustment: log2(target_brightness / current_brightness)
        # In ONNX this would be: log2(target/brightness) = log(target)/log(2) - log(brightness)/log(2)
        # For now we'll use a simplified approach
        ev_tensor = oh.make_tensor(ev_name, TensorProto.FLOAT, [1], [0.0])  # Placeholder
        inits.append(ev_tensor)
        ev_node = oh.make_node("Constant", inputs=[], outputs=[ev_name], value=ev_tensor, 
                              name=f"{ev_name}_const")
        nodes.append(ev_node)
        
        # 6. Calculate gain adjustment (simplified)
        gain_tensor = oh.make_tensor(gain_name, TensorProto.FLOAT, [1], [1.0])
        inits.append(gain_tensor)
        gain_node = oh.make_node("Constant", inputs=[], outputs=[gain_name], value=gain_tensor, 
                                name=f"{gain_name}_const")
        nodes.append(gain_node)
        
        # Add output value info
        vis.append(oh.make_tensor_value_info(brightness_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(target_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ratio_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "brightness": {"name": brightness_name, "shape": [1], "type": TensorProto.FLOAT},
            "target": {"name": target_name, "shape": [1], "type": TensorProto.FLOAT},
            "brightness_ratio": {"name": ratio_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_algo(self, stage, prev_stages=None):
        """Build the algorithmic part of the auto exposure block."""
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage, prev_stages=None):
        """Test algorithmic implementation of auto exposure."""
        return self.build_applier(stage, prev_stages)


class AutoExposureV1(AutoExposureBase):
    name = "autoexposure"
    version = "v1"


class AutoExposurePassthrough(AutoExposureBase):
    name = "autoexposure"
    version = "v1-passthrough"