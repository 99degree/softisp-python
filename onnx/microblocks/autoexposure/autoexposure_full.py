# autoexposure.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
import numpy as np


def _n(stage, suffix):
    """Generate unique node name per stage"""
    return f"{stage}.{suffix}"




class AutoExposureBase(MicroblockBase):
    """
    Auto Exposure Base Class - DO NOT REGISTER
    Input:  [n,3,h,w] input RGB image from previous stage (demosaic)
    Output:  auto exposure parameters (exposure value, gain)
    """

    # Set name=None to prevent auto-registration
    name = None
    version = None

    def _make_output_name(self, stage, suffix):
        """Generate unique output name with class prefix"""
        cls_name = self.__class__.__name__
        return f"{stage}.{cls_name}.{suffix}"

    def build_applier(self, stage, prev_stages=None):
        """
        Build an auto exposure estimator that calculates exposure parameters
        based on image statistics.
        """
        # Get input from previous stage (demosaic)
        upstream = prev_stages[0] if prev_stages and len(prev_stages) > 0 else stage
        input_image = f"{upstream}.applier"
        
        # Output names - use unique suffixes (used as FunctionProto outputs)
        ev_name = self._make_output_name(stage, "ev")
        gain_name = self._make_output_name(stage, "gain")
        target_name = self._make_output_name(stage, "target")
        brightness_name = self._make_output_name(stage, "brightness")
        ratio_name = self._make_output_name(stage, "ratio")
        brightness_4d = self._make_output_name(stage, "brightness_4d")
        
        # Internal names for initializers (auto-converted to Constant nodes by _to_function_and_call)
        ev_const = self._make_output_name(stage, "ev_c")
        gain_const = self._make_output_name(stage, "gain_c")
        target_const = self._make_output_name(stage, "target_c")
        one_const = self._make_output_name(stage, "one_c")
        
        nodes, inits, vis = [], [], []
        
        # Add input value info
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # 1. Calculate mean brightness of the image across all channels
        mean_node = oh.make_node("ReduceMean", inputs=[input_image], outputs=[brightness_name], 
                              axes=[2, 3], keepdims=0)
        nodes.append(mean_node)
        
        # 2. Reshape brightness to add batch dimension back
        # Only create initializer - _to_function_and_call will create Constant node
        inits.append(oh.make_tensor(one_const, TensorProto.INT64, [4], [1, 1, 1, 1]))
        reshape_node = oh.make_node("Reshape", inputs=[brightness_name, one_const], 
                                   outputs=[brightness_4d])
        nodes.append(reshape_node)
        
        # 3. Create target brightness (18% gray card standard)
        # Only create initializer - _to_function_and_call will create Constant node
        # Then use Identity to bridge from internal const to function output
        inits.append(oh.make_tensor(target_const, TensorProto.FLOAT, [1], [0.18]))
        # Identity bridge from internal constant to function output
        identity_node = oh.make_node("Identity", inputs=[target_const], outputs=[target_name])
        nodes.append(identity_node)
        
        # 4. Calculate brightness to target ratio
        ratio_node = oh.make_node("Div", inputs=[brightness_4d, target_name], outputs=[ratio_name])
        nodes.append(ratio_node)
        
        # 5. Calculate exposure adjustment
        # Only create initializer
        inits.append(oh.make_tensor(ev_const, TensorProto.FLOAT, [1], [0.0]))
        # Identity bridge from internal constant to function output
        identity_node2 = oh.make_node("Identity", inputs=[ev_const], outputs=[ev_name])
        nodes.append(identity_node2)
        
        # 6. Calculate gain adjustment
        # Only create initializer
        inits.append(oh.make_tensor(gain_const, TensorProto.FLOAT, [1], [1.0]))
        # Identity bridge from internal constant to function output
        identity_node3 = oh.make_node("Identity", inputs=[gain_const], outputs=[gain_name])
        nodes.append(identity_node3)
        
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
        return super().build_coordinator(stage, prev_stages)

    def build_algo(self, stage, prev_stages=None):
        """Build the algorithmic part of the auto exposure block."""
        return self.build_applier(stage, prev_stages)

    def build_test_algo(self, stage, prev_stages=None):
        """Test algorithmic implementation of auto exposure."""
        return self.build_applier(stage, prev_stages)


class AutoExposureV1(AutoExposureBase):
    name = "autoexposure"
    version = "v1.legacy"


class AutoExposurePassthrough(AutoExposureBase):
    name = "autoexposure"
    version = "v1.passthrough"