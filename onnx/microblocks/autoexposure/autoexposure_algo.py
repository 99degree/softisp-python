# autoexposure.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


def _n(stage, suffix):
    """Generate unique node name per stage"""
    return f"{stage}.{suffix}"


class AutoExposureBase(MicroblockBase):
    """Auto Exposure Base Class - DO NOT REGISTER"""
    
    name = None
    version = None
    
    def build_algo(self, stage, prev_stages=None):
        """
        Algorithm implementation that calculates exposure statistics.
        This is the stats calculation part.
        """
        return self._calculate_stats(stage, prev_stages)
    
    def _calculate_stats(self, stage, prev_stages=None):
        """Internal method to calculate statistics"""
        # Get input from previous stage
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        # Output names for statistics
        stats_name = _n(stage, "stats")
        ev_name = _n(stage, "ev")
        gain_name = _n(stage, "gain")
        
        nodes, inits, vis = [], [], []
        
        # Add input value info
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Calculate mean brightness of the image
        mean_node = oh.make_node("ReduceMean", inputs=[input_image], outputs=[stats_name], 
                              axes=[2, 3], keepdims=0)
        nodes.append(mean_node)
        
        # Create exposure value calculation (simplified)
        ev_tensor = oh.make_tensor(ev_name + "_c", TensorProto.FLOAT, [1], [1.0])
        gain_tensor = oh.make_tensor(gain_name + "_c", TensorProto.FLOAT, [1], [1.0])
        
        inits.extend([ev_tensor, gain_tensor])
        
        # Create constant nodes
        ev_node = oh.make_node("Constant", inputs=[], outputs=[ev_name], value=ev_tensor, 
                             name=ev_name + "_c")
        gain_node = oh.make_node("Constant", inputs=[], outputs=[gain_name], value=gain_tensor, 
                                name=gain_name + "_c")
        
        nodes.extend([ev_node, gain_node])
        
        # Add output value info
        vis.append(oh.make_tensor_value_info(stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "stats": {"name": stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """
        Applier implementation that uses coefficients from algo stage.
        This applies the calculated exposure compensation.
        """
        # Get coefficients from algorithm stage
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        coeffs_source = f"{upstream}.exposure_value" if prev_stages else f"{stage}.exposure_value"
        
        # Apply exposure compensation
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        
        # Add input value info
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Apply exposure compensation (multiply by exposure value)
        mul_node = oh.make_node("Mul", inputs=[input_image, coeffs_source], outputs=[output_image])
        nodes.append(mul_node)
        
        outputs = {
            "output": {"name": output_image, "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage, prev_stages=None):
        """Test algorithm implementation"""
        return self.build_algo(stage, prev_stages)


class AutoExposureStats(AutoExposureBase):
    """
    Statistics-based auto exposure approach.
    Calculates stats directly from RGB image.
    """
    name = "autoexposure_stats"
    version = "v1"


class AutoExposureYUV(AutoExposureBase):
    """
    YUV-based auto exposure approach.
    Uses YUV statistics for exposure calculation.
    """
    name = "autoexposure_yuv"
    version = "v1"
    
    def _calculate_stats(self, stage, prev_stages=None):
        """Calculate statistics from YUV input"""
        # Get input from previous stage (should be YUV)
        upstream = prev_stages[0] if prev_stages else stage
        input_yuv = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        # Output names for statistics
        stats_name = _n(stage, "stats")
        ev_name = _n(stage, "ev")
        gain_name = _n(stage, "gain")
        
        nodes, inits, vis = [], [], []
        
        # Add input value info (YUV format)
        vis.append(oh.make_tensor_value_info(input_yuv, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Calculate mean brightness from Y channel (first channel)
        # First extract Y channel
        y_channel = f"{stage}.y_channel"
        slice_node = oh.make_node("Slice", inputs=[input_yuv, "zero_const", "one_const", "axis_const"], 
                                 outputs=[y_channel])
        nodes.append(slice_node)
        
        # Add slice constants
        zero_tensor = oh.make_tensor("zero_const", TensorProto.INT64, [1], [0])
        one_tensor = oh.make_tensor("one_const", TensorProto.INT64, [1], [1])
        axis_tensor = oh.make_tensor("axis_const", TensorProto.INT64, [1], [1])
        inits.extend([zero_tensor, one_tensor, axis_tensor])
        
        # Calculate mean brightness from Y channel
        mean_node = oh.make_node("ReduceMean", inputs=[y_channel], outputs=[stats_name], 
                              axes=[2, 3], keepdims=0)
        nodes.append(mean_node)
        
        # Create exposure value calculation
        ev_tensor = oh.make_tensor(ev_name + "_c", TensorProto.FLOAT, [1], [1.0])
        gain_tensor = oh.make_tensor(gain_name + "_c", TensorProto.FLOAT, [1], [1.0])
        
        inits.extend([ev_tensor, gain_tensor])
        
        # Create constant nodes
        ev_node = oh.make_node("Constant", inputs=[], outputs=[ev_name], value=ev_tensor, 
                             name=ev_name + "_c")
        gain_node = oh.make_node("Constant", inputs=[], outputs=[gain_name], value=gain_tensor, 
                                name=gain_name + "_c")
        
        nodes.extend([ev_node, gain_node])
        
        # Add output value info
        vis.append(oh.make_tensor_value_info(stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "stats": {"name": stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_yuv, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result