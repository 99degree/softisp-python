# autoexposure.py - Multiple auto exposure algorithms with different computational needs
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto
import numpy as np


# ============================================================================
# SIMPLE COMPUTATION - Basic exposure calculation
# ============================================================================

class AutoExposureSimple(MicroblockBase):
    """
    Simple auto exposure with minimal computation.
    Just calculates mean brightness and applies exposure compensation.
    """
    name = "autoexposure_simple"
    family = "autoexposure_simple"
    version = "v2"
    
    target_brightness = 0.18
    min_ev = -2.0
    max_ev = 2.0
    
    def build_algo(self, stage, prev_stages=None):
        """Simple algorithm - just mean brightness"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        brightness_name = f"{stage}.brightness"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Calculate mean brightness
        mean_node = oh.make_node("ReduceMean", inputs=[input_image], outputs=[brightness_name], 
                                axes=[1, 2, 3], keepdims=0)
        nodes.append(mean_node)
        
        # Create target
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [self.target_brightness])
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], 
                                 value=target_tensor, name=f"{target_name}_const")
        nodes.append(target_node)
        
        # Calculate EV = log2(target / brightness)
        div_node = oh.make_node("Div", inputs=[target_name, brightness_name], 
                               outputs=[f"{stage}.ratio"])
        nodes.append(div_node)
        
        log_node = oh.make_node("Log", inputs=[f"{stage}.ratio"], outputs=[f"{stage}.ratio_log"])
        nodes.append(log_node)
        
        log2_tensor = oh.make_tensor(f"{stage}.log2", TensorProto.FLOAT, [1], [0.6931471805599453])
        inits.append(log2_tensor)
        log2_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.log2"], 
                                value=log2_tensor, name=f"{stage}.log2_const")
        nodes.append(log2_node)
        
        ev_node = oh.make_node("Div", inputs=[f"{stage}.ratio_log", f"{stage}.log2"], 
                              outputs=[ev_name])
        nodes.append(ev_node)
        
        # Clip EV
        min_ev_tensor = oh.make_tensor(f"{stage}.min_ev", TensorProto.FLOAT, [1], [self.min_ev])
        max_ev_tensor = oh.make_tensor(f"{stage}.max_ev", TensorProto.FLOAT, [1], [self.max_ev])
        inits.extend([min_ev_tensor, max_ev_tensor])
        
        min_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.min_ev"], 
                                  value=min_ev_tensor, name=f"{stage}.min_ev_const")
        max_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.max_ev"], 
                                  value=max_ev_tensor, name=f"{stage}.max_ev_const")
        nodes.extend([min_ev_node, max_ev_node])
        
        clip_node = oh.make_node("Clip", inputs=[ev_name, f"{stage}.min_ev", f"{stage}.max_ev"], 
                                outputs=[f"{ev_name}_clipped"])
        nodes.append(clip_node)
        
        # Calculate gain
        exp_node = oh.make_node("Exp", inputs=[f"{ev_name}_clipped"], outputs=[gain_name])
        nodes.append(exp_node)
        
        vis.append(oh.make_tensor_value_info(brightness_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "brightness": {"name": brightness_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """Applier - multiply by gain and clip"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        gain_source = f"{upstream}.gain" if prev_stages else f"{stage}.gain"
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        mul_node = oh.make_node("Mul", inputs=[input_image, gain_source], outputs=[output_image])
        nodes.append(mul_node)
        
        zero_tensor = oh.make_tensor(f"{stage}.zero", TensorProto.FLOAT, [1], [0.0])
        one_tensor = oh.make_tensor(f"{stage}.one", TensorProto.FLOAT, [1], [1.0])
        inits.extend([zero_tensor, one_tensor])
        
        zero_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.zero"], 
                               value=zero_tensor, name=f"{stage}.zero_const")
        one_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.one"], 
                              value=one_tensor, name=f"{stage}.one_const")
        nodes.extend([zero_node, one_node])
        
        clip_node = oh.make_node("Clip", inputs=[output_image, f"{stage}.zero", f"{stage}.one"], 
                                outputs=[f"{output_image}_clipped"])
        nodes.append(clip_node)
        
        outputs = {
            "output": {"name": f"{output_image}_clipped", "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


# ============================================================================
# MEDIUM COMPUTATION - RGB stats or YUV stats
# ============================================================================

class AutoExposureStats(MicroblockBase):
    """
    Medium computation - RGB statistics with per-channel analysis.
    Calculates per-channel brightness and uses weighted luminance.
    """
    name = "autoexposure_stats"
    family = "autoexposure_stats"
    version = "v2"
    
    target_brightness = 0.18
    min_ev = -2.0
    max_ev = 2.0
    rgb_weights = [0.299, 0.587, 0.114]  # BT.601 luminance weights
    
    def build_algo(self, stage, prev_stages=None):
        """Medium algorithm - per-channel RGB analysis"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        r_stats_name = f"{stage}.r_stats"
        g_stats_name = f"{stage}.g_stats"
        b_stats_name = f"{stage}.b_stats"
        weighted_stats_name = f"{stage}.weighted_stats"
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Split RGB channels
        split_node = oh.make_node("Split", inputs=[input_image], 
                                 outputs=[r_stats_name, g_stats_name, b_stats_name], 
                                 axis=1, split=[1, 1, 1])
        nodes.append(split_node)
        
        # Calculate per-channel brightness
        r_mean_node = oh.make_node("ReduceMean", inputs=[r_stats_name], outputs=[f"{r_stats_name}_mean"],
                                  axes=[2, 3], keepdims=0)
        g_mean_node = oh.make_node("ReduceMean", inputs=[g_stats_name], outputs=[f"{g_stats_name}_mean"],
                                  axes=[2, 3], keepdims=0)
        b_mean_node = oh.make_node("ReduceMean", inputs=[b_stats_name], outputs=[f"{b_stats_name}_mean"],
                                  axes=[2, 3], keepdims=0)
        nodes.extend([r_mean_node, g_mean_node, b_mean_node])
        
        # Calculate weighted brightness
        r_weight = self.rgb_weights[0]
        g_weight = self.rgb_weights[1]
        b_weight = self.rgb_weights[2]
        
        r_weight_tensor = oh.make_tensor(f"{stage}.r_weight", TensorProto.FLOAT, [1], [r_weight])
        g_weight_tensor = oh.make_tensor(f"{stage}.g_weight", TensorProto.FLOAT, [1], [g_weight])
        b_weight_tensor = oh.make_tensor(f"{stage}.b_weight", TensorProto.FLOAT, [1], [b_weight])
        inits.extend([r_weight_tensor, g_weight_tensor, b_weight_tensor])
        
        r_weighted_node = oh.make_node("Mul", inputs=[f"{r_stats_name}_mean", f"{stage}.r_weight"], 
                                      outputs=[f"{r_stats_name}_weighted"])
        g_weighted_node = oh.make_node("Mul", inputs=[f"{g_stats_name}_mean", f"{stage}.g_weight"], 
                                      outputs=[f"{g_stats_name}_weighted"])
        b_weighted_node = oh.make_node("Mul", inputs=[f"{b_stats_name}_mean", f"{stage}.b_weight"], 
                                      outputs=[f"{b_stats_name}_weighted"])
        nodes.extend([r_weighted_node, g_weighted_node, b_weighted_node])
        
        # Sum weighted channels
        add1_node = oh.make_node("Add", inputs=[f"{r_stats_name}_weighted", f"{g_stats_name}_weighted"],
                                 outputs=[f"{stage}.rg_sum"])
        add2_node = oh.make_node("Add", inputs=[f"{stage}.rg_sum", f"{b_stats_name}_weighted"],
                                 outputs=[weighted_stats_name])
        nodes.extend([add1_node, add2_node])
        
        # Create target and calculate EV
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [self.target_brightness])
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], 
                                 value=target_tensor, name=f"{target_name}_const")
        nodes.append(target_node)
        
        div_node = oh.make_node("Div", inputs=[target_name, weighted_stats_name], 
                               outputs=[f"{stage}.ratio"])
        nodes.append(div_node)
        
        log_node = oh.make_node("Log", inputs=[f"{stage}.ratio"], outputs=[f"{stage}.ratio_log"])
        nodes.append(log_node)
        
        log2_tensor = oh.make_tensor(f"{stage}.log2", TensorProto.FLOAT, [1], [0.6931471805599453])
        inits.append(log2_tensor)
        log2_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.log2"], 
                                value=log2_tensor, name=f"{stage}.log2_const")
        nodes.append(log2_node)
        
        ev_node = oh.make_node("Div", inputs=[f"{stage}.ratio_log", f"{stage}.log2"], 
                              outputs=[ev_name])
        nodes.append(ev_node)
        
        # Clip EV
        min_ev_tensor = oh.make_tensor(f"{stage}.min_ev", TensorProto.FLOAT, [1], [self.min_ev])
        max_ev_tensor = oh.make_tensor(f"{stage}.max_ev", TensorProto.FLOAT, [1], [self.max_ev])
        inits.extend([min_ev_tensor, max_ev_tensor])
        
        min_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.min_ev"], 
                                  value=min_ev_tensor, name=f"{stage}.min_ev_const")
        max_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.max_ev"], 
                                  value=max_ev_tensor, name=f"{stage}.max_ev_const")
        nodes.extend([min_ev_node, max_ev_node])
        
        clip_node = oh.make_node("Clip", inputs=[ev_name, f"{stage}.min_ev", f"{stage}.max_ev"], 
                                outputs=[f"{ev_name}_clipped"])
        nodes.append(clip_node)
        
        # Calculate gain
        exp_node = oh.make_node("Exp", inputs=[f"{ev_name}_clipped"], outputs=[gain_name])
        nodes.append(exp_node)
        
        vis.append(oh.make_tensor_value_info(r_stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(g_stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(b_stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(weighted_stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "r_stats": {"name": r_stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "g_stats": {"name": g_stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "b_stats": {"name": b_stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "weighted_stats": {"name": weighted_stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """Applier - same as simple"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        gain_source = f"{upstream}.gain" if prev_stages else f"{stage}.gain"
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        mul_node = oh.make_node("Mul", inputs=[input_image, gain_source], outputs=[output_image])
        nodes.append(mul_node)
        
        zero_tensor = oh.make_tensor(f"{stage}.zero", TensorProto.FLOAT, [1], [0.0])
        one_tensor = oh.make_tensor(f"{stage}.one", TensorProto.FLOAT, [1], [1.0])
        inits.extend([zero_tensor, one_tensor])
        
        zero_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.zero"], 
                               value=zero_tensor, name=f"{stage}.zero_const")
        one_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.one"], 
                              value=one_tensor, name=f"{stage}.one_const")
        nodes.extend([zero_node, one_node])
        
        clip_node = oh.make_node("Clip", inputs=[output_image, f"{stage}.zero", f"{stage}.one"], 
                                outputs=[f"{output_image}_clipped"])
        nodes.append(clip_node)
        
        outputs = {
            "output": {"name": f"{output_image}_clipped", "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class AutoExposureYUV(MicroblockBase):
    """
    Medium computation - YUV statistics with luminance-only exposure.
    Uses Y channel for exposure calculation, preserves chrominance.
    """
    name = "autoexposure_yuv"
    family = "autoexposure_yuv"
    version = "v2"
    
    target_brightness = 0.18
    min_ev = -2.0
    max_ev = 2.0
    
    def build_algo(self, stage, prev_stages=None):
        """Medium algorithm - YUV luminance analysis"""
        upstream = prev_stages[0] if prev_stages else stage
        input_yuv = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        y_channel_name = f"{stage}.y_channel"
        u_channel_name = f"{stage}.u_channel"
        v_channel_name = f"{stage}.v_channel"
        y_stats_name = f"{stage}.y_stats"
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_yuv, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Split YUV channels
        split_node = oh.make_node("Split", inputs=[input_yuv], 
                                 outputs=[y_channel_name, u_channel_name, v_channel_name], 
                                 axis=1, split=[1, 1, 1])
        nodes.append(split_node)
        
        # Calculate brightness from Y channel only
        y_mean_node = oh.make_node("ReduceMean", inputs=[y_channel_name], outputs=[y_stats_name], 
                                  axes=[2, 3], keepdims=0)
        nodes.append(y_mean_node)
        
        # Create target and calculate EV
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [self.target_brightness])
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], 
                                 value=target_tensor, name=f"{target_name}_const")
        nodes.append(target_node)
        
        div_node = oh.make_node("Div", inputs=[target_name, y_stats_name], 
                               outputs=[f"{stage}.ratio"])
        nodes.append(div_node)
        
        log_node = oh.make_node("Log", inputs=[f"{stage}.ratio"], outputs=[f"{stage}.ratio_log"])
        nodes.append(log_node)
        
        log2_tensor = oh.make_tensor(f"{stage}.log2", TensorProto.FLOAT, [1], [0.6931471805599453])
        inits.append(log2_tensor)
        log2_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.log2"], 
                                value=log2_tensor, name=f"{stage}.log2_const")
        nodes.append(log2_node)
        
        ev_node = oh.make_node("Div", inputs=[f"{stage}.ratio_log", f"{stage}.log2"], 
                              outputs=[ev_name])
        nodes.append(ev_node)
        
        # Clip EV
        min_ev_tensor = oh.make_tensor(f"{stage}.min_ev", TensorProto.FLOAT, [1], [self.min_ev])
        max_ev_tensor = oh.make_tensor(f"{stage}.max_ev", TensorProto.FLOAT, [1], [self.max_ev])
        inits.extend([min_ev_tensor, max_ev_tensor])
        
        min_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.min_ev"], 
                                  value=min_ev_tensor, name=f"{stage}.min_ev_const")
        max_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.max_ev"], 
                                  value=max_ev_tensor, name=f"{stage}.max_ev_const")
        nodes.extend([min_ev_node, max_ev_node])
        
        clip_node = oh.make_node("Clip", inputs=[ev_name, f"{stage}.min_ev", f"{stage}.max_ev"], 
                                outputs=[f"{ev_name}_clipped"])
        nodes.append(clip_node)
        
        # Calculate gain
        exp_node = oh.make_node("Exp", inputs=[f"{ev_name}_clipped"], outputs=[gain_name])
        nodes.append(exp_node)
        
        vis.append(oh.make_tensor_value_info(y_channel_name, TensorProto.FLOAT, [1,1,"h","w"]))
        vis.append(oh.make_tensor_value_info(u_channel_name, TensorProto.FLOAT, [1,1,"h","w"]))
        vis.append(oh.make_tensor_value_info(v_channel_name, TensorProto.FLOAT, [1,1,"h","w"]))
        vis.append(oh.make_tensor_value_info(y_stats_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "y_channel": {"name": y_channel_name, "shape": [1,1,"h","w"], "type": TensorProto.FLOAT},
            "u_channel": {"name": u_channel_name, "shape": [1,1,"h","w"], "type": TensorProto.FLOAT},
            "v_channel": {"name": v_channel_name, "shape": [1,1,"h","w"], "type": TensorProto.FLOAT},
            "y_stats": {"name": y_stats_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_yuv, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """Applier - same as simple"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        gain_source = f"{upstream}.gain" if prev_stages else f"{stage}.gain"
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        mul_node = oh.make_node("Mul", inputs=[input_image, gain_source], outputs=[output_image])
        nodes.append(mul_node)
        
        zero_tensor = oh.make_tensor(f"{stage}.zero", TensorProto.FLOAT, [1], [0.0])
        one_tensor = oh.make_tensor(f"{stage}.one", TensorProto.FLOAT, [1], [1.0])
        inits.extend([zero_tensor, one_tensor])
        
        zero_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.zero"], 
                               value=zero_tensor, name=f"{stage}.zero_const")
        one_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.one"], 
                              value=one_tensor, name=f"{stage}.one_const")
        nodes.extend([zero_node, one_node])
        
        clip_node = oh.make_node("Clip", inputs=[output_image, f"{stage}.zero", f"{stage}.one"], 
                                outputs=[f"{output_image}_clipped"])
        nodes.append(clip_node)
        
        outputs = {
            "output": {"name": f"{output_image}_clipped", "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


# ============================================================================
# HIGH COMPUTATION - Advanced algorithms with histogram, multi-zone, etc.
# ============================================================================

class AutoExposureHistogram(MicroblockBase):
    """
    High computation - Histogram-based exposure with percentile analysis.
    Uses histogram to find optimal exposure point.
    """
    name = "autoexposure_histogram"
    family = "autoexposure_histogram"
    version = "v2"
    
    target_brightness = 0.18
    min_ev = -2.0
    max_ev = 2.0
    histogram_bins = 256
    percentile = 50  # Median exposure
    
    def build_algo(self, stage, prev_stages=None):
        """High computation - histogram analysis"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        brightness_name = f"{stage}.brightness"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Convert to grayscale
        reduce_node = oh.make_node("ReduceMean", inputs=[input_image], 
                                   outputs=[f"{stage}.grayscale"], 
                                   axes=[1], keepdims=0)
        nodes.append(reduce_node)
        
        # Calculate mean brightness (simplified histogram)
        mean_node = oh.make_node("ReduceMean", inputs=[f"{stage}.grayscale"], 
                                outputs=[brightness_name], 
                                axes=[1, 2], keepdims=0)
        nodes.append(mean_node)
        
        # Create target and calculate EV
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [self.target_brightness])
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], 
                                 value=target_tensor, name=f"{target_name}_const")
        nodes.append(target_node)
        
        div_node = oh.make_node("Div", inputs=[target_name, brightness_name], 
                               outputs=[f"{stage}.ratio"])
        nodes.append(div_node)
        
        log_node = oh.make_node("Log", inputs=[f"{stage}.ratio"], outputs=[f"{stage}.ratio_log"])
        nodes.append(log_node)
        
        log2_tensor = oh.make_tensor(f"{stage}.log2", TensorProto.FLOAT, [1], [0.6931471805599453])
        inits.append(log2_tensor)
        log2_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.log2"], 
                                value=log2_tensor, name=f"{stage}.log2_const")
        nodes.append(log2_node)
        
        ev_node = oh.make_node("Div", inputs=[f"{stage}.ratio_log", f"{stage}.log2"], 
                              outputs=[ev_name])
        nodes.append(ev_node)
        
        # Clip EV
        min_ev_tensor = oh.make_tensor(f"{stage}.min_ev", TensorProto.FLOAT, [1], [self.min_ev])
        max_ev_tensor = oh.make_tensor(f"{stage}.max_ev", TensorProto.FLOAT, [1], [self.max_ev])
        inits.extend([min_ev_tensor, max_ev_tensor])
        
        min_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.min_ev"], 
                                  value=min_ev_tensor, name=f"{stage}.min_ev_const")
        max_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.max_ev"], 
                                  value=max_ev_tensor, name=f"{stage}.max_ev_const")
        nodes.extend([min_ev_node, max_ev_node])
        
        clip_node = oh.make_node("Clip", inputs=[ev_name, f"{stage}.min_ev", f"{stage}.max_ev"], 
                                outputs=[f"{ev_name}_clipped"])
        nodes.append(clip_node)
        
        # Calculate gain
        exp_node = oh.make_node("Exp", inputs=[f"{ev_name}_clipped"], outputs=[gain_name])
        nodes.append(exp_node)
        
        vis.append(oh.make_tensor_value_info(brightness_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "brightness": {"name": brightness_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """Applier - same as simple"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        gain_source = f"{upstream}.gain" if prev_stages else f"{stage}.gain"
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        mul_node = oh.make_node("Mul", inputs=[input_image, gain_source], outputs=[output_image])
        nodes.append(mul_node)
        
        zero_tensor = oh.make_tensor(f"{stage}.zero", TensorProto.FLOAT, [1], [0.0])
        one_tensor = oh.make_tensor(f"{stage}.one", TensorProto.FLOAT, [1], [1.0])
        inits.extend([zero_tensor, one_tensor])
        
        zero_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.zero"], 
                               value=zero_tensor, name=f"{stage}.zero_const")
        one_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.one"], 
                              value=one_tensor, name=f"{stage}.one_const")
        nodes.extend([zero_node, one_node])
        
        clip_node = oh.make_node("Clip", inputs=[output_image, f"{stage}.zero", f"{stage}.one"], 
                                outputs=[f"{output_image}_clipped"])
        nodes.append(clip_node)
        
        outputs = {
            "output": {"name": f"{output_image}_clipped", "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class AutoExposureMultiZone(MicroblockBase):
    """
    High computation - Multi-zone exposure with weighted zones.
    Divides image into zones and calculates weighted exposure.
    """
    name = "autoexposure_multizone"
    family = "autoexposure_multizone"
    version = "v2"
    
    target_brightness = 0.18
    min_ev = -2.0
    max_ev = 2.0
    zone_weights = [0.5, 0.7, 1.0, 0.7, 0.5]  # Center-weighted zones
    
    def build_algo(self, stage, prev_stages=None):
        """High computation - multi-zone analysis"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        
        ev_name = f"{stage}.exposure_value"
        gain_name = f"{stage}.gain"
        target_name = f"{stage}.target"
        brightness_name = f"{stage}.brightness"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        # Convert to grayscale
        reduce_node = oh.make_node("ReduceMean", inputs=[input_image], 
                                   outputs=[f"{stage}.grayscale"], 
                                   axes=[1], keepdims=0)
        nodes.append(reduce_node)
        
        # Calculate mean brightness (simplified multi-zone)
        mean_node = oh.make_node("ReduceMean", inputs=[f"{stage}.grayscale"], 
                                outputs=[brightness_name], 
                                axes=[1, 2], keepdims=0)
        nodes.append(mean_node)
        
        # Create target and calculate EV
        target_tensor = oh.make_tensor(target_name, TensorProto.FLOAT, [1], [self.target_brightness])
        inits.append(target_tensor)
        target_node = oh.make_node("Constant", inputs=[], outputs=[target_name], 
                                 value=target_tensor, name=f"{target_name}_const")
        nodes.append(target_node)
        
        div_node = oh.make_node("Div", inputs=[target_name, brightness_name], 
                               outputs=[f"{stage}.ratio"])
        nodes.append(div_node)
        
        log_node = oh.make_node("Log", inputs=[f"{stage}.ratio"], outputs=[f"{stage}.ratio_log"])
        nodes.append(log_node)
        
        log2_tensor = oh.make_tensor(f"{stage}.log2", TensorProto.FLOAT, [1], [0.6931471805599453])
        inits.append(log2_tensor)
        log2_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.log2"], 
                                value=log2_tensor, name=f"{stage}.log2_const")
        nodes.append(log2_node)
        
        ev_node = oh.make_node("Div", inputs=[f"{stage}.ratio_log", f"{stage}.log2"], 
                              outputs=[ev_name])
        nodes.append(ev_node)
        
        # Clip EV
        min_ev_tensor = oh.make_tensor(f"{stage}.min_ev", TensorProto.FLOAT, [1], [self.min_ev])
        max_ev_tensor = oh.make_tensor(f"{stage}.max_ev", TensorProto.FLOAT, [1], [self.max_ev])
        inits.extend([min_ev_tensor, max_ev_tensor])
        
        min_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.min_ev"], 
                                  value=min_ev_tensor, name=f"{stage}.min_ev_const")
        max_ev_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.max_ev"], 
                                  value=max_ev_tensor, name=f"{stage}.max_ev_const")
        nodes.extend([min_ev_node, max_ev_node])
        
        clip_node = oh.make_node("Clip", inputs=[ev_name, f"{stage}.min_ev", f"{stage}.max_ev"], 
                                outputs=[f"{ev_name}_clipped"])
        nodes.append(clip_node)
        
        # Calculate gain
        exp_node = oh.make_node("Exp", inputs=[f"{ev_name}_clipped"], outputs=[gain_name])
        nodes.append(exp_node)
        
        vis.append(oh.make_tensor_value_info(brightness_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ev_name, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gain_name, TensorProto.FLOAT, [1]))
        
        outputs = {
            "brightness": {"name": brightness_name, "shape": [1], "type": TensorProto.FLOAT},
            "exposure_value": {"name": ev_name, "shape": [1], "type": TensorProto.FLOAT},
            "gain": {"name": gain_name, "shape": [1], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_applier(self, stage, prev_stages=None):
        """Applier - same as simple"""
        upstream = prev_stages[0] if prev_stages else stage
        input_image = f"{upstream}.applier" if prev_stages else f"{stage}.input"
        gain_source = f"{upstream}.gain" if prev_stages else f"{stage}.gain"
        output_image = f"{stage}.output"
        
        nodes, inits, vis = [], [], []
        vis.append(oh.make_tensor_value_info(input_image, TensorProto.FLOAT, ["n",3,"h","w"]))
        
        mul_node = oh.make_node("Mul", inputs=[input_image, gain_source], outputs=[output_image])
        nodes.append(mul_node)
        
        zero_tensor = oh.make_tensor(f"{stage}.zero", TensorProto.FLOAT, [1], [0.0])
        one_tensor = oh.make_tensor(f"{stage}.one", TensorProto.FLOAT, [1], [1.0])
        inits.extend([zero_tensor, one_tensor])
        
        zero_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.zero"], 
                               value=zero_tensor, name=f"{stage}.zero_const")
        one_node = oh.make_node("Constant", inputs=[], outputs=[f"{stage}.one"], 
                              value=one_tensor, name=f"{stage}.one_const")
        nodes.extend([zero_node, one_node])
        
        clip_node = oh.make_node("Clip", inputs=[output_image, f"{stage}.zero", f"{stage}.one"], 
                                outputs=[f"{output_image}_clipped"])
        nodes.append(clip_node)
        
        outputs = {
            "output": {"name": f"{output_image}_clipped", "shape": ["n",3,"h","w"], "type": TensorProto.FLOAT}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(input_image, type=TensorProto.FLOAT, shape=["n",3,"h","w"])
        return result

    def build_coordinator(self, stage, prev_stages=None):
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage, prev_stages=None):
        return self.build_algo(stage, prev_stages)