from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class ExposureFusionApplierBase(MicroblockBase):
    """
    ExposureFusionApplierBase (v0)
    -----------------------------
    APPLIER: Apply tone mapping to fused frame.

    Position: After control loop, applies tone mapping.
    
    Purpose: Apply tone mapping to fused HDR frame for display.
    
    Needs:
        - fused_frame [n,3,h,w] : Fused HDR frame
        - dynamic_range [1] : Estimated dynamic range

    Provides:
        - output_frame [n,3,h,w] : Tone-mapped output frame

    Behavior:
        - build_algo: Not used (algo extracts exposure info)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies tone mapping

    Tone Mapping:
        - Simple gamma correction
        - Linear scaling to [0, 1]
        - Clipping to valid range

    Complexity: ~15-25 ONNX nodes
    Use Case: Real-time tone mapping
    """
    name = 'exposure_fusion_applier_base'
    family = 'exposure_fusion_applier_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts exposure info.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply tone mapping to fused frame.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        fused_frame = f'{upstream}.fused_frame'
        dynamic_range = f'{upstream}.dynamic_range'
        output_frame = f'{stage}.output_frame'
        
        # Normalize to [0, 1] based on dynamic range
        normalized = f'{stage}.normalized'
        nodes.append(oh.make_node('Div', inputs=[fused_frame, dynamic_range], outputs=[normalized],
                                  name=f'{stage}.div_normalize'))
        
        # Apply gamma correction
        gamma = f'{stage}.gamma'
        gamma_inv = f'{stage}.gamma_inv'
        inits.append(oh.make_tensor(gamma, TensorProto.FLOAT, [], [2.2]))
        inits.append(oh.make_tensor(gamma_inv, TensorProto.FLOAT, [], [1.0/2.2]))
        
        gamma_corrected = f'{stage}.gamma_corrected'
        nodes.append(oh.make_node('Pow', inputs=[normalized, gamma_inv], outputs=[gamma_corrected],
                                  name=f'{stage}.pow_gamma'))
        
        # Clip to [0, 1]
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        clipped = f'{stage}.clipped'
        nodes.append(oh.make_node('Clip', inputs=[gamma_corrected, zero, one], outputs=[clipped],
                                  name=f'{stage}.clip'))
        
        # Scale to [0, 255]
        scale = f'{stage}.scale'
        inits.append(oh.make_tensor(scale, TensorProto.FLOAT, [], [255.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[clipped, scale], outputs=[output_frame],
                                  name=f'{stage}.mul_scale'))
        
        vis.append(oh.make_tensor_value_info(output_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        
        outputs = {'output_frame': {'name': output_frame}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(fused_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(dynamic_range, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class ExposureFusionApplierV1(MicroblockBase):
    """
    ExposureFusionApplierV1 (v1)
    ----------------------------
    APPLIER: Apply adaptive tone mapping.

    Position: After control loop, applies tone mapping.
    
    Purpose: Apply adaptive tone mapping based on content.
    
    Needs:
        - fused_frame [n,3,h,w] : Fused HDR frame
        - dynamic_range [1] : Estimated dynamic range
        - target_brightness [1] : Target brightness (0-1)

    Provides:
        - output_frame [n,3,h,w] : Tone-mapped output frame
        - exposure_compensation [1] : Applied exposure compensation

    Behavior:
        - build_algo: Not used (algo extracts exposure info)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies adaptive tone mapping

    Tone Mapping:
        - Adaptive exposure compensation
        - Local tone mapping
        - Contrast enhancement

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality tone mapping
    """
    name = 'exposure_fusion_applier_v1'
    family = 'exposure_fusion_applier_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts exposure info.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply adaptive tone mapping to fused frame.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        fused_frame = f'{upstream}.fused_frame'
        dynamic_range = f'{upstream}.dynamic_range'
        target_brightness = f'{upstream}.target_brightness'
        output_frame = f'{stage}.output_frame'
        
        # Calculate current brightness
        current_brightness = f'{stage}.current_brightness'
        nodes.append(oh.make_node('ReduceMean', inputs=[fused_frame], outputs=[current_brightness],
                                  name=f'{stage}.reduce_mean_brightness',
                                  axes=[1, 2, 3], keepdims=0))
        
        # Calculate exposure compensation
        exposure_compensation = f'{stage}.exposure_compensation'
        brightness_ratio = f'{stage}.brightness_ratio'
        nodes.append(oh.make_node('Div', inputs=[target_brightness, current_brightness], outputs=[brightness_ratio],
                                  name=f'{stage}.div_brightness_ratio'))
        
        # Clamp exposure compensation
        min_exp = f'{stage}.min_exp'
        max_exp = f'{stage}.max_exp'
        inits.append(oh.make_tensor(min_exp, TensorProto.FLOAT, [], [0.1]))
        inits.append(oh.make_tensor(max_exp, TensorProto.FLOAT, [], [10.0]))
        
        nodes.append(oh.make_node('Clip', inputs=[brightness_ratio, min_exp, max_exp], outputs=[exposure_compensation],
                                  name=f'{stage}.clip_exposure'))
        
        # Apply exposure compensation
        exposed = f'{stage}.exposed'
        nodes.append(oh.make_node('Mul', inputs=[fused_frame, exposure_compensation], outputs=[exposed],
                                  name=f'{stage}.mul_exposure'))
        
        # Normalize to [0, 1]
        normalized = f'{stage}.normalized'
        nodes.append(oh.make_node('Div', inputs=[exposed, dynamic_range], outputs=[normalized],
                                  name=f'{stage}.div_normalize'))
        
        # Apply gamma correction
        gamma_inv = f'{stage}.gamma_inv'
        inits.append(oh.make_tensor(gamma_inv, TensorProto.FLOAT, [], [1.0/2.2]))
        
        gamma_corrected = f'{stage}.gamma_corrected'
        nodes.append(oh.make_node('Pow', inputs=[normalized, gamma_inv], outputs=[gamma_corrected],
                                  name=f'{stage}.pow_gamma'))
        
        # Clip to [0, 1]
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        clipped = f'{stage}.clipped'
        nodes.append(oh.make_node('Clip', inputs=[gamma_corrected, zero, one], outputs=[clipped],
                                  name=f'{stage}.clip'))
        
        # Scale to [0, 255]
        scale = f'{stage}.scale'
        inits.append(oh.make_tensor(scale, TensorProto.FLOAT, [], [255.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[clipped, scale], outputs=[output_frame],
                                  name=f'{stage}.mul_scale'))
        
        vis.append(oh.make_tensor_value_info(output_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(exposure_compensation, TensorProto.FLOAT, [1]))
        
        outputs = {
            'output_frame': {'name': output_frame},
            'exposure_compensation': {'name': exposure_compensation}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(fused_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(dynamic_range, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(target_brightness, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class ExposureFusionApplierV2(MicroblockBase):
    """
    ExposureFusionApplierV2 (v2)
    ----------------------------
    APPLIER: Apply professional tone mapping with local adaptation.

    Position: After control loop, applies tone mapping.
    
    Purpose: Apply professional tone mapping with local adaptation.
    
    Needs:
        - fused_frame [n,3,h,w] : Fused HDR frame
        - dynamic_range [1] : Estimated dynamic range
        - target_brightness [1] : Target brightness (0-1)
        - local_adaptation [1] : Local adaptation strength (0-1)

    Provides:
        - output_frame [n,3,h,w] : Tone-mapped output frame
        - exposure_compensation [1] : Applied exposure compensation
        - local_luminance [n,1,h,w] : Local luminance map

    Behavior:
        - build_algo: Not used (algo extracts exposure info)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies professional tone mapping

    Tone Mapping:
        - Global exposure compensation
        - Local luminance adaptation
        - Contrast enhancement
        - Color preservation

    Complexity: ~50-70 ONNX nodes
    Use Case: Professional tone mapping
    """
    name = 'exposure_fusion_applier_v2'
    family = 'exposure_fusion_applier_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts exposure info.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply professional tone mapping with local adaptation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        fused_frame = f'{upstream}.fused_frame'
        dynamic_range = f'{upstream}.dynamic_range'
        target_brightness = f'{upstream}.target_brightness'
        local_adaptation = f'{upstream}.local_adaptation'
        output_frame = f'{stage}.output_frame'
        
        # Calculate current brightness
        current_brightness = f'{stage}.current_brightness'
        nodes.append(oh.make_node('ReduceMean', inputs=[fused_frame], outputs=[current_brightness],
                                  name=f'{stage}.reduce_mean_brightness',
                                  axes=[1, 2, 3], keepdims=0))
        
        # Calculate exposure compensation
        exposure_compensation = f'{stage}.exposure_compensation'
        brightness_ratio = f'{stage}.brightness_ratio'
        nodes.append(oh.make_node('Div', inputs=[target_brightness, current_brightness], outputs=[brightness_ratio],
                                  name=f'{stage}.div_brightness_ratio'))
        
        # Clamp exposure compensation
        min_exp = f'{stage}.min_exp'
        max_exp = f'{stage}.max_exp'
        inits.append(oh.make_tensor(min_exp, TensorProto.FLOAT, [], [0.1]))
        inits.append(oh.make_tensor(max_exp, TensorProto.FLOAT, [], [10.0]))
        
        nodes.append(oh.make_node('Clip', inputs=[brightness_ratio, min_exp, max_exp], outputs=[exposure_compensation],
                                  name=f'{stage}.clip_exposure'))
        
        # Apply exposure compensation
        exposed = f'{stage}.exposed'
        nodes.append(oh.make_node('Mul', inputs=[fused_frame, exposure_compensation], outputs=[exposed],
                                  name=f'{stage}.mul_exposure'))
        
        # Calculate local luminance
        luminance = f'{stage}.luminance'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        nodes.append(oh.make_node('ReduceMean', inputs=[exposed], outputs=[luminance],
                                  name=f'{stage}.reduce_mean_luminance',
                                  axes=[1], keepdims=1))
        
        # Apply Gaussian blur for local adaptation (simplified: use average pooling)
        local_luminance = f'{stage}.local_luminance'
        kernel_size = f'{stage}.kernel_size'
        inits.append(oh.make_tensor(kernel_size, TensorProto.INT64, [], [5]))
        
        # Simplified: use Resize for local adaptation
        local_luminance_small = f'{stage}.local_luminance_small'
        h_small = f'{stage}.h_small'
        w_small = f'{stage}.w_small'
        vis.append(oh.make_tensor_value_info(h_small, TensorProto.FLOAT, []))
        vis.append(oh.make_tensor_value_info(w_small, TensorProto.FLOAT, []))
        
        nodes.append(oh.make_node('Resize', inputs=[luminance, h_small, w_small], outputs=[local_luminance_small],
                                  name=f'{stage}.resize_luminance_small',
                                  mode='nearest'))
        
        local_luminance_full = f'{stage}.local_luminance_full'
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        nodes.append(oh.make_node('Resize', inputs=[local_luminance_small, h_coord, w_coord], outputs=[local_luminance_full],
                                  name=f'{stage}.resize_luminance_full',
                                  mode='bilinear'))
        
        # Blend global and local luminance
        luminance_adapted = f'{stage}.luminance_adapted'
        one_minus_local = f'{stage}.one_minus_local'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        nodes.append(oh.make_node('Sub', inputs=[one, local_adaptation], outputs=[one_minus_local],
                                  name=f'{stage}.sub_one_minus_local'))
        
        luminance_global = f'{stage}.luminance_global'
        nodes.append(oh.make_node('Mul', inputs=[luminance, one_minus_local], outputs=[luminance_global],
                                  name=f'{stage}.mul_luminance_global'))
        
        luminance_local = f'{stage}.luminance_local'
        nodes.append(oh.make_node('Mul', inputs=[local_luminance_full, local_adaptation], outputs=[luminance_local],
                                  name=f'{stage}.mul_luminance_local'))
        
        nodes.append(oh.make_node('Add', inputs=[luminance_global, luminance_local], outputs=[luminance_adapted],
                                  name=f'{stage}.add_luminance_adapted'))
        
        # Apply tone mapping using adapted luminance
        tone_mapped = f'{stage}.tone_mapped'
        nodes.append(oh.make_node('Div', inputs=[exposed, luminance_adapted], outputs=[tone_mapped],
                                  name=f'{stage}.div_tone_map'))
        
        # Normalize to [0, 1]
        normalized = f'{stage}.normalized'
        nodes.append(oh.make_node('Div', inputs=[tone_mapped, dynamic_range], outputs=[normalized],
                                  name=f'{stage}.div_normalize'))
        
        # Apply gamma correction
        gamma_inv = f'{stage}.gamma_inv'
        inits.append(oh.make_tensor(gamma_inv, TensorProto.FLOAT, [], [1.0/2.2]))
        
        gamma_corrected = f'{stage}.gamma_corrected'
        nodes.append(oh.make_node('Pow', inputs=[normalized, gamma_inv], outputs=[gamma_corrected],
                                  name=f'{stage}.pow_gamma'))
        
        # Clip to [0, 1]
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        clipped = f'{stage}.clipped'
        nodes.append(oh.make_node('Clip', inputs=[gamma_corrected, zero, one], outputs=[clipped],
                                  name=f'{stage}.clip'))
        
        # Scale to [0, 255]
        scale = f'{stage}.scale'
        inits.append(oh.make_tensor(scale, TensorProto.FLOAT, [], [255.0]))
        
        nodes.append(oh.make_node('Mul', inputs=[clipped, scale], outputs=[output_frame],
                                  name=f'{stage}.mul_scale'))
        
        vis.append(oh.make_tensor_value_info(output_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(exposure_compensation, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(local_luminance, TensorProto.FLOAT, ['n', 1, 'h', 'w']))
        
        outputs = {
            'output_frame': {'name': output_frame},
            'exposure_compensation': {'name': exposure_compensation},
            'local_luminance': {'name': local_luminance}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(fused_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(dynamic_range, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(target_brightness, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(local_adaptation, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)