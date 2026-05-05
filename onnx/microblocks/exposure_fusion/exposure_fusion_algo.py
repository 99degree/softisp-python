from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class ExposureFusionAlgoBase(MicroblockBase):
    """
    ExposureFusionAlgoBase (v0)
    -------------------------
    ALGORITHM: Extract exposure information from multiple frames.

    Position: Post-process stage after YUV/RGB conversion.
    
    Purpose: Analyze exposure levels and calculate exposure ratios.
    
    Needs:
        - frames [n,m,3,h,w] : Multiple exposure frames (m exposures)
        - exposure_values [m] : Exposure values for each frame

    Provides:
        - exposure_ratios [m] : Normalized exposure ratios
        - brightness [m] : Average brightness per frame
        - contrast [m] : Contrast per frame
        - saturation [m] : Saturation per frame

    Behavior:
        - build_algo: Extracts exposure information from frames
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles fusion)

    Complexity: ~20-30 ONNX nodes
    Use Case: Real-time exposure analysis
    """
    name = 'exposure_fusion_algo_base'
    family = 'exposure_fusion_algo_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract exposure information from multiple frames.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        frames = f'{upstream}.frames'
        exposure_values = f'{upstream}.exposure_values'
        
        # Calculate brightness per frame
        brightness = f'{stage}.brightness'
        nodes.append(oh.make_node('ReduceMean', inputs=[frames], outputs=[brightness],
                                  name=f'{stage}.reduce_mean_brightness',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate contrast per frame (simplified: std dev)
        contrast = f'{stage}.contrast'
        nodes.append(oh.make_node('ReduceStd', inputs=[frames], outputs=[contrast],
                                  name=f'{stage}.reduce_std_contrast',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate saturation per frame (simplified: max - min)
        frame_max = f'{stage}.frame_max'
        frame_min = f'{stage}.frame_min'
        nodes.append(oh.make_node('ReduceMax', inputs=[frames], outputs=[frame_max],
                                  name=f'{stage}.reduce_max',
                                  axes=[2, 3, 4], keepdims=0))
        nodes.append(oh.make_node('ReduceMin', inputs=[frames], outputs=[frame_min],
                                  name=f'{stage}.reduce_min',
                                  axes=[2, 3, 4], keepdims=0))
        
        saturation = f'{stage}.saturation'
        nodes.append(oh.make_node('Sub', inputs=[frame_max, frame_min], outputs=[saturation],
                                  name=f'{stage}.sub_saturation'))
        
        # Calculate exposure ratios (normalized)
        exposure_sum = f'{stage}.exposure_sum'
        nodes.append(oh.make_node('ReduceSum', inputs=[exposure_values], outputs=[exposure_sum],
                                  name=f'{stage}.reduce_sum_exposure',
                                  axes=[0], keepdims=1))
        
        exposure_ratios = f'{stage}.exposure_ratios'
        nodes.append(oh.make_node('Div', inputs=[exposure_values, exposure_sum], outputs=[exposure_ratios],
                                  name=f'{stage}.div_exposure_ratios'))
        
        vis.append(oh.make_tensor_value_info(exposure_ratios, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(brightness, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(contrast, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(saturation, TensorProto.FLOAT, ['m']))
        
        outputs = {
            'exposure_ratios': {'name': exposure_ratios},
            'brightness': {'name': brightness},
            'contrast': {'name': contrast},
            'saturation': {'name': saturation}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frames, type=TensorProto.FLOAT, shape=['n', 'm', 3, 'h', 'w'])
        result.appendInput(exposure_values, type=TensorProto.FLOAT, shape=['m'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class ExposureFusionAlgoV1(MicroblockBase):
    """
    ExposureFusionAlgoV1 (v1)
    ------------------------
    ALGORITHM: Extract exposure information with quality metrics.

    Position: Post-process stage after YUV/RGB conversion.
    
    Purpose: Analyze exposure levels with quality metrics.
    
    Needs:
        - frames [n,m,3,h,w] : Multiple exposure frames (m exposures)
        - exposure_values [m] : Exposure values for each frame

    Provides:
        - exposure_ratios [m] : Normalized exposure ratios
        - brightness [m] : Average brightness per frame
        - contrast [m] : Contrast per frame
        - saturation [m] : Saturation per frame
        - quality [m] : Quality score per frame
        - well_exposed [m] : Well-exposed mask per frame

    Behavior:
        - build_algo: Extracts exposure information with quality metrics
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles fusion)

    Complexity: ~40-50 ONNX nodes
    Use Case: High-quality exposure analysis
    """
    name = 'exposure_fusion_algo_v1'
    family = 'exposure_fusion_algo_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract exposure information with quality metrics.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        frames = f'{upstream}.frames'
        exposure_values = f'{upstream}.exposure_values'
        
        # Calculate brightness per frame
        brightness = f'{stage}.brightness'
        nodes.append(oh.make_node('ReduceMean', inputs=[frames], outputs=[brightness],
                                  name=f'{stage}.reduce_mean_brightness',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate contrast per frame
        contrast = f'{stage}.contrast'
        nodes.append(oh.make_node('ReduceStd', inputs=[frames], outputs=[contrast],
                                  name=f'{stage}.reduce_std_contrast',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate saturation per frame
        frame_max = f'{stage}.frame_max'
        frame_min = f'{stage}.frame_min'
        nodes.append(oh.make_node('ReduceMax', inputs=[frames], outputs=[frame_max],
                                  name=f'{stage}.reduce_max',
                                  axes=[2, 3, 4], keepdims=0))
        nodes.append(oh.make_node('ReduceMin', inputs=[frames], outputs=[frame_min],
                                  name=f'{stage}.reduce_min',
                                  axes=[2, 3, 4], keepdims=0))
        
        saturation = f'{stage}.saturation'
        nodes.append(oh.make_node('Sub', inputs=[frame_max, frame_min], outputs=[saturation],
                                  name=f'{stage}.sub_saturation'))
        
        # Calculate quality score (simplified: brightness + contrast + saturation)
        quality = f'{stage}.quality'
        brightness_norm = f'{stage}.brightness_norm'
        contrast_norm = f'{stage}.contrast_norm'
        saturation_norm = f'{stage}.saturation_norm'
        
        # Normalize to [0, 1]
        brightness_max = f'{stage}.brightness_max'
        contrast_max = f'{stage}.contrast_max'
        saturation_max = f'{stage}.saturation_max'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        nodes.append(oh.make_node('Div', inputs=[brightness, one], outputs=[brightness_norm],
                                  name=f'{stage}.div_brightness_norm'))
        nodes.append(oh.make_node('Div', inputs=[contrast, one], outputs=[contrast_norm],
                                  name=f'{stage}.div_contrast_norm'))
        nodes.append(oh.make_node('Div', inputs=[saturation, one], outputs=[saturation_norm],
                                  name=f'{stage}.div_saturation_norm'))
        
        # Combine into quality score
        quality_sum = f'{stage}.quality_sum'
        nodes.append(oh.make_node('Add', inputs=[brightness_norm, contrast_norm], outputs=[quality_sum],
                                  name=f'{stage}.add_quality_sum1'))
        nodes.append(oh.make_node('Add', inputs=[quality_sum, saturation_norm], outputs=[quality_sum],
                                  name=f'{stage}.add_quality_sum2'))
        
        nodes.append(oh.make_node('Div', inputs=[quality_sum, one], outputs=[quality],
                                  name=f'{stage}.div_quality'))
        
        # Calculate well-exposed mask (brightness close to 0.5)
        well_exposed = f'{stage}.well_exposed'
        target = f'{stage}.target'
        inits.append(oh.make_tensor(target, TensorProto.FLOAT, [], [0.5]))
        
        brightness_diff = f'{stage}.brightness_diff'
        nodes.append(oh.make_node('Sub', inputs=[brightness, target], outputs=[brightness_diff],
                                  name=f'{stage}.sub_brightness_diff'))
        
        brightness_diff_abs = f'{stage}.brightness_diff_abs'
        nodes.append(oh.make_node('Abs', inputs=[brightness_diff], outputs=[brightness_diff_abs],
                                  name=f'{stage}.abs_brightness_diff'))
        
        threshold = f'{stage}.threshold'
        inits.append(oh.make_tensor(threshold, TensorProto.FLOAT, [], [0.2]))
        
        well_exposed = f'{stage}.well_exposed'
        nodes.append(oh.make_node('Less', inputs=[brightness_diff_abs, threshold], outputs=[well_exposed],
                                  name=f'{stage}.less_well_exposed'))
        
        # Calculate exposure ratios
        exposure_sum = f'{stage}.exposure_sum'
        nodes.append(oh.make_node('ReduceSum', inputs=[exposure_values], outputs=[exposure_sum],
                                  name=f'{stage}.reduce_sum_exposure',
                                  axes=[0], keepdims=1))
        
        exposure_ratios = f'{stage}.exposure_ratios'
        nodes.append(oh.make_node('Div', inputs=[exposure_values, exposure_sum], outputs=[exposure_ratios],
                                  name=f'{stage}.div_exposure_ratios'))
        
        vis.append(oh.make_tensor_value_info(exposure_ratios, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(brightness, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(contrast, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(saturation, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(quality, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(well_exposed, TensorProto.BOOL, ['m']))
        
        outputs = {
            'exposure_ratios': {'name': exposure_ratios},
            'brightness': {'name': brightness},
            'contrast': {'name': contrast},
            'saturation': {'name': saturation},
            'quality': {'name': quality},
            'well_exposed': {'name': well_exposed}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frames, type=TensorProto.FLOAT, shape=['n', 'm', 3, 'h', 'w'])
        result.appendInput(exposure_values, type=TensorProto.FLOAT, shape=['m'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class ExposureFusionAlgoV2(MicroblockBase):
    """
    ExposureFusionAlgoV2 (v2)
    ------------------------
    ALGORITHM: Extract exposure information with pixel-level analysis.

    Position: Post-process stage after YUV/RGB conversion.
    
    Purpose: Analyze exposure levels with pixel-level quality metrics.
    
    Needs:
        - frames [n,m,3,h,w] : Multiple exposure frames (m exposures)
        - exposure_values [m] : Exposure values for each frame

    Provides:
        - exposure_ratios [m] : Normalized exposure ratios
        - brightness [m] : Average brightness per frame
        - contrast [m] : Contrast per frame
        - saturation [m] : Saturation per frame
        - quality [m] : Quality score per frame
        - well_exposed [m,h,w] : Well-exposed mask per pixel
        - weight_map [m,h,w] : Weight map for fusion

    Behavior:
        - build_algo: Extracts exposure information with pixel-level analysis
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles fusion)

    Complexity: ~60-80 ONNX nodes
    Use Case: Professional exposure analysis
    """
    name = 'exposure_fusion_algo_v2'
    family = 'exposure_fusion_algo_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract exposure information with pixel-level analysis.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        frames = f'{upstream}.frames'
        exposure_values = f'{upstream}.exposure_values'
        
        # Calculate brightness per frame
        brightness = f'{stage}.brightness'
        nodes.append(oh.make_node('ReduceMean', inputs=[frames], outputs=[brightness],
                                  name=f'{stage}.reduce_mean_brightness',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate contrast per frame
        contrast = f'{stage}.contrast'
        nodes.append(oh.make_node('ReduceStd', inputs=[frames], outputs=[contrast],
                                  name=f'{stage}.reduce_std_contrast',
                                  axes=[2, 3, 4], keepdims=0))
        
        # Calculate saturation per frame
        frame_max = f'{stage}.frame_max'
        frame_min = f'{stage}.frame_min'
        nodes.append(oh.make_node('ReduceMax', inputs=[frames], outputs=[frame_max],
                                  name=f'{stage}.reduce_max',
                                  axes=[2, 3, 4], keepdims=0))
        nodes.append(oh.make_node('ReduceMin', inputs=[frames], outputs=[frame_min],
                                  name=f'{stage}.reduce_min',
                                  axes=[2, 3, 4], keepdims=0))
        
        saturation = f'{stage}.saturation'
        nodes.append(oh.make_node('Sub', inputs=[frame_max, frame_min], outputs=[saturation],
                                  name=f'{stage}.sub_saturation'))
        
        # Calculate pixel-level well-exposed mask
        # Convert to grayscale for per-pixel analysis
        frames_gray = f'{stage}.frames_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        nodes.append(oh.make_node('ReduceMean', inputs=[frames], outputs=[frames_gray],
                                  name=f'{stage}.reduce_mean_gray',
                                  axes=[2], keepdims=1))
        
        # Calculate well-exposed per pixel
        target = f'{stage}.target'
        inits.append(oh.make_tensor(target, TensorProto.FLOAT, [], [0.5]))
        
        brightness_diff = f'{stage}.brightness_diff'
        nodes.append(oh.make_node('Sub', inputs=[frames_gray, target], outputs=[brightness_diff],
                                  name=f'{stage}.sub_brightness_diff'))
        
        brightness_diff_abs = f'{stage}.brightness_diff_abs'
        nodes.append(oh.make_node('Abs', inputs=[brightness_diff], outputs=[brightness_diff_abs],
                                  name=f'{stage}.abs_brightness_diff'))
        
        threshold = f'{stage}.threshold'
        inits.append(oh.make_tensor(threshold, TensorProto.FLOAT, [], [0.2]))
        
        well_exposed = f'{stage}.well_exposed'
        nodes.append(oh.make_node('Less', inputs=[brightness_diff_abs, threshold], outputs=[well_exposed],
                                  name=f'{stage}.less_well_exposed'))
        
        # Calculate weight map (simplified: based on well-exposed)
        weight_map = f'{stage}.weight_map'
        nodes.append(oh.make_node('Cast', inputs=[well_exposed], outputs=[weight_map],
                                  name=f'{stage}.cast_weight_map',
                                  to=TensorProto.FLOAT))
        
        # Calculate quality score
        quality = f'{stage}.quality'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        brightness_norm = f'{stage}.brightness_norm'
        contrast_norm = f'{stage}.contrast_norm'
        saturation_norm = f'{stage}.saturation_norm'
        
        nodes.append(oh.make_node('Div', inputs=[brightness, one], outputs=[brightness_norm],
                                  name=f'{stage}.div_brightness_norm'))
        nodes.append(oh.make_node('Div', inputs=[contrast, one], outputs=[contrast_norm],
                                  name=f'{stage}.div_contrast_norm'))
        nodes.append(oh.make_node('Div', inputs=[saturation, one], outputs=[saturation_norm],
                                  name=f'{stage}.div_saturation_norm'))
        
        quality_sum = f'{stage}.quality_sum'
        nodes.append(oh.make_node('Add', inputs=[brightness_norm, contrast_norm], outputs=[quality_sum],
                                  name=f'{stage}.add_quality_sum1'))
        nodes.append(oh.make_node('Add', inputs=[quality_sum, saturation_norm], outputs=[quality_sum],
                                  name=f'{stage}.add_quality_sum2'))
        
        nodes.append(oh.make_node('Div', inputs=[quality_sum, one], outputs=[quality],
                                  name=f'{stage}.div_quality'))
        
        # Calculate exposure ratios
        exposure_sum = f'{stage}.exposure_sum'
        nodes.append(oh.make_node('ReduceSum', inputs=[exposure_values], outputs=[exposure_sum],
                                  name=f'{stage}.reduce_sum_exposure',
                                  axes=[0], keepdims=1))
        
        exposure_ratios = f'{stage}.exposure_ratios'
        nodes.append(oh.make_node('Div', inputs=[exposure_values, exposure_sum], outputs=[exposure_ratios],
                                  name=f'{stage}.div_exposure_ratios'))
        
        vis.append(oh.make_tensor_value_info(exposure_ratios, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(brightness, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(contrast, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(saturation, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(quality, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(well_exposed, TensorProto.BOOL, ['n', 'm', 1, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(weight_map, TensorProto.FLOAT, ['n', 'm', 1, 'h', 'w']))
        
        outputs = {
            'exposure_ratios': {'name': exposure_ratios},
            'brightness': {'name': brightness},
            'contrast': {'name': contrast},
            'saturation': {'name': saturation},
            'quality': {'name': quality},
            'well_exposed': {'name': well_exposed},
            'weight_map': {'name': weight_map}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frames, type=TensorProto.FLOAT, shape=['n', 'm', 3, 'h', 'w'])
        result.appendInput(exposure_values, type=TensorProto.FLOAT, shape=['m'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)