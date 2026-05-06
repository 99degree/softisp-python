from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class ExposureFusionLoopBase(MicroblockBase):
    """
    ExposureFusionLoopBase (v0)
    --------------------------
    CONTROL LOOP: Iterative exposure fusion with weighting.

    Position: Between algo and applier, fuses multiple exposures.
    
    Purpose: Fuse multiple exposures with adaptive weighting.
    
    Needs:
        - frames [n,m,3,h,w] : Multiple exposure frames (m exposures)
        - exposure_ratios [m] : Normalized exposure ratios
        - brightness [m] : Average brightness per frame
        - contrast [m] : Contrast per frame
        - saturation [m] : Saturation per frame
        - fusion_strength [1] : Fusion strength (0-1)
        - iterations [1] : Number of fusion iterations

    Provides:
        - fused_frame [n,3,h,w] : Fused frame
        - weights [m] : Final weights for each exposure
        - dynamic_range [1] : Estimated dynamic range

    Behavior:
        - build_algo: Not used (algo extracts exposure info)
        - build_coordinator: Creates Loop for exposure fusion
        - build_applier: Not used (applier handles tone mapping)

    Loop Body:
        1. Calculate adaptive weights based on quality metrics
        2. Normalize weights
        3. Blend exposures using weights
        4. Update weights for next iteration

    Complexity: ~50-70 ONNX nodes
    Use Case: Real-time exposure fusion
    """
    name = 'exposure_fusion_loop_base'
    family = 'exposure_fusion_loop_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts exposure info.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop operator for iterative exposure fusion.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frames = f'{upstream}.frames'
        exposure_ratios = f'{upstream}.exposure_ratios'
        brightness = f'{upstream}.brightness'
        contrast = f'{upstream}.contrast'
        saturation = f'{upstream}.saturation'
        fusion_strength = f'{upstream}.fusion_strength'
        iterations = f'{upstream}.iterations'
        
        # Initial weights: based on exposure ratios
        weights_init = f'{stage}.weights_init'
        nodes.append(oh.make_node('Identity', inputs=[exposure_ratios], outputs=[weights_init],
                                  name=f'{stage}.identity_weights_init'))
        
        # Create loop body graph
        loop_body_name = f'{stage}_loop_body'
        loop_body_graph = oh.make_graph(
            name=loop_body_name,
            inputs=[
                oh.make_tensor_value_info('iter', TensorProto.INT64, []),
                oh.make_tensor_value_info('cond', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state_weights', TensorProto.FLOAT, ['m']),
                oh.make_tensor_value_info('state_fused', TensorProto.FLOAT, ['n', 3, 'h', 'w']),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('weights_out', TensorProto.FLOAT, ['m']),
                oh.make_tensor_value_info('fused_out', TensorProto.FLOAT, ['n', 3, 'h', 'w']),
                oh.make_tensor_value_info('dynamic_range_out', TensorProto.FLOAT, [1]),
            ],
            nodes=[]
        )
        
        # Add loop body nodes
        loop_nodes = []
        
        # Calculate adaptive weights based on quality metrics
        # Weight = α * brightness + β * contrast + γ * saturation
        one = f'{loop_body_name}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        # Normalize quality metrics
        brightness_norm = f'{loop_body_name}.brightness_norm'
        contrast_norm = f'{loop_body_name}.contrast_norm'
        saturation_norm = f'{loop_body_name}.saturation_norm'
        
        loop_nodes.append(oh.make_node('Div', inputs=['brightness', one], outputs=[brightness_norm],
                                       name=f'{loop_body_name}.div_brightness_norm'))
        loop_nodes.append(oh.make_node('Div', inputs=['contrast', one], outputs=[contrast_norm],
                                       name=f'{loop_body_name}.div_contrast_norm'))
        loop_nodes.append(oh.make_node('Div', inputs=['saturation', one], outputs=[saturation_norm],
                                       name=f'{loop_body_name}.div_saturation_norm'))
        
        # Combine into quality score
        quality = f'{loop_body_name}.quality'
        quality_sum = f'{loop_body_name}.quality_sum'
        loop_nodes.append(oh.make_node('Add', inputs=[brightness_norm, contrast_norm], outputs=[quality_sum],
                                       name=f'{loop_body_name}.add_quality_sum1'))
        loop_nodes.append(oh.make_node('Add', inputs=[quality_sum, saturation_norm], outputs=[quality_sum],
                                       name=f'{loop_body_name}.add_quality_sum2'))
        loop_nodes.append(oh.make_node('Div', inputs=[quality_sum, one], outputs=[quality],
                                       name=f'{loop_body_name}.div_quality'))
        
        # Update weights: w_new = w_old * (1 - α) + quality * α
        weights_new = f'{loop_body_name}.weights_new'
        weights_quality = f'{loop_body_name}.weights_quality'
        loop_nodes.append(oh.make_node('Mul', inputs=['state_weights', 'fusion_strength'],
                                       outputs=[weights_quality],
                                       name=f'{loop_body_name}.mul_weights_quality'))
        
        quality_weights = f'{loop_body_name}.quality_weights'
        loop_nodes.append(oh.make_node('Mul', inputs=[quality, 'fusion_strength'],
                                       outputs=[quality_weights],
                                       name=f'{loop_body_name}.mul_quality_weights'))
        
        loop_nodes.append(oh.make_node('Add', inputs=[weights_quality, quality_weights], outputs=[weights_new],
                                       name=f'{loop_body_name}.add_weights_new'))
        
        # Normalize weights
        weights_sum = f'{loop_body_name}.weights_sum'
        loop_nodes.append(oh.make_node('ReduceSum', inputs=[weights_new], outputs=[weights_sum],
                                       name=f'{loop_body_name}.reduce_sum_weights',
                                       axes=[0], keepdims=1))
        
        weights_normalized = f'{loop_body_name}.weights_normalized'
        loop_nodes.append(oh.make_node('Div', inputs=[weights_new, weights_sum], outputs=[weights_normalized],
                                       name=f'{loop_body_name}.div_weights_normalized'))
        
        # Blend exposures using weights
        # fused = Σ (weights[i] * frames[i])
        fused = f'{loop_body_name}.fused'
        
        # Reshape weights for broadcasting
        weights_reshaped = f'{loop_body_name}.weights_reshaped'
        loop_nodes.append(oh.make_node('Unsqueeze', inputs=[weights_normalized], outputs=[weights_reshaped],
                                       name=f'{loop_body_name}.unsqueeze_weights',
                                       axes=[0, 2, 3, 4]))
        
        # Multiply each frame by its weight
        frames_weighted = f'{loop_body_name}.frames_weighted'
        loop_nodes.append(oh.make_node('Mul', inputs=['frames', weights_reshaped], outputs=[frames_weighted],
                                       name=f'{loop_body_name}.mul_frames_weighted'))
        
        # Sum across exposure dimension
        loop_nodes.append(oh.make_node('ReduceSum', inputs=[frames_weighted], outputs=[fused],
                                       name=f'{loop_body_name}.reduce_sum_fused',
                                       axes=[1], keepdims=0))
        
        # Calculate dynamic range (simplified: max - min brightness)
        dynamic_range = f'{loop_body_name}.dynamic_range'
        brightness_max = f'{loop_body_name}.brightness_max'
        brightness_min = f'{loop_body_name}.brightness_min'
        loop_nodes.append(oh.make_node('ReduceMax', inputs=['brightness'], outputs=[brightness_max],
                                       name=f'{loop_body_name}.reduce_max_brightness',
                                       axes=[0], keepdims=0))
        loop_nodes.append(oh.make_node('ReduceMin', inputs=['brightness'], outputs=[brightness_min],
                                       name=f'{loop_body_name}.reduce_min_brightness',
                                       axes=[0], keepdims=0))
        loop_nodes.append(oh.make_node('Sub', inputs=[brightness_max, brightness_min], outputs=[dynamic_range],
                                       name=f'{loop_body_name}.sub_dynamic_range'))
        
        # Update loop body graph
        loop_body_graph.nodes.extend(loop_nodes)
        
        # Create Loop node
        loop_output = f'{stage}.loop_output'
        fused_init = f'{stage}.fused_init'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, ['n', 3, 'h', 'w'],
                                  [0.0, 0.0, 0.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[fused_init],
                                  name=f'{stage}.identity_fused_init'))
        
        nodes.append(oh.make_node('Loop', 
                                  inputs=[iterations, 'true', weights_init, fused_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        weights = f'{stage}.weights'
        fused_frame = f'{stage}.fused_frame'
        dynamic_range = f'{stage}.dynamic_range'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[weights, fused_frame, dynamic_range],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        vis.append(oh.make_tensor_value_info(fused_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(weights, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(dynamic_range, TensorProto.FLOAT, [1]))
        
        outputs = {
            'fused_frame': {'name': fused_frame},
            'weights': {'name': weights},
            'dynamic_range': {'name': dynamic_range}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frames, type=TensorProto.FLOAT, shape=['n', 'm', 3, 'h', 'w'])
        result.appendInput(exposure_ratios, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(brightness, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(contrast, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(saturation, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(fusion_strength, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles tone mapping.
        """
        return super().build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)


class ExposureFusionLoopV1(MicroblockBase):
    """
    ExposureFusionLoopV1 (v1)
    ------------------------
    CONTROL LOOP: Exposure fusion with ghosting reduction.

    Position: Between algo and applier, fuses multiple exposures.
    
    Purpose: Fuse multiple exposures with ghosting reduction.
    
    Needs:
        - frames [n,m,3,h,w] : Multiple exposure frames (m exposures)
        - exposure_ratios [m] : Normalized exposure ratios
        - brightness [m] : Average brightness per frame
        - contrast [m] : Contrast per frame
        - saturation [m] : Saturation per frame
        - quality [m] : Quality score per frame
        - well_exposed [m] : Well-exposed mask per frame
        - fusion_strength [1] : Fusion strength (0-1)
        - ghost_threshold [1] : Ghosting detection threshold
        - iterations [1] : Number of fusion iterations

    Provides:
        - fused_frame [n,3,h,w] : Fused frame
        - weights [m] : Final weights for each exposure
        - dynamic_range [1] : Estimated dynamic range
        - ghost_mask [n,1,h,w] : Ghosting mask

    Behavior:
        - build_algo: Not used (algo extracts exposure info)
        - build_coordinator: Creates Loop with ghosting reduction
        - build_applier: Not used (applier handles tone mapping)

    Loop Body:
        1. Calculate adaptive weights based on quality metrics
        2. Detect and suppress ghosting artifacts
        3. Normalize weights
        4. Blend exposures using weights
        5. Update weights for next iteration

    Complexity: ~70-90 ONNX nodes
    Use Case: High-quality exposure fusion with ghosting reduction
    """
    name = 'exposure_fusion_loop_v1'
    family = 'exposure_fusion_loop_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts exposure info.
        """
        return super().build_algo(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop with ghosting reduction for exposure fusion.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frames = f'{upstream}.frames'
        exposure_ratios = f'{upstream}.exposure_ratios'
        brightness = f'{upstream}.brightness'
        contrast = f'{upstream}.contrast'
        saturation = f'{upstream}.saturation'
        quality = f'{upstream}.quality'
        well_exposed = f'{upstream}.well_exposed'
        fusion_strength = f'{upstream}.fusion_strength'
        ghost_threshold = f'{upstream}.ghost_threshold'
        iterations = f'{upstream}.iterations'
        
        # Initial weights: based on quality
        weights_init = f'{stage}.weights_init'
        nodes.append(oh.make_node('Identity', inputs=[quality], outputs=[weights_init],
                                  name=f'{stage}.identity_weights_init'))
        
        # Create loop body graph
        loop_body_name = f'{stage}_loop_body'
        loop_body_graph = oh.make_graph(
            name=loop_body_name,
            inputs=[
                oh.make_tensor_value_info('iter', TensorProto.INT64, []),
                oh.make_tensor_value_info('cond', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state_weights', TensorProto.FLOAT, ['m']),
                oh.make_tensor_value_info('state_fused', TensorProto.FLOAT, ['n', 3, 'h', 'w']),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('weights_out', TensorProto.FLOAT, ['m']),
                oh.make_tensor_value_info('fused_out', TensorProto.FLOAT, ['n', 3, 'h', 'w']),
                oh.make_tensor_value_info('dynamic_range_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('ghost_mask_out', TensorProto.BOOL, ['n', 1, 'h', 'w']),
            ],
            nodes=[]
        )
        
        # Add loop body nodes
        loop_nodes = []
        
        # Calculate adaptive weights based on quality metrics
        one = f'{loop_body_name}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        # Normalize quality metrics
        brightness_norm = f'{loop_body_name}.brightness_norm'
        contrast_norm = f'{loop_body_name}.contrast_norm'
        saturation_norm = f'{loop_body_name}.saturation_norm'
        
        loop_nodes.append(oh.make_node('Div', inputs=['brightness', one], outputs=[brightness_norm],
                                       name=f'{loop_body_name}.div_brightness_norm'))
        loop_nodes.append(oh.make_node('Div', inputs=['contrast', one], outputs=[contrast_norm],
                                       name=f'{loop_body_name}.div_contrast_norm'))
        loop_nodes.append(oh.make_node('Div', inputs=['saturation', one], outputs=[saturation_norm],
                                       name=f'{loop_body_name}.div_saturation_norm'))
        
        # Combine into quality score
        quality_calc = f'{loop_body_name}.quality_calc'
        quality_sum = f'{loop_body_name}.quality_sum'
        loop_nodes.append(oh.make_node('Add', inputs=[brightness_norm, contrast_norm], outputs=[quality_sum],
                                       name=f'{loop_body_name}.add_quality_sum1'))
        loop_nodes.append(oh.make_node('Add', inputs=[quality_sum, saturation_norm], outputs=[quality_sum],
                                       name=f'{loop_body_name}.add_quality_sum2'))
        loop_nodes.append(oh.make_node('Div', inputs=[quality_sum, one], outputs=[quality_calc],
                                       name=f'{loop_body_name}.div_quality'))
        
        # Update weights with well-exposed mask
        weights_new = f'{loop_body_name}.weights_new'
        well_exposed_float = f'{loop_body_name}.well_exposed_float'
        loop_nodes.append(oh.make_node('Cast', inputs=['well_exposed'], outputs=[well_exposed_float],
                                       name=f'{loop_body_name}.cast_well_exposed',
                                       to=TensorProto.FLOAT))
        
        quality_well = f'{loop_body_name}.quality_well'
        loop_nodes.append(oh.make_node('Mul', inputs=[quality_calc, well_exposed_float], outputs=[quality_well],
                                       name=f'{loop_body_name}.mul_quality_well'))
        
        # Blend with previous weights
        weights_old = f'{loop_body_name}.weights_old'
        loop_nodes.append(oh.make_node('Mul', inputs=['state_weights', 'fusion_strength'],
                                       outputs=[weights_old],
                                       name=f'{loop_body_name}.mul_weights_old'))
        
        loop_nodes.append(oh.make_node('Add', inputs=[weights_old, quality_well], outputs=[weights_new],
                                       name=f'{loop_body_name}.add_weights_new'))
        
        # Normalize weights
        weights_sum = f'{loop_body_name}.weights_sum'
        loop_nodes.append(oh.make_node('ReduceSum', inputs=[weights_new], outputs=[weights_sum],
                                       name=f'{loop_body_name}.reduce_sum_weights',
                                       axes=[0], keepdims=1))
        
        weights_normalized = f'{loop_body_name}.weights_normalized'
        loop_nodes.append(oh.make_node('Div', inputs=[weights_new, weights_sum], outputs=[weights_normalized],
                                       name=f'{loop_body_name}.div_weights_normalized'))
        
        # Detect ghosting (simplified: difference between frames)
        ghost_mask = f'{loop_body_name}.ghost_mask'
        frame_diff = f'{loop_body_name}.frame_diff'
        
        # Calculate difference between consecutive frames
        frame_0 = f'{loop_body_name}.frame_0'
        frame_1 = f'{loop_body_name}.frame_1'
        loop_nodes.append(oh.make_node('Slice', inputs=['frames'], outputs=[frame_0],
                                       name=f'{loop_body_name}.slice_frame_0',
                                       starts=[0, 0], ends=[1, -1], axes=[0, 1]))
        loop_nodes.append(oh.make_node('Slice', inputs=['frames'], outputs=[frame_1],
                                       name=f'{loop_body_name}.slice_frame_1',
                                       starts=[0, 1], ends=[1, -1], axes=[0, 1]))
        
        loop_nodes.append(oh.make_node('Sub', inputs=[frame_0, frame_1], outputs=[frame_diff],
                                       name=f'{loop_body_name}.sub_frame_diff'))
        
        # Threshold for ghosting
        frame_diff_abs = f'{loop_body_name}.frame_diff_abs'
        loop_nodes.append(oh.make_node('Abs', inputs=[frame_diff], outputs=[frame_diff_abs],
                                       name=f'{loop_body_name}.abs_frame_diff'))
        
        loop_nodes.append(oh.make_node('Greater', inputs=[frame_diff_abs, 'ghost_threshold'], outputs=[ghost_mask],
                                       name=f'{loop_body_name}.greater_ghost_mask'))
        
        # Blend exposures using weights
        fused = f'{loop_body_name}.fused'
        
        # Reshape weights for broadcasting
        weights_reshaped = f'{loop_body_name}.weights_reshaped'
        loop_nodes.append(oh.make_node('Unsqueeze', inputs=[weights_normalized], outputs=[weights_reshaped],
                                       name=f'{loop_body_name}.unsqueeze_weights',
                                       axes=[0, 2, 3, 4]))
        
        # Multiply each frame by its weight
        frames_weighted = f'{loop_body_name}.frames_weighted'
        loop_nodes.append(oh.make_node('Mul', inputs=['frames', weights_reshaped], outputs=[frames_weighted],
                                       name=f'{loop_body_name}.mul_frames_weighted'))
        
        # Sum across exposure dimension
        loop_nodes.append(oh.make_node('ReduceSum', inputs=[frames_weighted], outputs=[fused],
                                       name=f'{loop_body_name}.reduce_sum_fused',
                                       axes=[1], keepdims=0))
        
        # Calculate dynamic range
        dynamic_range = f'{loop_body_name}.dynamic_range'
        brightness_max = f'{loop_body_name}.brightness_max'
        brightness_min = f'{loop_body_name}.brightness_min'
        loop_nodes.append(oh.make_node('ReduceMax', inputs=['brightness'], outputs=[brightness_max],
                                       name=f'{loop_body_name}.reduce_max_brightness',
                                       axes=[0], keepdims=0))
        loop_nodes.append(oh.make_node('ReduceMin', inputs=['brightness'], outputs=[brightness_min],
                                       name=f'{loop_body_name}.reduce_min_brightness',
                                       axes=[0], keepdims=0))
        loop_nodes.append(oh.make_node('Sub', inputs=[brightness_max, brightness_min], outputs=[dynamic_range],
                                       name=f'{loop_body_name}.sub_dynamic_range'))
        
        # Update loop body graph
        loop_body_graph.nodes.extend(loop_nodes)
        
        # Create Loop node
        loop_output = f'{stage}.loop_output'
        fused_init = f'{stage}.fused_init'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, ['n', 3, 'h', 'w'],
                                  [0.0, 0.0, 0.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[fused_init],
                                  name=f'{stage}.identity_fused_init'))
        
        nodes.append(oh.make_node('Loop', 
                                  inputs=[iterations, 'true', weights_init, fused_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        weights = f'{stage}.weights'
        fused_frame = f'{stage}.fused_frame'
        dynamic_range = f'{stage}.dynamic_range'
        ghost_mask = f'{stage}.ghost_mask'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[weights, fused_frame, dynamic_range, ghost_mask],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        vis.append(oh.make_tensor_value_info(fused_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(weights, TensorProto.FLOAT, ['m']))
        vis.append(oh.make_tensor_value_info(dynamic_range, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(ghost_mask, TensorProto.BOOL, ['n', 1, 'h', 'w']))
        
        outputs = {
            'fused_frame': {'name': fused_frame},
            'weights': {'name': weights},
            'dynamic_range': {'name': dynamic_range},
            'ghost_mask': {'name': ghost_mask}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frames, type=TensorProto.FLOAT, shape=['n', 'm', 3, 'h', 'w'])
        result.appendInput(exposure_ratios, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(brightness, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(contrast, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(saturation, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(quality, type=TensorProto.FLOAT, shape=['m'])
        result.appendInput(well_exposed, type=TensorProto.BOOL, shape=['m'])
        result.appendInput(fusion_strength, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(ghost_threshold, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles tone mapping.
        """
        return super().build_applier(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)