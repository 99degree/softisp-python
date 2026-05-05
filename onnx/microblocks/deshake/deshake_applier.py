from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeApplierBase(MicroblockBase):
    """
    DeshakeApplierBase (v0)
    ----------------------
    APPLIER: Apply fused coordinate grid to stabilize frame.
    
    Position: After control loop, applies motion compensation.
    
    Purpose: Apply fused coordinate grid (GDC + Deshake) to stabilize frame.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - fused_grid [h,w,2] : Fused coordinate grid from coordinator

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies fused coordinate grid using GridSample

    Fused Grid Format:
        The fused_grid contains the final sampling coordinates after applying:
        1. GDC distortion correction
        2. Deshake homography transformation
        
        Grid format: grid[..., 0] = x coordinates, grid[..., 1] = y coordinates
        Coordinates are normalized to [-1, 1] for GridSample.

    Complexity: ~5-10 ONNX nodes
    Use Case: Real-time motion compensation with GDC fusion
    """
    name = 'deshake_applier_base'
    family = 'deshake_applier_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply fused coordinate grid using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        fused_grid = f'{upstream}.fused_grid'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Unsqueeze grid for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[fused_grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for motion compensation
        nodes.append(oh.make_node('GridSample', inputs=[current_frame, grid_expanded],
                                  outputs=[stabilized_frame],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        
        outputs = {'stabilized_frame': {'name': stabilized_frame}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(fused_grid, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV1(MicroblockBase):
    """
    DeshakeApplierV1 (v1)
    --------------------
    APPLIER: Apply fused coordinate grid with valid mask.
    
    Position: After control loop, applies motion compensation.
    
    Purpose: Apply fused coordinate grid with valid region detection.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - fused_grid [h,w,2] : Fused coordinate grid from coordinator

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - valid_mask [n,1,h,w] : Valid region mask

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies fused coordinate grid with valid mask

    Valid Mask:
        Pixels where the fused grid coordinates are within [-1, 1] are valid.
        Pixels outside this range are invalid (outside image bounds).

    Complexity: ~10-15 ONNX nodes
    Use Case: High-quality motion compensation with valid region detection
    """
    name = 'deshake_applier_v1'
    family = 'deshake_applier_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply fused coordinate grid with valid mask.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        fused_grid = f'{upstream}.fused_grid'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Extract x and y coordinates from fused grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[fused_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[fused_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Check if coordinates are within valid range [-1, 1]
        one = f'{stage}.one'
        minus_one = f'{stage}.minus_one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(minus_one, TensorProto.FLOAT, [], [-1.0]))
        
        x_valid = f'{stage}.x_valid'
        y_valid = f'{stage}.y_valid'
        nodes.append(oh.make_node('And', 
                                  inputs=[
                                      oh.make_node('Greater', inputs=[x, minus_one], outputs=['x_gt_minus1']),
                                      oh.make_node('Less', inputs=[x, one], outputs=['x_lt_1'])
                                  ],
                                  outputs=[x_valid],
                                  name=f'{stage}.and_x_valid'))
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[y, minus_one], outputs=['y_gt_minus1']),
                                      oh.make_node('Less', inputs=[y, one], outputs=['y_lt_1'])
                                  ],
                                  outputs=[y_valid],
                                  name=f'{stage}.and_y_valid'))
        
        # Combine x and y validity
        valid_mask = f'{stage}.valid_mask'
        nodes.append(oh.make_node('And', inputs=[x_valid, y_valid], outputs=[valid_mask],
                                  name=f'{stage}.and_valid'))
        
        # Unsqueeze for output [1,1,h,w]
        valid_mask_expanded = f'{stage}.valid_mask_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask', axes=[0]))
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask_expanded], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask_2', axes=[0]))
        
        # Unsqueeze grid for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[fused_grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for motion compensation
        nodes.append(oh.make_node('GridSample', inputs=[current_frame, grid_expanded],
                                  outputs=[stabilized_frame],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(valid_mask_expanded, TensorProto.BOOL, ['n', 1, 'h', 'w']))
        
        outputs = {
            'stabilized_frame': {'name': stabilized_frame},
            'valid_mask': {'name': valid_mask_expanded}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(fused_grid, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV2(MicroblockBase):
    """
    DeshakeApplierV2 (v2)
    --------------------
    APPLIER: Apply fused coordinate grid with gain correction.
    
    Position: After control loop, applies motion compensation.
    
    Purpose: Apply fused coordinate grid with ALSC gain correction.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - fused_grid [h,w,2] : Fused coordinate grid from coordinator
        - gain_map [h,w] : ALSC gain map from coordinator

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - valid_mask [n,1,h,w] : Valid region mask

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies fused coordinate grid with gain correction

    Gain Correction:
        The gain_map contains per-pixel gain values for ALSC correction.
        Each channel is multiplied by the gain map.

    Complexity: ~15-20 ONNX nodes
    Use Case: Professional motion compensation with gain correction
    """
    name = 'deshake_applier_v2'
    family = 'deshake_applier_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply fused coordinate grid with gain correction.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        fused_grid = f'{upstream}.fused_grid'
        gain_map = f'{upstream}.gain_map'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Unsqueeze grid for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[fused_grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample for motion compensation
        warped_frame = f'{stage}.warped_frame'
        nodes.append(oh.make_node('GridSample', inputs=[current_frame, grid_expanded],
                                  outputs=[warped_frame],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        # Apply gain correction
        # Unsqueeze gain map for broadcasting [1,1,h,w]
        gain_expanded = f'{stage}.gain_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[gain_map], outputs=[gain_expanded],
                                  name=f'{stage}.unsqueeze_gain', axes=[0]))
        nodes.append(oh.make_node('Unsqueeze', inputs=[gain_expanded], outputs=[gain_expanded],
                                  name=f'{stage}.unsqueeze_gain_2', axes=[0]))
        
        # Multiply each channel by gain map
        stabilized_frame = f'{stage}.stabilized_frame'
        nodes.append(oh.make_node('Mul', inputs=[warped_frame, gain_expanded],
                                  outputs=[stabilized_frame],
                                  name=f'{stage}.mul_gain'))
        
        # Create valid mask
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[fused_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[fused_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        one = f'{stage}.one'
        minus_one = f'{stage}.minus_one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(minus_one, TensorProto.FLOAT, [], [-1.0]))
        
        x_valid = f'{stage}.x_valid'
        y_valid = f'{stage}.y_valid'
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[x, minus_one], outputs=['x_gt_minus1']),
                                      oh.make_node('Less', inputs=[x, one], outputs=['x_lt_1'])
                                  ],
                                  outputs=[x_valid],
                                  name=f'{stage}.and_x_valid'))
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[y, minus_one], outputs=['y_gt_minus1']),
                                      oh.make_node('Less', inputs=[y, one], outputs=['y_lt_1'])
                                  ],
                                  outputs=[y_valid],
                                  name=f'{stage}.and_y_valid'))
        
        valid_mask = f'{stage}.valid_mask'
        nodes.append(oh.make_node('And', inputs=[x_valid, y_valid], outputs=[valid_mask],
                                  name=f'{stage}.and_valid'))
        
        valid_mask_expanded = f'{stage}.valid_mask_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask', axes=[0]))
        nodes.append(oh.make_node('Unsqueeze', inputs=[valid_mask_expanded], outputs=[valid_mask_expanded],
                                  name=f'{stage}.unsqueeze_valid_mask_2', axes=[0]))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(valid_mask_expanded, TensorProto.BOOL, ['n', 1, 'h', 'w']))
        
        outputs = {
            'stabilized_frame': {'name': stabilized_frame},
            'valid_mask': {'name': valid_mask_expanded}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(fused_grid, type=TensorProto.FLOAT, shape=['h', 'w', 2])
        result.appendInput(gain_map, type=TensorProto.FLOAT, shape=['h', 'w'])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)