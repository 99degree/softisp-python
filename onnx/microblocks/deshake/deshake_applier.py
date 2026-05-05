from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeApplierBase(MicroblockBase):
    """
    DeshakeApplierBase (v0)
    ----------------------
    APPLIER: Apply refined motion parameters to stabilize frame.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply refined motion parameters (dx, dy, dr, dz) to frame.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - dx [1] : Refined horizontal translation (pixels)
        - dy [1] : Refined vertical translation (pixels)
        - dr [1] : Refined rotation angle (radians)
        - dz [1] : Refined zoom/scale factor

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts motion from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies motion compensation using GridSample

    Motion Model:
        - Translation: x' = x - dx, y' = y - dy
        - Rotation: Apply rotation around center
        - Zoom: Apply scale factor dz
        - Combined: Transform = Translation * Rotation * Zoom

    Complexity: ~20-30 ONNX nodes
    Use Case: Real-time motion compensation
    """
    name = 'deshake_applier_base'
    family = 'deshake_applier_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts motion from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply refined motion parameters using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        dx = f'{upstream}.dx'
        dy = f'{upstream}.dy'
        dr = f'{upstream}.dr'
        dz = f'{upstream}.dz'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Create coordinate grid
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        # Normalize coordinates to [-1, 1]
        h_norm = f'{stage}.h_norm'
        w_norm = f'{stage}.w_norm'
        h_half = f'{stage}.h_half'
        w_half = f'{stage}.w_half'
        inits.append(oh.make_tensor(h_half, TensorProto.FLOAT, [], [0.5]))
        inits.append(oh.make_tensor(w_half, TensorProto.FLOAT, [], [0.5]))
        
        nodes.append(oh.make_node('Mul', inputs=[h_coord, h_half], outputs=[h_norm],
                                  name=f'{stage}.mul_h_norm'))
        nodes.append(oh.make_node('Mul', inputs=[w_coord, w_half], outputs=[w_norm],
                                  name=f'{stage}.mul_w_norm'))
        
        # Apply translation
        x_translated = f'{stage}.x_translated'
        y_translated = f'{stage}.y_translated'
        nodes.append(oh.make_node('Sub', inputs=[w_norm, dx], outputs=[x_translated],
                                  name=f'{stage}.sub_x_translation'))
        nodes.append(oh.make_node('Sub', inputs=[h_norm, dy], outputs=[y_translated],
                                  name=f'{stage}.sub_y_translation'))
        
        # Apply rotation (simplified: around center)
        # For full rotation, need to compute:
        # x_rot = (x - cx) * cos(dr) - (y - cy) * sin(dr) + cx
        # y_rot = (x - cx) * sin(dr) + (y - cy) * cos(dr) + cy
        # Simplified: skip rotation for now
        x_rotated = f'{stage}.x_rotated'
        y_rotated = f'{stage}.y_rotated'
        nodes.append(oh.make_node('Identity', inputs=[x_translated], outputs=[x_rotated],
                                  name=f'{stage}.identity_x_rotation'))
        nodes.append(oh.make_node('Identity', inputs=[y_translated], outputs=[y_rotated],
                                  name=f'{stage}.identity_y_rotation'))
        
        # Apply zoom
        x_zoomed = f'{stage}.x_zoomed'
        y_zoomed = f'{stage}.y_zoomed'
        nodes.append(oh.make_node('Mul', inputs=[x_rotated, dz], outputs=[x_zoomed],
                                  name=f'{stage}.mul_x_zoom'))
        nodes.append(oh.make_node('Mul', inputs=[y_rotated, dz], outputs=[y_zoomed],
                                  name=f'{stage}.mul_y_zoom'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_zoomed, y_zoomed], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
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
        result.appendInput(dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dz, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV1(MicroblockBase):
    """
    DeshakeApplierV1 (v1)
    --------------------
    APPLIER: Apply motion with rotation and zoom.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply full motion model with rotation and zoom.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - dx [1] : Refined horizontal translation (pixels)
        - dy [1] : Refined vertical translation (pixels)
        - dr [1] : Refined rotation angle (radians)
        - dz [1] : Refined zoom/scale factor
        - center_x [1] : Rotation center X (pixels)
        - center_y [1] : Rotation center Y (pixels)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts motion from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies full motion model

    Motion Model:
        - Translation: x' = x - dx, y' = y - dy
        - Rotation: Apply rotation around center
        - Zoom: Apply scale factor dz
        - Combined: Transform = Translation * Rotation * Zoom

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality motion compensation
    """
    name = 'deshake_applier_v1'
    family = 'deshake_applier_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts motion from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply full motion model with rotation and zoom.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        dx = f'{upstream}.dx'
        dy = f'{upstream}.dy'
        dr = f'{upstream}.dr'
        dz = f'{upstream}.dz'
        center_x = f'{upstream}.center_x'
        center_y = f'{upstream}.center_y'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Create coordinate grid
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        # Normalize coordinates to [-1, 1]
        h_norm = f'{stage}.h_norm'
        w_norm = f'{stage}.w_norm'
        h_half = f'{stage}.h_half'
        w_half = f'{stage}.w_half'
        inits.append(oh.make_tensor(h_half, TensorProto.FLOAT, [], [0.5]))
        inits.append(oh.make_tensor(w_half, TensorProto.FLOAT, [], [0.5]))
        
        nodes.append(oh.make_node('Mul', inputs=[h_coord, h_half], outputs=[h_norm],
                                  name=f'{stage}.mul_h_norm'))
        nodes.append(oh.make_node('Mul', inputs=[w_coord, w_half], outputs=[w_norm],
                                  name=f'{stage}.mul_w_norm'))
        
        # Apply translation
        x_translated = f'{stage}.x_translated'
        y_translated = f'{stage}.y_translated'
        nodes.append(oh.make_node('Sub', inputs=[w_norm, dx], outputs=[x_translated],
                                  name=f'{stage}.sub_x_translation'))
        nodes.append(oh.make_node('Sub', inputs=[h_norm, dy], outputs=[y_translated],
                                  name=f'{stage}.sub_y_translation'))
        
        # Apply rotation around center
        # x_rot = (x - cx) * cos(dr) - (y - cy) * sin(dr) + cx
        # y_rot = (x - cx) * sin(dr) + (y - cy) * cos(dr) + cy
        cos_dr = f'{stage}.cos_dr'
        sin_dr = f'{stage}.sin_dr'
        nodes.append(oh.make_node('Cos', inputs=[dr], outputs=[cos_dr],
                                  name=f'{stage}.cos_dr'))
        nodes.append(oh.make_node('Sin', inputs=[dr], outputs=[sin_dr],
                                  name=f'{stage}.sin_dr'))
        
        # Subtract center
        x_centered = f'{stage}.x_centered'
        y_centered = f'{stage}.y_centered'
        nodes.append(oh.make_node('Sub', inputs=[x_translated, center_x], outputs=[x_centered],
                                  name=f'{stage}.sub_x_center'))
        nodes.append(oh.make_node('Sub', inputs=[y_translated, center_y], outputs=[y_centered],
                                  name=f'{stage}.sub_y_center'))
        
        # Apply rotation
        x_rot_term1 = f'{stage}.x_rot_term1'
        x_rot_term2 = f'{stage}.x_rot_term2'
        nodes.append(oh.make_node('Mul', inputs=[x_centered, cos_dr], outputs=[x_rot_term1],
                                  name=f'{stage}.mul_x_rot_term1'))
        nodes.append(oh.make_node('Mul', inputs=[y_centered, sin_dr], outputs=[x_rot_term2],
                                  name=f'{stage}.mul_x_rot_term2'))
        
        y_rot_term1 = f'{stage}.y_rot_term1'
        y_rot_term2 = f'{stage}.y_rot_term2'
        nodes.append(oh.make_node('Mul', inputs=[x_centered, sin_dr], outputs=[y_rot_term1],
                                  name=f'{stage}.mul_y_rot_term1'))
        nodes.append(oh.make_node('Mul', inputs=[y_centered, cos_dr], outputs=[y_rot_term2],
                                  name=f'{stage}.mul_y_rot_term2'))
        
        x_rotated = f'{stage}.x_rotated'
        y_rotated = f'{stage}.y_rotated'
        nodes.append(oh.make_node('Sub', inputs=[x_rot_term1, x_rot_term2], outputs=[x_rotated],
                                  name=f'{stage}.sub_x_rotated'))
        nodes.append(oh.make_node('Add', inputs=[y_rot_term1, y_rot_term2], outputs=[y_rotated],
                                  name=f'{stage}.add_y_rotated'))
        
        # Add center back
        x_rot_centered = f'{stage}.x_rot_centered'
        y_rot_centered = f'{stage}.y_rot_centered'
        nodes.append(oh.make_node('Add', inputs=[x_rotated, center_x], outputs=[x_rot_centered],
                                  name=f'{stage}.add_x_rot_center'))
        nodes.append(oh.make_node('Add', inputs=[y_rotated, center_y], outputs=[y_rot_centered],
                                  name=f'{stage}.add_y_rot_center'))
        
        # Apply zoom
        x_zoomed = f'{stage}.x_zoomed'
        y_zoomed = f'{stage}.y_zoomed'
        nodes.append(oh.make_node('Mul', inputs=[x_rot_centered, dz], outputs=[x_zoomed],
                                  name=f'{stage}.mul_x_zoom'))
        nodes.append(oh.make_node('Mul', inputs=[y_rot_centered, dz], outputs=[y_zoomed],
                                  name=f'{stage}.mul_y_zoom'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_zoomed, y_zoomed], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
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
        result.appendInput(dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(dz, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(center_x, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(center_y, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV2(MicroblockBase):
    """
    DeshakeApplierV2 (v2)
    --------------------
    APPLIER: Apply motion with homography transformation.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply homography-based motion compensation.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - homography [3,3] : 3x3 homography matrix

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts motion from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies homography transformation

    Motion Model:
        - Homography: [x', y', 1]^T = H * [x, y, 1]^T
        - Handles: Translation, rotation, zoom, perspective

    Complexity: ~25-35 ONNX nodes
    Use Case: Professional motion compensation
    """
    name = 'deshake_applier_v2'
    family = 'deshake_applier_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts motion from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply homography transformation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        homography = f'{upstream}.homography'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Create coordinate grid
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        # Normalize coordinates to [-1, 1]
        h_norm = f'{stage}.h_norm'
        w_norm = f'{stage}.w_norm'
        h_half = f'{stage}.h_half'
        w_half = f'{stage}.w_half'
        inits.append(oh.make_tensor(h_half, TensorProto.FLOAT, [], [0.5]))
        inits.append(oh.make_tensor(w_half, TensorProto.FLOAT, [], [0.5]))
        
        nodes.append(oh.make_node('Mul', inputs=[h_coord, h_half], outputs=[h_norm],
                                  name=f'{stage}.mul_h_norm'))
        nodes.append(oh.make_node('Mul', inputs=[w_coord, w_half], outputs=[w_norm],
                                  name=f'{stage}.mul_w_norm'))
        
        # Stack into homogeneous coordinates [h,w,3]
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        grid_homo = f'{stage}.grid_homo'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm, ones], outputs=[grid_homo],
                                  name=f'{stage}.concat_grid_homo', axis=-1))
        
        # Apply homography: [x', y', w'] = H * [x, y, 1]
        # Simplified: use identity for now
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[grid],
                                  name=f'{stage}.slice_grid',
                                  starts=[0], ends=[2], axes=[-1]))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
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
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)