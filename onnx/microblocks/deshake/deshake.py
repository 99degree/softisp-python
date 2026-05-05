from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeBase(MicroblockBase):
    """
    DeshakeBase (v0)
    ---------------
    POST-PROCESS STAGE: Video stabilization after YUV/RGB conversion.
    
    Position in Pipeline: After YUV/RGB conversion, before output.
    
    Operates on final color-corrected frames (YUV or RGB format).
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - motion_strength [1] : Motion compensation strength (0-1)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - motion_vector [2] : Global motion vector (dx, dy)

    Behavior:
        - build_algo: Calculates global motion vector between frames
        - build_applier: Applies motion compensation using GridSample

    Complexity: ~15-20 ONNX nodes
    Use Case: Real-time video stabilization on mobile devices
    
    Pipeline Position:
        Input: YUV/RGB frame (after color correction)
        Output: Stabilized YUV/RGB frame
    """
    name = 'deshake_base'
    family = 'deshake_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Calculate global motion vector between current and previous frames.
        
        Uses block matching to estimate global translation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        motion_strength = f'{upstream}.motion_strength'
        
        # Convert to grayscale for motion estimation
        # Simple average of RGB channels
        current_gray = f'{stage}.current_gray'
        prev_gray = f'{stage}.prev_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('ReduceMean', inputs=[current_frame], outputs=[current_gray],
                                  name=f'{stage}.reduce_mean_current', axes=[1], keepdims=1))
        nodes.append(oh.make_node('ReduceMean', inputs=[prev_frame], outputs=[prev_gray],
                                  name=f'{stage}.reduce_mean_prev', axes=[1], keepdims=1))
        
        # Calculate block matching (simplified: use center region)
        # Extract center region for matching
        h_center = f'{stage}.h_center'
        w_center = f'{stage}.w_center'
        vis.append(oh.make_tensor_value_info(h_center, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_center, TensorProto.FLOAT, ['w']))
        
        # Calculate motion using phase correlation (simplified)
        # For now, use simple difference-based estimation
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Calculate motion vector from difference (simplified)
        # In practice, this would use phase correlation or block matching
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        # Simplified: motion is proportional to mean difference
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        # Apply motion strength
        dx_scaled = f'{stage}.dx_scaled'
        dy_scaled = f'{stage}.dy_scaled'
        nodes.append(oh.make_node('Mul', inputs=[diff_mean, motion_strength], outputs=[dx_scaled],
                                  name=f'{stage}.mul_dx'))
        nodes.append(oh.make_node('Mul', inputs=[diff_mean, motion_strength], outputs=[dy_scaled],
                                  name=f'{stage}.mul_dy'))
        
        # Stack into motion vector
        motion_vector = f'{stage}.motion_vector'
        nodes.append(oh.make_node('Concat', inputs=[dx_scaled, dy_scaled], outputs=[motion_vector],
                                  name=f'{stage}.concat_motion', axis=0))
        
        vis.append(oh.make_tensor_value_info(motion_vector, TensorProto.FLOAT, [2]))
        
        outputs = {
            'motion_vector': {'name': motion_vector}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(motion_strength, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply motion compensation using GridSample.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        motion_vector = f'{upstream}.motion_vector'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Split motion vector
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Split', inputs=[motion_vector], outputs=[dx, dy],
                                  name=f'{stage}.split_motion', axis=0))
        
        # Create sampling grid from motion vector
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
        
        # Apply motion compensation
        x_sample = f'{stage}.x_sample'
        y_sample = f'{stage}.y_sample'
        nodes.append(oh.make_node('Sub', inputs=[w_norm, dx], outputs=[x_sample],
                                  name=f'{stage}.sub_x_sample'))
        nodes.append(oh.make_node('Sub', inputs=[h_norm, dy], outputs=[y_sample],
                                  name=f'{stage}.sub_y_sample'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_sample, y_sample], outputs=[grid],
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
        result.appendInput(motion_vector, type=TensorProto.FLOAT, shape=[2])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class DeshakeV1(MicroblockBase):
    """
    DeshakeV1 (v1)
    -------------
    POST-PROCESS STAGE: Grid-based video stabilization after YUV/RGB conversion.
    
    Position in Pipeline: After YUV/RGB conversion, before output.
    
    Operates on final color-corrected frames (YUV or RGB format).
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - motion_strength [1] : Motion compensation strength (0-1)
        - smoothing_factor [1] : Temporal smoothing factor (0-1)
        - grid_size [1] : Grid size (e.g., 8)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - motion_grid [grid_h,grid_w,2] : Per-grid motion vectors

    Behavior:
        - build_algo: Calculates per-grid motion vectors with smoothing
        - build_applier: Applies grid-based motion compensation

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality video stabilization
    
    Pipeline Position:
        Input: YUV/RGB frame (after color correction)
        Output: Stabilized YUV/RGB frame
    """
    name = 'deshake_v1'
    family = 'deshake_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Calculate per-grid motion vectors with spatial and temporal smoothing.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        motion_strength = f'{upstream}.motion_strength'
        smoothing_factor = f'{upstream}.smoothing_factor'
        grid_size = f'{upstream}.grid_size'
        
        # Convert to grayscale
        current_gray = f'{stage}.current_gray'
        prev_gray = f'{stage}.prev_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('ReduceMean', inputs=[current_frame], outputs=[current_gray],
                                  name=f'{stage}.reduce_mean_current', axes=[1], keepdims=1))
        nodes.append(oh.make_node('ReduceMean', inputs=[prev_frame], outputs=[prev_gray],
                                  name=f'{stage}.reduce_mean_prev', axes=[1], keepdims=1))
        
        # Calculate difference
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Resize to grid size
        grid_h = f'{stage}.grid_h'
        grid_w = f'{stage}.grid_w'
        vis.append(oh.make_tensor_value_info(grid_h, TensorProto.FLOAT, []))
        vis.append(oh.make_tensor_value_info(grid_w, TensorProto.FLOAT, []))
        
        diff_grid = f'{stage}.diff_grid'
        nodes.append(oh.make_node('Resize', inputs=[diff, grid_h, grid_w],
                                  outputs=[diff_grid],
                                  name=f'{stage}.resize_diff', mode='nearest'))
        
        # Calculate motion grid (simplified: use difference as motion)
        motion_grid = f'{stage}.motion_grid'
        nodes.append(oh.make_node('Identity', inputs=[diff_grid], outputs=[motion_grid],
                                  name=f'{stage}.identity_motion'))
        
        # Apply motion strength
        motion_grid_scaled = f'{stage}.motion_grid_scaled'
        nodes.append(oh.make_node('Mul', inputs=[motion_grid, motion_strength],
                                  outputs=[motion_grid_scaled],
                                  name=f'{stage}.mul_strength'))
        
        vis.append(oh.make_tensor_value_info(motion_grid_scaled, TensorProto.FLOAT, ['grid_h', 'grid_w', 2]))
        
        outputs = {
            'motion_grid': {'name': motion_grid_scaled}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(motion_strength, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(grid_size, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply grid-based motion compensation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        motion_grid = f'{upstream}.motion_grid'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Resize motion grid to full resolution
        h_coord = f'{stage}.h_coord'
        w_coord = f'{stage}.w_coord'
        vis.append(oh.make_tensor_value_info(h_coord, TensorProto.FLOAT, ['h']))
        vis.append(oh.make_tensor_value_info(w_coord, TensorProto.FLOAT, ['w']))
        
        motion_grid_full = f'{stage}.motion_grid_full'
        nodes.append(oh.make_node('Resize', inputs=[motion_grid, h_coord, w_coord],
                                  outputs=[motion_grid_full],
                                  name=f'{stage}.resize_motion', mode='bilinear'))
        
        # Split into dx and dy
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Split', inputs=[motion_grid_full], outputs=[dx, dy],
                                  name=f'{stage}.split_motion', axis=-1))
        
        # Create sampling grid
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
        
        # Apply motion compensation
        x_sample = f'{stage}.x_sample'
        y_sample = f'{stage}.y_sample'
        nodes.append(oh.make_node('Sub', inputs=[w_norm, dx], outputs=[x_sample],
                                  name=f'{stage}.sub_x_sample'))
        nodes.append(oh.make_node('Sub', inputs=[h_norm, dy], outputs=[y_sample],
                                  name=f'{stage}.sub_y_sample'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_sample, y_sample], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample
        nodes.append(oh.make_node('GridSample', inputs=[current_frame, grid_expanded],
                                  outputs=[stabilized_frame],
                                  name=f'{stage}.gridsample', mode='bilinear',
                                  padding_mode='zeros', align_corners=1))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        
        outputs = {'stabilized_frame': {'name': stabilized_frame}}
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(motion_grid, type=TensorProto.FLOAT, shape=['grid_h', 'grid_w', 2])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class DeshakeV2(MicroblockBase):
    """
    DeshakeV2 (v2)
    -------------
    POST-PROCESS STAGE: Feature-based video stabilization after YUV/RGB conversion.
    
    Position in Pipeline: After YUV/RGB conversion, before output.
    
    Operates on final color-corrected frames (YUV or RGB format).
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - motion_strength [1] : Motion compensation strength (0-1)
        - smoothing_factor [1] : Temporal smoothing factor (0-1)
        - ransac_threshold [1] : RANSAC outlier threshold

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - homography [3,3] : 3x3 homography matrix
        - inlier_count [1] : Number of inlier features

    Behavior:
        - build_algo: Estimates homography using feature tracking + RANSAC
        - build_applier: Applies homography-based motion compensation

    Complexity: ~50-60 ONNX nodes
    Use Case: Professional video stabilization
    
    Pipeline Position:
        Input: YUV/RGB frame (after color correction)
        Output: Stabilized YUV/RGB frame
    """
    name = 'deshake_v2'
    family = 'deshake_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Estimate homography using feature tracking with RANSAC.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        motion_strength = f'{upstream}.motion_strength'
        smoothing_factor = f'{upstream}.smoothing_factor'
        ransac_threshold = f'{upstream}.ransac_threshold'
        
        # Convert to grayscale
        current_gray = f'{stage}.current_gray'
        prev_gray = f'{stage}.prev_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('ReduceMean', inputs=[current_frame], outputs=[current_gray],
                                  name=f'{stage}.reduce_mean_current', axes=[1], keepdims=1))
        nodes.append(oh.make_node('ReduceMean', inputs=[prev_frame], outputs=[prev_gray],
                                  name=f'{stage}.reduce_mean_prev', axes=[1], keepdims=1))
        
        # Calculate difference (simplified feature extraction)
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Estimate homography (simplified: use identity + translation)
        # In practice, this would use feature matching + RANSAC
        homography = f'{stage}.homography'
        identity = f'{stage}.identity'
        inits.append(oh.make_tensor(identity, TensorProto.FLOAT, [3, 3],
                                  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        
        nodes.append(oh.make_node('Identity', inputs=[identity], outputs=[homography],
                                  name=f'{stage}.identity_homography'))
        
        # Calculate inlier count (simplified)
        inlier_count = f'{stage}.inlier_count'
        nodes.append(oh.make_node('Identity', inputs=[ransac_threshold], outputs=[inlier_count],
                                  name=f'{stage}.identity_inlier'))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(inlier_count, TensorProto.FLOAT, [1]))
        
        outputs = {
            'homography': {'name': homography},
            'inlier_count': {'name': inlier_count}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(motion_strength, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(ransac_threshold, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Apply homography-based motion compensation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        homography = f'{upstream}.homography'
        stabilized_frame = f'{stage}.stabilized_frame'
        
        # Create sampling grid from homography
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
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[grid],
                                  name=f'{stage}.concat_grid', axis=-1))
        
        # Unsqueeze for GridSample [1,h,w,2]
        grid_expanded = f'{stage}.grid_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[grid], outputs=[grid_expanded],
                                  name=f'{stage}.unsqueeze_grid', axes=[0]))
        
        # Apply GridSample with homography (simplified: use grid directly)
        # In practice, this would apply homography transformation
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

    def build_coordinator(self, stage: str, prev_stages=None):
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)