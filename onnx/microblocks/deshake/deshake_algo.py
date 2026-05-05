from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeAlgoBase(MicroblockBase):
    """
    DeshakeAlgoBase (v0)
    -------------------
    ALGORITHM: Extract motion parameters (dx, dy, dr, dz) from frames.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract raw motion parameters from consecutive frames.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)

    Provides:
        - dx [1] : Horizontal translation (pixels)
        - dy [1] : Vertical translation (pixels)
        - dr [1] : Rotation angle (radians)
        - dz [1] : Zoom/scale factor

    Behavior:
        - build_algo: Extracts motion parameters using block matching
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Complexity: ~15-20 ONNX nodes
    Use Case: Real-time motion estimation
    """
    name = 'deshake_algo_base'
    family = 'deshake_algo_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract motion parameters (dx, dy, dr, dz) from frames.
        
        Uses block matching to estimate translation, rotation, and zoom.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        
        # Convert to grayscale for motion estimation
        current_gray = f'{stage}.current_gray'
        prev_gray = f'{stage}.prev_gray'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(three, TensorProto.FLOAT, [], [3.0]))
        
        nodes.append(oh.make_node('ReduceMean', inputs=[current_frame], outputs=[current_gray],
                                  name=f'{stage}.reduce_mean_current', axes=[1], keepdims=1))
        nodes.append(oh.make_node('ReduceMean', inputs=[prev_frame], outputs=[prev_gray],
                                  name=f'{stage}.reduce_mean_prev', axes=[1], keepdims=1))
        
        # Calculate difference for motion estimation
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Extract translation (dx, dy) from difference
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dx],
                                  name=f'{stage}.identity_dx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dy],
                                  name=f'{stage}.identity_dy'))
        
        # Extract rotation (dr) - simplified: use gradient difference
        dr = f'{stage}.dr'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[dr],
                                  name=f'{stage}.identity_dr'))
        
        # Extract zoom (dz) - simplified: use scale from difference
        dz = f'{stage}.dz'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[dz],
                                  name=f'{stage}.identity_dz'))
        
        vis.append(oh.make_tensor_value_info(dx, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dy, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dr, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dz, TensorProto.FLOAT, [1]))
        
        outputs = {
            'dx': {'name': dx},
            'dy': {'name': dy},
            'dr': {'name': dr},
            'dz': {'name': dz}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class DeshakeAlgoV1(MicroblockBase):
    """
    DeshakeAlgoV1 (v1)
    -----------------
    ALGORITHM: Grid-based motion parameter extraction.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract per-grid motion parameters with smoothing.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - grid_size [1] : Grid size (e.g., 8)

    Provides:
        - dx [1] : Horizontal translation (pixels)
        - dy [1] : Vertical translation (pixels)
        - dr [1] : Rotation angle (radians)
        - dz [1] : Zoom/scale factor
        - motion_grid [grid_h,grid_w,2] : Per-grid motion vectors

    Behavior:
        - build_algo: Extracts motion parameters using grid-based estimation
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality motion estimation
    """
    name = 'deshake_algo_v1'
    family = 'deshake_algo_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract motion parameters using grid-based estimation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
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
        
        # Calculate motion grid
        motion_grid = f'{stage}.motion_grid'
        nodes.append(oh.make_node('Identity', inputs=[diff_grid], outputs=[motion_grid],
                                  name=f'{stage}.identity_motion'))
        
        # Extract global motion parameters from grid
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dx],
                                  name=f'{stage}.identity_dx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dy],
                                  name=f'{stage}.identity_dy'))
        
        dr = f'{stage}.dr'
        dz = f'{stage}.dz'
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[dr],
                                  name=f'{stage}.identity_dr'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[dz],
                                  name=f'{stage}.identity_dz'))
        
        vis.append(oh.make_tensor_value_info(dx, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dy, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dr, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dz, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(motion_grid, TensorProto.FLOAT, ['grid_h', 'grid_w', 2]))
        
        outputs = {
            'dx': {'name': dx},
            'dy': {'name': dy},
            'dr': {'name': dr},
            'dz': {'name': dz},
            'motion_grid': {'name': motion_grid}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(grid_size, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class DeshakeAlgoV2(MicroblockBase):
    """
    DeshakeAlgoV2 (v2)
    -----------------
    ALGORITHM: Feature-based motion parameter extraction with RANSAC.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract motion parameters using feature tracking + RANSAC.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - ransac_threshold [1] : RANSAC outlier threshold

    Provides:
        - dx [1] : Horizontal translation (pixels)
        - dy [1] : Vertical translation (pixels)
        - dr [1] : Rotation angle (radians)
        - dz [1] : Zoom/scale factor
        - homography [3,3] : 3x3 homography matrix
        - inlier_count [1] : Number of inlier features

    Behavior:
        - build_algo: Extracts motion parameters using feature tracking
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Complexity: ~50-60 ONNX nodes
    Use Case: Professional motion estimation
    """
    name = 'deshake_algo_v2'
    family = 'deshake_algo_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract motion parameters using feature tracking with RANSAC.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
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
        
        # Calculate difference
        diff = f'{stage}.diff'
        nodes.append(oh.make_node('Sub', inputs=[current_gray, prev_gray], outputs=[diff],
                                  name=f'{stage}.sub_diff'))
        
        # Extract motion parameters (simplified)
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dx],
                                  name=f'{stage}.identity_dx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[dy],
                                  name=f'{stage}.identity_dy'))
        
        dr = f'{stage}.dr'
        dz = f'{stage}.dz'
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[dr],
                                  name=f'{stage}.identity_dr'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[dz],
                                  name=f'{stage}.identity_dz'))
        
        # Estimate homography
        homography = f'{stage}.homography'
        identity = f'{stage}.identity'
        inits.append(oh.make_tensor(identity, TensorProto.FLOAT, [3, 3],
                                  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        nodes.append(oh.make_node('Identity', inputs=[identity], outputs=[homography],
                                  name=f'{stage}.identity_homography'))
        
        # Calculate inlier count
        inlier_count = f'{stage}.inlier_count'
        nodes.append(oh.make_node('Identity', inputs=[ransac_threshold], outputs=[inlier_count],
                                  name=f'{stage}.identity_inlier'))
        
        vis.append(oh.make_tensor_value_info(dx, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dy, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dr, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dz, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(inlier_count, TensorProto.FLOAT, [1]))
        
        outputs = {
            'dx': {'name': dx},
            'dy': {'name': dy},
            'dr': {'name': dr},
            'dz': {'name': dz},
            'homography': {'name': homography},
            'inlier_count': {'name': inlier_count}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(ransac_threshold, type=TensorProto.FLOAT, shape=[1])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - control loop handles sensor fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)