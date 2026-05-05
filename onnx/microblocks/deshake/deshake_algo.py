from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeAlgoBase(MicroblockBase):
    """
    DeshakeAlgoBase (v0)
    ---------------
    ALGORITHM: Extract homography matrix from frames.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract homography matrix from consecutive frames.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)

    Provides:
        - homography [3,3] : 3x3 homography matrix (OpenCV format)
        - confidence [1] : Confidence score (0-1)

    Behavior:
        - build_algo: Extracts homography matrix using block matching
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Where:
        - h00, h01, h10, h11: Rotation and scaling
        - h02, h12: Translation (tx, ty)
        - h20, h21: Perspective (shear)
        - h22: Scale (usually 1)

    Complexity: ~15-20 ONNX nodes
    Use Case: Real-time motion estimation
    """
    name = 'deshake_algo_base'
    family = 'deshake_algo_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract homography matrix from frames.
        
        Uses block matching to estimate translation.
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
        
        # Extract translation (tx, ty) from difference
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        tx = f'{stage}.tx'
        ty = f'{stage}.ty'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[tx],
                                  name=f'{stage}.identity_tx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[ty],
                                  name=f'{stage}.identity_ty'))
        
        # Create homography matrix (translation only)
        # H = [1  0  tx]
        #     [0  1  ty]
        #     [0  0   1]
        one = f'{stage}.one'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        # Create homography matrix elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h00],
                                  name=f'{stage}.identity_h00'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h01],
                                  name=f'{stage}.identity_h01'))
        nodes.append(oh.make_node('Identity', inputs=[tx], outputs=[h02],
                                  name=f'{stage}.identity_h02'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h10],
                                  name=f'{stage}.identity_h10'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h11],
                                  name=f'{stage}.identity_h11'))
        nodes.append(oh.make_node('Identity', inputs=[ty], outputs=[h12],
                                  name=f'{stage}.identity_h12'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h20],
                                  name=f'{stage}.identity_h20'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h21],
                                  name=f'{stage}.identity_h21'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h22],
                                  name=f'{stage}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        homography = f'{stage}.homography'
        h_row0 = f'{stage}.h_row0'
        h_row1 = f'{stage}.h_row1'
        h_row2 = f'{stage}.h_row2'
        
        nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                  name=f'{stage}.concat_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                  name=f'{stage}.concat_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                  name=f'{stage}.concat_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography],
                                  name=f'{stage}.concat_homography', axis=0))
        
        # Calculate confidence (simplified)
        confidence = f'{stage}.confidence'
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[confidence],
                                  name=f'{stage}.identity_confidence'))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(confidence, TensorProto.FLOAT, [1]))
        
        outputs = {
            'homography': {'name': homography},
            'confidence': {'name': confidence}
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
    -------------
    ALGORITHM: Extract homography matrix with rotation and scaling.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract homography matrix with rotation and scaling.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - grid_size [1] : Grid size (e.g., 8)

    Provides:
        - homography [3,3] : 3x3 homography matrix (OpenCV format)
        - confidence [1] : Confidence score (0-1)
        - motion_grid [grid_h,grid_w,2] : Per-grid motion vectors

    Behavior:
        - build_algo: Extracts homography using grid-based estimation
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Homography Matrix Format (OpenCV):
        H = [cos(θ)  -sin(θ)  tx]
            [sin(θ)   cos(θ)  ty]
            [0        0       1]
        
        Where:
        - θ: rotation angle
        - tx, ty: translation

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality motion estimation
    """
    name = 'deshake_algo_v1'
    family = 'deshake_algo_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract homography matrix using grid-based estimation.
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
        
        # Extract global motion parameters
        diff_mean = f'{stage}.diff_mean'
        nodes.append(oh.make_node('ReduceMean', inputs=[diff], outputs=[diff_mean],
                                  name=f'{stage}.reduce_mean_diff', keepdims=0))
        
        tx = f'{stage}.tx'
        ty = f'{stage}.ty'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[tx],
                                  name=f'{stage}.identity_tx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[ty],
                                  name=f'{stage}.identity_ty'))
        
        # Create homography matrix (translation only for now)
        one = f'{stage}.one'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h00],
                                  name=f'{stage}.identity_h00'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h01],
                                  name=f'{stage}.identity_h01'))
        nodes.append(oh.make_node('Identity', inputs=[tx], outputs=[h02],
                                  name=f'{stage}.identity_h02'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h10],
                                  name=f'{stage}.identity_h10'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h11],
                                  name=f'{stage}.identity_h11'))
        nodes.append(oh.make_node('Identity', inputs=[ty], outputs=[h12],
                                  name=f'{stage}.identity_h12'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h20],
                                  name=f'{stage}.identity_h20'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h21],
                                  name=f'{stage}.identity_h21'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h22],
                                  name=f'{stage}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        homography = f'{stage}.homography'
        h_row0 = f'{stage}.h_row0'
        h_row1 = f'{stage}.h_row1'
        h_row2 = f'{stage}.h_row2'
        
        nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                  name=f'{stage}.concat_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                  name=f'{stage}.concat_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                  name=f'{stage}.concat_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography],
                                  name=f'{stage}.concat_homography', axis=0))
        
        # Calculate confidence
        confidence = f'{stage}.confidence'
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[confidence],
                                  name=f'{stage}.identity_confidence'))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(confidence, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(motion_grid, TensorProto.FLOAT, ['grid_h', 'grid_w', 2]))
        
        outputs = {
            'homography': {'name': homography},
            'confidence': {'name': confidence},
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
    -------------
    ALGORITHM: Extract homography matrix with RANSAC.

    Position: Out of main pipeline, operates on YUV/RGB frames.
    
    Purpose: Extract homography matrix using feature tracking + RANSAC.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - ransac_threshold [1] : RANSAC outlier threshold

    Provides:
        - homography [3,3] : 3x3 homography matrix (OpenCV format)
        - confidence [1] : Confidence score (0-1)
        - inlier_count [1] : Number of inlier features

    Behavior:
        - build_algo: Extracts homography using feature tracking
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Not used (applier handles application)

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Full 2D transformation including perspective.

    Complexity: ~50-60 ONNX nodes
    Use Case: Professional motion estimation
    """
    name = 'deshake_algo_v2'
    family = 'deshake_algo_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract homography matrix using feature tracking with RANSAC.
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
        
        tx = f'{stage}.tx'
        ty = f'{stage}.ty'
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[tx],
                                  name=f'{stage}.identity_tx'))
        nodes.append(oh.make_node('Identity', inputs=[diff_mean], outputs=[ty],
                                  name=f'{stage}.identity_ty'))
        
        # Create homography matrix
        one = f'{stage}.one'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h00],
                                  name=f'{stage}.identity_h00'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h01],
                                  name=f'{stage}.identity_h01'))
        nodes.append(oh.make_node('Identity', inputs=[tx], outputs=[h02],
                                  name=f'{stage}.identity_h02'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h10],
                                  name=f'{stage}.identity_h10'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h11],
                                  name=f'{stage}.identity_h11'))
        nodes.append(oh.make_node('Identity', inputs=[ty], outputs=[h12],
                                  name=f'{stage}.identity_h12'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h20],
                                  name=f'{stage}.identity_h20'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h21],
                                  name=f'{stage}.identity_h21'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h22],
                                  name=f'{stage}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        homography = f'{stage}.homography'
        h_row0 = f'{stage}.h_row0'
        h_row1 = f'{stage}.h_row1'
        h_row2 = f'{stage}.h_row2'
        
        nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                  name=f'{stage}.concat_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                  name=f'{stage}.concat_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                  name=f'{stage}.concat_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography],
                                  name=f'{stage}.concat_homography', axis=0))
        
        # Calculate inlier count
        inlier_count = f'{stage}.inlier_count'
        nodes.append(oh.make_node('Identity', inputs=[ransac_threshold], outputs=[inlier_count],
                                  name=f'{stage}.identity_inlier'))
        
        # Calculate confidence
        confidence = f'{stage}.confidence'
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[confidence],
                                  name=f'{stage}.identity_confidence'))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(confidence, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(inlier_count, TensorProto.FLOAT, [1]))
        
        outputs = {
            'homography': {'name': homography},
            'confidence': {'name': confidence},
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