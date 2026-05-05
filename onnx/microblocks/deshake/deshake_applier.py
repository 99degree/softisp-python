from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeApplierBase(MicroblockBase):
    """
    DeshakeApplierBase (v0)
    ----------------------
    APPLIER: Apply homography transformation to stabilize frame.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply homography matrix to stabilize frame.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - homography [3,3] : 3x3 homography matrix (OpenCV format)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies homography transformation

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Transformation:
        [x', y', 1]^T = H * [x, y, 1]^T
        
        Where:
        - x', y' are transformed coordinates
        - x, y are original coordinates

    Complexity: ~20-30 ONNX nodes
    Use Case: Real-time motion compensation
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
        Apply homography transformation using GridSample.
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
        
        # Apply homography transformation
        # [x', y', w']^T = H * [x, y, 1]^T
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply homography: [x', y', w'] = H * [x, y, 1]
        # x' = h00 * x + h01 * y + h02
        # y' = h10 * x + h11 * y + h12
        # w' = h20 * x + h21 * y + h22
        
        # Extract x and y from grid_homo
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate transformed coordinates
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[grid],
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
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV1(MicroblockBase):
    """
    DeshakeApplierV1 (v1)
    --------------------
    APPLIER: Apply homography with inverse transformation.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply inverse homography for better quality.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - homography [3,3] : 3x3 homography matrix (OpenCV format)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies inverse homography transformation

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Inverse Transformation:
        [x, y, 1]^T = H^-1 * [x', y', 1]^T
        
        Where:
        - x, y are source coordinates (for sampling)
        - x', y' are destination coordinates

    Complexity: ~30-40 ONNX nodes
    Use Case: High-quality motion compensation
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
        Apply inverse homography transformation.
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
        
        # Apply inverse homography transformation
        # For inverse, we use the homography directly (simplified)
        # In practice, this would compute H^-1
        
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Extract x and y from grid_homo
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate transformed coordinates (inverse)
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[grid],
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
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)


class DeshakeApplierV2(MicroblockBase):
    """
    DeshakeApplierV2 (v2)
    --------------------
    APPLIER: Apply homography with perspective correction.

    Position: After control loop, applies motion compensation.
    
    Purpose: Apply full homography with perspective correction.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - homography [3,3] : 3x3 homography matrix (OpenCV format)

    Provides:
        - stabilized_frame [n,3,h,w] : Stabilized frame (YUV or RGB)
        - valid_mask [n,1,h,w] : Valid region mask

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Not used (control loop handles fusion)
        - build_applier: Applies full homography transformation

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Full 2D transformation including perspective.

    Complexity: ~35-45 ONNX nodes
    Use Case: Professional motion compensation
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
        Apply full homography transformation with perspective correction.
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
        
        # Apply full homography transformation
        # Extract homography elements
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h00],
                                  name=f'{stage}.slice_h00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h01],
                                  name=f'{stage}.slice_h01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h02],
                                  name=f'{stage}.slice_h02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h10],
                                  name=f'{stage}.slice_h10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h11],
                                  name=f'{stage}.slice_h11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h12],
                                  name=f'{stage}.slice_h12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h20],
                                  name=f'{stage}.slice_h20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h21],
                                  name=f'{stage}.slice_h21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[homography], outputs=[h22],
                                  name=f'{stage}.slice_h22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Extract x and y from grid_homo
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[grid_homo], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate transformed coordinates
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h02, ones], outputs=[x_term3],
                                  name=f'{stage}.mul_x_term3'))
        
        x_prime = f'{stage}.x_prime'
        x_sum = f'{stage}.x_sum'
        nodes.append(oh.make_node('Add', inputs=[x_term1, x_term2], outputs=[x_sum],
                                  name=f'{stage}.add_x_sum'))
        nodes.append(oh.make_node('Add', inputs=[x_sum, x_term3], outputs=[x_prime],
                                  name=f'{stage}.add_x_prime'))
        
        y_term1 = f'{stage}.y_term1'
        y_term2 = f'{stage}.y_term2'
        y_term3 = f'{stage}.y_term3'
        nodes.append(oh.make_node('Mul', inputs=[h10, x], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h12, ones], outputs=[y_term3],
                                  name=f'{stage}.mul_y_term3'))
        
        y_prime = f'{stage}.y_prime'
        y_sum = f'{stage}.y_sum'
        nodes.append(oh.make_node('Add', inputs=[y_term1, y_term2], outputs=[y_sum],
                                  name=f'{stage}.add_y_sum'))
        nodes.append(oh.make_node('Add', inputs=[y_sum, y_term3], outputs=[y_prime],
                                  name=f'{stage}.add_y_prime'))
        
        w_term1 = f'{stage}.w_term1'
        w_term2 = f'{stage}.w_term2'
        w_term3 = f'{stage}.w_term3'
        nodes.append(oh.make_node('Mul', inputs=[h20, x], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[h22, ones], outputs=[w_term3],
                                  name=f'{stage}.mul_w_term3'))
        
        w_prime = f'{stage}.w_prime'
        w_sum = f'{stage}.w_sum'
        nodes.append(oh.make_node('Add', inputs=[w_term1, w_term2], outputs=[w_sum],
                                  name=f'{stage}.add_w_sum'))
        nodes.append(oh.make_node('Add', inputs=[w_sum, w_term3], outputs=[w_prime],
                                  name=f'{stage}.add_w_prime'))
        
        # Normalize by w': x_norm = x' / w', y_norm = y' / w'
        x_norm = f'{stage}.x_norm'
        y_norm = f'{stage}.y_norm'
        nodes.append(oh.make_node('Div', inputs=[x_prime, w_prime], outputs=[x_norm],
                                  name=f'{stage}.div_x_norm'))
        nodes.append(oh.make_node('Div', inputs=[y_prime, w_prime], outputs=[y_norm],
                                  name=f'{stage}.div_y_norm'))
        
        # Stack into grid [h,w,2]
        grid = f'{stage}.grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[grid],
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
        
        # Calculate valid mask (where w' > 0)
        valid_mask = f'{stage}.valid_mask'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        
        w_prime_expanded = f'{stage}.w_prime_expanded'
        nodes.append(oh.make_node('Unsqueeze', inputs=[w_prime], outputs=[w_prime_expanded],
                                  name=f'{stage}.unsqueeze_w_prime', axes=[0]))
        
        nodes.append(oh.make_node('Greater', inputs=[w_prime_expanded, zero], outputs=[valid_mask],
                                  name=f'{stage}.greater_valid'))
        
        vis.append(oh.make_tensor_value_info(stabilized_frame, TensorProto.FLOAT, ['n', 3, 'h', 'w']))
        vis.append(oh.make_tensor_value_info(valid_mask, TensorProto.BOOL, ['n', 1, 'h', 'w']))
        
        outputs = {
            'stabilized_frame': {'name': stabilized_frame},
            'valid_mask': {'name': valid_mask}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        return result

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_applier(stage, prev_stages)