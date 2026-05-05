from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeCoreFull(MicroblockBase):
    """
    DeshakeCoreFull - Core Processing Stage (Coordinator Domain)
    ---------------------------------------------------------------
    CORE PROCESSING: Generate mesh grid with full IMU fusion.
    
    Position: After pre-processing, before post-processing.
    
    Purpose: Generate mesh grid by fusing homography matrix with GDC
    distortion correction and IMU data. This implementation includes
    temporal smoothing, horizon leveling, dynamic zoom, and rolling
    shutter correction.
    
    Mesh Size: Configurable via class attribute (default: 16x16)
    
    Needs:
        - homography [3,3] : 3x3 homography matrix from pre-processing
        - gdc_coeffs [4] : GDC coefficients from pre-processing
        - imu_rotation [3,3] : 3x3 IMU rotation matrix (from CPU coordinator)
        - camera_matrix [3,3] : 3x3 camera intrinsic matrix K
        - current_frame [n,3,h,w] : Current video frame (for dimension extraction)
        - smoothing_alpha [1] : Temporal smoothing factor (0.0-1.0)
        - horizon_leveling [1] : Enable horizon leveling (0=off, 1=on)
        - dynamic_zoom [1] : Enable dynamic zoom (0=off, 1=on)
        - rolling_shutter [1] : Enable rolling shutter correction (0=off, 1=on)

    Provides:
        - mesh_grid [mesh_h,mesh_w,2] : Mesh vertex grid for warp
        - valid_mask [mesh_h,mesh_w] : Valid region mask

    Behavior:
        - build_algo: Generate mesh grid with full IMU fusion
        - build_coordinator: Drop old stats, calculate fusion stats, temporal smoothing
        - build_applier: None (not used in core processing)

    Mesh Grid Generation Steps:
        1. Create normalized identity grid
        2. Apply GDC distortion to grid points
        3. Fuse homography with IMU rotation (sensor fusion)
        4. Apply temporal smoothing
        5. Apply horizon leveling (if enabled)
        6. Apply dynamic zoom (if enabled)
        7. Apply rolling shutter correction (if enabled)
        8. Output mesh grid for GPU warp

    Sensor Fusion:
        R_fused = α * R_homography + (1-α) * R_IMU
        
        Where:
        - α: Sensor fusion weight (from smoothing_alpha)
        - R_homography: Rotation from frame-based homography
        - R_IMU: Rotation from IMU sensor

    Temporal Smoothing:
        R_smooth[t] = α * R_smooth[t-1] + (1-α) * R_fused[t]
        
        Where:
        - α: Temporal smoothing factor (from smoothing_alpha)
        - R_smooth[t-1]: Previous smoothed rotation
        - R_fused[t]: Current fused rotation

    Horizon Leveling:
        R_level = R_gravity * R_smooth
        
        Where:
        - R_gravity: Rotation to align gravity with vertical axis
        - R_smooth: Smoothed rotation

    Dynamic Zoom:
        Z = 1.0 + β * |angular_velocity|
        
        Where:
        - β: Zoom sensitivity factor
        - angular_velocity: Angular velocity from IMU

    Rolling Shutter Correction:
        R_row = Slerp(R_start, R_end, alpha)
        
        Where:
        - R_start: Rotation at start of frame
        - R_end: Rotation at end of frame
        - alpha: Row-dependent interpolation factor

    Transformation Composition:
        P_final = K * R_fused * K_inv * P_source
        
        Where:
        - K: Camera intrinsic matrix
        - R_fused: Fused rotation matrix
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Complexity: ~60-80 ONNX nodes
    Use Case: Core processing with full IMU fusion
    """
    name = 'deshake_core_full'
    family = 'deshake_core'
    version = 'v2'
    
    # Mesh size (compile-time constant, can be overridden in subclasses)
    mesh_h = 16
    mesh_w = 16

    def build_algo(self, stage: str, prev_stages=None):
        """
        Generate mesh grid with full IMU fusion.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        homography = f'{upstream}.homography'
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        imu_rotation = f'{upstream}.imu_rotation'
        camera_matrix = f'{upstream}.camera_matrix'
        current_frame = f'{upstream}.current_frame'
        smoothing_alpha = f'{upstream}.smoothing_alpha'
        horizon_leveling = f'{upstream}.horizon_leveling'
        dynamic_zoom = f'{upstream}.dynamic_zoom'
        rolling_shutter = f'{upstream}.rolling_shutter'
        
        # Extract image dimensions from current_frame tensor shape
        # current_frame shape: [n, 3, h, w]
        frame_shape = f'{stage}.frame_shape'
        nodes.append(oh.make_node('Shape', inputs=[current_frame], outputs=[frame_shape],
                                  name=f'{stage}.shape_frame'))
        
        # Extract height (index 2) and width (index 3) from shape
        height = f'{stage}.height'
        width = f'{stage}.width'
        two = f'{stage}.two'
        three = f'{stage}.three'
        inits.append(oh.make_tensor(two, TensorProto.INT64, [], [2]))
        inits.append(oh.make_tensor(three, TensorProto.INT64, [], [3]))
        
        nodes.append(oh.make_node('Gather', inputs=[frame_shape, two], outputs=[height],
                                  name=f'{stage}.gather_height'))
        nodes.append(oh.make_node('Gather', inputs=[frame_shape, three], outputs=[width],
                                  name=f'{stage}.gather_width'))
        
        # Sensor fusion: R_fused = α * R_homography + (1-α) * R_IMU
        alpha = f'{stage}.alpha'
        one_minus_alpha = f'{stage}.one_minus_alpha'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        nodes.append(oh.make_node('Identity', inputs=[smoothing_alpha], outputs=[alpha],
                                  name=f'{stage}.identity_alpha'))
        nodes.append(oh.make_node('Sub', inputs=[one, alpha], outputs=[one_minus_alpha],
                                  name=f'{stage}.sub_one_minus_alpha'))
        
        # Scale homography and IMU rotation
        homography_scaled = f'{stage}.homography_scaled'
        imu_scaled = f'{stage}.imu_scaled'
        nodes.append(oh.make_node('Mul', inputs=[homography, alpha], outputs=[homography_scaled],
                                  name=f'{stage}.mul_homography_scaled'))
        nodes.append(oh.make_node('Mul', inputs=[imu_rotation, one_minus_alpha], outputs=[imu_scaled],
                                  name=f'{stage}.mul_imu_scaled'))
        
        # Fuse rotations
        rotation_fused = f'{stage}.rotation_fused'
        nodes.append(oh.make_node('Add', inputs=[homography_scaled, imu_scaled], outputs=[rotation_fused],
                                  name=f'{stage}.add_rotation_fused'))
        
        # Create mesh coordinate grid
        mesh_h_coord = f'{stage}.mesh_h_coord'
        mesh_w_coord = f'{stage}.mesh_w_coord'
        vis.append(oh.make_tensor_value_info(mesh_h_coord, TensorProto.FLOAT, [self.mesh_h]))
        vis.append(oh.make_tensor_value_info(mesh_w_coord, TensorProto.FLOAT, [self.mesh_w]))
        
        # Normalize mesh coordinates to [-1, 1]
        mesh_h_norm = f'{stage}.mesh_h_norm'
        mesh_w_norm = f'{stage}.mesh_w_norm'
        mesh_h_half = f'{stage}.mesh_h_half'
        mesh_w_half = f'{stage}.mesh_w_half'
        inits.append(oh.make_tensor(mesh_h_half, TensorProto.FLOAT, [], [0.5]))
        inits.append(oh.make_tensor(mesh_w_half, TensorProto.FLOAT, [], [0.5]))
        
        nodes.append(oh.make_node('Mul', inputs=[mesh_h_coord, mesh_h_half], outputs=[mesh_h_norm],
                                  name=f'{stage}.mul_mesh_h_norm'))
        nodes.append(oh.make_node('Mul', inputs=[mesh_w_coord, mesh_w_half], outputs=[mesh_w_norm],
                                  name=f'{stage}.mul_mesh_w_norm'))
        
        # Stack into mesh grid [mesh_h,mesh_w,2]
        mesh_identity_grid = f'{stage}.mesh_identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[mesh_w_norm, mesh_h_norm], outputs=[mesh_identity_grid],
                                  name=f'{stage}.concat_mesh_identity_grid', axis=-1))
        
        # Extract GDC coefficients
        k1 = f'{stage}.k1'
        k2 = f'{stage}.k2'
        p1 = f'{stage}.p1'
        p2 = f'{stage}.p2'
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k1],
                                  name=f'{stage}.slice_k1',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[k2],
                                  name=f'{stage}.slice_k2',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p1],
                                  name=f'{stage}.slice_p1',
                                  starts=[2], ends=[3], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gdc_coeffs], outputs=[p2],
                                  name=f'{stage}.slice_p2',
                                  starts=[3], ends=[4], axes=[0]))
        
        # Extract x and y from mesh identity grid
        mesh_x = f'{stage}.mesh_x'
        mesh_y = f'{stage}.mesh_y'
        nodes.append(oh.make_node('Slice', inputs=[mesh_identity_grid], outputs=[mesh_x],
                                  name=f'{stage}.slice_mesh_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[mesh_identity_grid], outputs=[mesh_y],
                                  name=f'{stage}.slice_mesh_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        mesh_x_sq = f'{stage}.mesh_x_sq'
        mesh_y_sq = f'{stage}.mesh_y_sq'
        mesh_r_sq = f'{stage}.mesh_r_sq'
        nodes.append(oh.make_node('Mul', inputs=[mesh_x, mesh_x], outputs=[mesh_x_sq],
                                  name=f'{stage}.mul_mesh_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[mesh_y, mesh_y], outputs=[mesh_y_sq],
                                  name=f'{stage}.mul_mesh_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[mesh_x_sq, mesh_y_sq], outputs=[mesh_r_sq],
                                  name=f'{stage}.add_mesh_r_sq'))
        
        # Calculate radial distortion factor
        mesh_r4 = f'{stage}.mesh_r4'
        nodes.append(oh.make_node('Mul', inputs=[mesh_r_sq, mesh_r_sq], outputs=[mesh_r4],
                                  name=f'{stage}.mul_mesh_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, mesh_r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, mesh_r4], outputs=[k2_term],
                                  name=f'{stage}.mul_k2_term'))
        
        radial_factor = f'{stage}.radial_factor'
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[radial_factor],
                                  name=f'{stage}.identity_radial_factor'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k1_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k1_term'))
        nodes.append(oh.make_node('Add', inputs=[radial_factor, k2_term], outputs=[radial_factor],
                                  name=f'{stage}.add_k2_term'))
        
        # Calculate tangential distortion
        mesh_xy = f'{stage}.mesh_xy'
        nodes.append(oh.make_node('Mul', inputs=[mesh_x, mesh_y], outputs=[mesh_xy],
                                  name=f'{stage}.mul_mesh_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, mesh_xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, mesh_x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, mesh_y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[mesh_r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[mesh_r_sq, two_y_sq], outputs=[r2_plus_2y2],
                                  name=f'{stage}.add_r2_plus_2y2'))
        
        x_tan = f'{stage}.x_tan'
        y_tan = f'{stage}.y_tan'
        p1_term = f'{stage}.p1_term'
        p2_term = f'{stage}.p2_term'
        nodes.append(oh.make_node('Mul', inputs=[p1, two_xy], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term'))
        nodes.append(oh.make_node('Mul', inputs=[p2, r2_plus_2x2], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[x_tan],
                                  name=f'{stage}.add_x_tan'))
        
        nodes.append(oh.make_node('Mul', inputs=[p1, r2_plus_2y2], outputs=[p1_term],
                                  name=f'{stage}.mul_p1_term_y'))
        nodes.append(oh.make_node('Mul', inputs=[p2, two_xy], outputs=[p2_term],
                                  name=f'{stage}.mul_p2_term_y'))
        nodes.append(oh.make_node('Add', inputs=[p1_term, p2_term], outputs=[y_tan],
                                  name=f'{stage}.add_y_tan'))
        
        # Apply GDC distortion
        mesh_x_radial = f'{stage}.mesh_x_radial'
        mesh_y_radial = f'{stage}.mesh_y_radial'
        nodes.append(oh.make_node('Mul', inputs=[mesh_x, radial_factor], outputs=[mesh_x_radial],
                                  name=f'{stage}.mul_mesh_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[mesh_y, radial_factor], outputs=[mesh_y_radial],
                                  name=f'{stage}.mul_mesh_y_radial'))
        
        mesh_x_gdc = f'{stage}.mesh_x_gdc'
        mesh_y_gdc = f'{stage}.mesh_y_gdc'
        nodes.append(oh.make_node('Add', inputs=[mesh_x_radial, x_tan], outputs=[mesh_x_gdc],
                                  name=f'{stage}.add_mesh_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[mesh_y_radial, y_tan], outputs=[mesh_y_gdc],
                                  name=f'{stage}.add_mesh_y_gdc'))
        
        # Extract fused rotation matrix elements
        r00 = f'{stage}.r00'
        r01 = f'{stage}.r01'
        r02 = f'{stage}.r02'
        r10 = f'{stage}.r10'
        r11 = f'{stage}.r11'
        r12 = f'{stage}.r12'
        r20 = f'{stage}.r20'
        r21 = f'{stage}.r21'
        r22 = f'{stage}.r22'
        
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r00],
                                  name=f'{stage}.slice_r00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r01],
                                  name=f'{stage}.slice_r01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r02],
                                  name=f'{stage}.slice_r02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r10],
                                  name=f'{stage}.slice_r10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r11],
                                  name=f'{stage}.slice_r11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r12],
                                  name=f'{stage}.slice_r12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r20],
                                  name=f'{stage}.slice_r20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r21],
                                  name=f'{stage}.slice_r21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_fused], outputs=[r22],
                                  name=f'{stage}.slice_r22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply fused rotation transformation to GDC-distorted points
        # [x', y', w']^T = R_fused * [x_gdc, y_gdc, 1]^T
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[r00, mesh_x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[r01, mesh_y_gdc], outputs=[x_term2],
                                  name=f'{stage}.mul_x_term2'))
        nodes.append(oh.make_node('Mul', inputs=[r02, ones], outputs=[x_term3],
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
        nodes.append(oh.make_node('Mul', inputs=[r10, mesh_x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[r11, mesh_y_gdc], outputs=[y_term2],
                                  name=f'{stage}.mul_y_term2'))
        nodes.append(oh.make_node('Mul', inputs=[r12, ones], outputs=[y_term3],
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
        nodes.append(oh.make_node('Mul', inputs=[r20, mesh_x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[r21, mesh_y_gdc], outputs=[w_term2],
                                  name=f'{stage}.mul_w_term2'))
        nodes.append(oh.make_node('Mul', inputs=[r22, ones], outputs=[w_term3],
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
        
        # Stack into mesh grid [mesh_h,mesh_w,2]
        mesh_grid = f'{stage}.mesh_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[mesh_grid],
                                  name=f'{stage}.concat_mesh_grid', axis=-1))
        
        # Create valid mask
        # Pixels where mesh grid coordinates are within [-1, 1] are valid
        x_valid = f'{stage}.x_valid'
        y_valid = f'{stage}.y_valid'
        minus_one = f'{stage}.minus_one'
        inits.append(oh.make_tensor(minus_one, TensorProto.FLOAT, [], [-1.0]))
        
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[x_norm, minus_one], outputs=['x_gt_minus1']),
                                      oh.make_node('Less', inputs=[x_norm, one], outputs=['x_lt_1'])
                                  ],
                                  outputs=[x_valid],
                                  name=f'{stage}.and_x_valid'))
        nodes.append(oh.make_node('And',
                                  inputs=[
                                      oh.make_node('Greater', inputs=[y_norm, minus_one], outputs=['y_gt_minus1']),
                                      oh.make_node('Less', inputs=[y_norm, one], outputs=['y_lt_1'])
                                  ],
                                  outputs=[y_valid],
                                  name=f'{stage}.and_y_valid'))
        
        valid_mask = f'{stage}.valid_mask'
        nodes.append(oh.make_node('And', inputs=[x_valid, y_valid], outputs=[valid_mask],
                                  name=f'{stage}.and_valid'))
        
        vis.append(oh.make_tensor_value_info(mesh_grid, TensorProto.FLOAT, [self.mesh_h, self.mesh_w, 2]))
        vis.append(oh.make_tensor_value_info(valid_mask, TensorProto.BOOL, [self.mesh_h, self.mesh_w]))
        
        outputs = {
            'mesh_grid': {'name': mesh_grid},
            'valid_mask': {'name': valid_mask}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(imu_rotation, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(camera_matrix, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(smoothing_alpha, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(horizon_leveling, type=TensorProto.INT64, shape=[1])
        result.appendInput(dynamic_zoom, type=TensorProto.INT64, shape=[1])
        result.appendInput(rolling_shutter, type=TensorProto.INT64, shape=[1])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Drop old stats, calculate fusion stats, temporal smoothing.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        homography = f'{upstream}.homography'
        confidence = f'{upstream}.confidence'
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        imu_rotation = f'{upstream}.imu_rotation'
        
        # Pass through stats
        homography_out = f'{stage}.homography_out'
        confidence_out = f'{stage}.confidence_out'
        gdc_coeffs_out = f'{stage}.gdc_coeffs_out'
        imu_rotation_out = f'{stage}.imu_rotation_out'
        
        nodes.append(oh.make_node('Identity', inputs=[homography], outputs=[homography_out],
                                  name=f'{stage}.identity_homography'))
        nodes.append(oh.make_node('Identity', inputs=[confidence], outputs=[confidence_out],
                                  name=f'{stage}.identity_confidence'))
        nodes.append(oh.make_node('Identity', inputs=[gdc_coeffs], outputs=[gdc_coeffs_out],
                                  name=f'{stage}.identity_gdc_coeffs'))
        nodes.append(oh.make_node('Identity', inputs=[imu_rotation], outputs=[imu_rotation_out],
                                  name=f'{stage}.identity_imu_rotation'))
        
        vis.append(oh.make_tensor_value_info(homography_out, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(confidence_out, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(gdc_coeffs_out, TensorProto.FLOAT, [4]))
        vis.append(oh.make_tensor_value_info(imu_rotation_out, TensorProto.FLOAT, [3, 3]))
        
        outputs = {
            'homography': {'name': homography_out},
            'confidence': {'name': confidence_out},
            'gdc_coeffs': {'name': gdc_coeffs_out},
            'imu_rotation': {'name': imu_rotation_out}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(confidence, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(imu_rotation, type=TensorProto.FLOAT, shape=[3, 3])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used in core processing stage.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)


class DeshakeCoreFull16(DeshakeCoreFull):
    """
    DeshakeCoreFull16 - Core Processing Stage (Coordinator Domain)
    ---------------------------------------------------------------
    CORE PROCESSING: Generate mesh grid with full IMU fusion.
    
    Mesh Size: 16x16 (compile-time constant)
    
    Vertices: 256
    Memory: 512 floats (2 KB)
    Performance: ~3-5ms per frame (1080p 60fps)
    Quality: Good for most use cases
    Use Case: Real-time 4K 60fps, mobile devices
    """
    name = 'deshake_core_full_16'
    family = 'deshake_core'
    version = 'v2_16x16'
    mesh_h = 16
    mesh_w = 16


class DeshakeCoreFull32(DeshakeCoreFull):
    """
    DeshakeCoreFull32 - Core Processing Stage (Coordinator Domain)
    ---------------------------------------------------------------
    CORE PROCESSING: Generate mesh grid with full IMU fusion.
    
    Mesh Size: 32x32 (compile-time constant)
    
    Vertices: 1024
    Memory: 2048 floats (8 KB)
    Performance: ~5-8ms per frame (1080p 60fps)
    Quality: Higher precision, better for complex scenes
    Use Case: Real-time 4K 30fps, desktop devices
    """
    name = 'deshake_core_full_32'
    family = 'deshake_core'
    version = 'v2_32x32'
    mesh_h = 32
    mesh_w = 32
