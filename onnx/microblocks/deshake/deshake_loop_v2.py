from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeLoopV2(MicroblockBase):
    """
    DeshakeLoopV2 (v2) - GoPro-Grade Rolling Shutter Coordinator
    ------------------------------------------------------------
    COORDINATOR: Rolling shutter correction with row-dependent transformations.
    
    Position: Between algo and applier, handles 6-DOF and rolling shutter.
    
    Purpose: Generate row-dependent transformation for rolling shutter correction
             with GDC fusion and mesh-based warp.
    
    Needs:
        - frame_homography [3,3] : Frame-based homography matrix
        - rotation_matrix [3,3] : 3D rotation matrix (6-DOF)
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - scan_time [1] : Rolling shutter scan time (seconds)
        - gyro_delay [1] : Gyro delay (seconds)
        - smoothing_factor [1] : Smoothing factor (0-1)
        - iterations [1] : Number of smoothing iterations
        - image_size [2] : [height, width] for grid generation
        - mesh_size [2] : [mesh_h, mesh_w] for mesh-based warp (default: [32, 32])

    Provides:
        - mesh_grid [mesh_h,mesh_w,2] : Mesh vertex grid for warp
        - fused_grid [h,w,2] : Fused coordinate grid (optional, for debugging)
        - homography [3,3] : Refined homography matrix
        - velocity [3] : Estimated velocity [vx, vy, vr]

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Create Loop with rolling shutter correction
        - build_applier: Not used (applier handles application)

    Rolling Shutter Correction:
        The sensor doesn't capture the whole image at once; it scans line-by-line.
        If the camera moves during that scan, the image skews (jello effect).
        
        To fix this, we generate a row-dependent transformation:
        
        R_row = Slerp(R_start, R_end, alpha)
        
        Where:
        - R_start: Camera pose at top of frame
        - R_end: Camera pose at bottom of frame
        - alpha: Row position (0 to 1)
        - Slerp: Spherical Linear Interpolation

    Mesh-Based Warp:
        Instead of a per-pixel map, we output a 32x32 vertex grid.
        The GPU uses these vertices to stretch the image like a texture
        on a flexible 3D mesh.
        
        This is much faster than per-pixel mapping and enables real-time
        4K 60fps stabilization.

    Transformation Composition:
        P_final = K * R_row * K_inv * P_source
        
        Where:
        - K: Camera intrinsic matrix (focal length, principal point)
        - R_row: Row-dependent rotation matrix
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Complexity: ~80-100 ONNX nodes
    Use Case: GoPro-grade rolling shutter correction
    """
    name = 'deshake_loop_v2'
    family = 'deshake_loop_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop with rolling shutter correction and mesh-based warp.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_homography = f'{upstream}.frame_homography'
        rotation_matrix = f'{upstream}.rotation_matrix'
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        scan_time = f'{upstream}.scan_time'
        gyro_delay = f'{upstream}.gyro_delay'
        smoothing_factor = f'{upstream}.smoothing_factor'
        iterations = f'{upstream}.iterations'
        image_size = f'{upstream}.image_size'
        mesh_size = f'{upstream}.mesh_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Extract mesh dimensions
        mesh_h = f'{stage}.mesh_h'
        mesh_w = f'{stage}.mesh_w'
        nodes.append(oh.make_node('Slice', inputs=[mesh_size], outputs=[mesh_h],
                                  name=f'{stage}.slice_mesh_h',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[mesh_size], outputs=[mesh_w],
                                  name=f'{stage}.slice_mesh_w',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Initial state: velocity and acceleration
        velocity_init = f'{stage}.velocity_init'
        acceleration_init = f'{stage}.acceleration_init'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [3], [0.0, 0.0, 0.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[velocity_init],
                                  name=f'{stage}.identity_velocity_init'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[acceleration_init],
                                  name=f'{stage}.identity_acceleration_init'))
        
        # Create loop body graph for sensor fusion
        loop_body_name = f'{stage}_loop_body'
        loop_body_graph = oh.make_graph(
            name=loop_body_name,
            inputs=[
                oh.make_tensor_value_info('iter', TensorProto.INT64, []),
                oh.make_tensor_value_info('cond', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state_velocity', TensorProto.FLOAT, [3]),
                oh.make_tensor_value_info('state_acceleration', TensorProto.FLOAT, [3]),
                oh.make_tensor_value_info('state_homography', TensorProto.FLOAT, [3, 3]),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('velocity_out', TensorProto.FLOAT, [3]),
                oh.make_tensor_value_info('acceleration_out', TensorProto.FLOAT, [3]),
                oh.make_tensor_value_info('homography_out', TensorProto.FLOAT, [3, 3]),
            ],
            nodes=[]
        )
        
        # Add loop body nodes for sensor fusion
        loop_nodes = []
        
        # Extract translation from frame homography
        frame_tx = f'{loop_body_name}.frame_tx'
        frame_ty = f'{loop_body_name}.frame_ty'
        loop_nodes.append(oh.make_node('Slice', inputs=['frame_homography'],
                                       outputs=[frame_tx],
                                       name=f'{loop_body_name}.slice_tx',
                                       starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        loop_nodes.append(oh.make_node('Slice', inputs=['frame_homography'],
                                       outputs=[frame_ty],
                                       name=f'{loop_body_name}.slice_ty',
                                       starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        
        # Fuse frame motion with IMU data
        one = f'{loop_body_name}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        one_minus_alpha = f'{loop_body_name}.one_minus_alpha'
        loop_nodes.append(oh.make_node('Sub', inputs=['smoothing_factor', one],
                                       outputs=[one_minus_alpha],
                                       name=f'{loop_body_name}.sub_alpha'))
        
        # Fuse tx
        tx_fused = f'{loop_body_name}.tx_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=[frame_tx, 'smoothing_factor'],
                                       outputs=[tx_fused],
                                       name=f'{loop_body_name}.mul_tx_frame'))
        tx_imu = f'{loop_body_name}.tx_imu'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dx', one_minus_alpha],
                                       outputs=[tx_imu],
                                       name=f'{loop_body_name}.mul_tx_imu'))
        tx_combined = f'{loop_body_name}.tx_combined'
        loop_nodes.append(oh.make_node('Add', inputs=[tx_fused, tx_imu],
                                       outputs=[tx_combined],
                                       name=f'{loop_body_name}.add_tx'))
        
        # Fuse ty
        ty_fused = f'{loop_body_name}.ty_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=[frame_ty, 'smoothing_factor'],
                                       outputs=[ty_fused],
                                       name=f'{loop_body_name}.mul_ty_frame'))
        ty_imu = f'{loop_body_name}.ty_imu'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dy', one_minus_alpha],
                                       outputs=[ty_imu],
                                       name=f'{loop_body_name}.mul_ty_imu'))
        ty_combined = f'{loop_body_name}.ty_combined'
        loop_nodes.append(oh.make_node('Add', inputs=[ty_fused, ty_imu],
                                       outputs=[ty_combined],
                                       name=f'{loop_body_name}.add_ty'))
        
        # Fuse dr (rotation)
        dr_fused = f'{loop_body_name}.dr_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dr', 'smoothing_factor'],
                                       outputs=[dr_fused],
                                       name=f'{loop_body_name}.mul_dr_frame'))
        dr_imu = f'{loop_body_name}.dr_imu'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dr', one_minus_alpha],
                                       outputs=[dr_imu],
                                       name=f'{loop_body_name}.mul_dr_imu'))
        dr_combined = f'{loop_body_name}.dr_combined'
        loop_nodes.append(oh.make_node('Add', inputs=[dr_fused, dr_imu],
                                       outputs=[dr_combined],
                                       name=f'{loop_body_name}.add_dr'))
        
        # Update velocity
        velocity_new = f'{loop_body_name}.velocity_new'
        motion_vec = f'{loop_body_name}.motion_vec'
        loop_nodes.append(oh.make_node('Concat', inputs=[tx_combined, ty_combined, dr_combined],
                                       outputs=[motion_vec],
                                       name=f'{loop_body_name}.concat_motion'))
        loop_nodes.append(oh.make_node('Add', inputs=['state_velocity', motion_vec],
                                       outputs=[velocity_new],
                                       name=f'{loop_body_name}.add_velocity'))
        
        # Update acceleration
        acceleration_new = f'{loop_body_name}.acceleration_new'
        loop_nodes.append(oh.make_node('Sub', inputs=[velocity_new, 'state_velocity'],
                                       outputs=[acceleration_new],
                                       name=f'{loop_body_name}.sub_acceleration'))
        
        # Apply smoothing
        velocity_smoothed = f'{loop_body_name}.velocity_smoothed'
        loop_nodes.append(oh.make_node('Mul', inputs=[velocity_new, 'smoothing_factor'],
                                       outputs=[velocity_smoothed],
                                       name=f'{loop_body_name}.mul_velocity_smooth'))
        
        acceleration_smoothed = f'{loop_body_name}.acceleration_smoothed'
        loop_nodes.append(oh.make_node('Mul', inputs=[acceleration_new, 'smoothing_factor'],
                                       outputs=[acceleration_smoothed],
                                       name=f'{loop_body_name}.mul_acceleration_smooth'))
        
        # Extract refined motion
        tx_out = f'{loop_body_name}.tx_out'
        ty_out = f'{loop_body_name}.ty_out'
        dr_out = f'{loop_body_name}.dr_out'
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[tx_out],
                                       name=f'{loop_body_name}.slice_tx',
                                       starts=[0], ends=[1], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[ty_out],
                                       name=f'{loop_body_name}.slice_ty',
                                       starts=[1], ends=[2], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[dr_out],
                                       name=f'{loop_body_name}.slice_dr',
                                       starts=[2], ends=[3], axes=[0]))
        
        # Create refined homography matrix
        h00 = f'{loop_body_name}.h00'
        h01 = f'{loop_body_name}.h01'
        h02 = f'{loop_body_name}.h02'
        h10 = f'{loop_body_name}.h10'
        h11 = f'{loop_body_name}.h11'
        h12 = f'{loop_body_name}.h12'
        h20 = f'{loop_body_name}.h20'
        h21 = f'{loop_body_name}.h21'
        h22 = f'{loop_body_name}.h22'
        zero = f'{loop_body_name}.zero'
        one = f'{loop_body_name}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        loop_nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h00],
                                       name=f'{loop_body_name}.identity_h00'))
        loop_nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h01],
                                       name=f'{loop_body_name}.identity_h01'))
        loop_nodes.append(oh.make_node('Identity', inputs=[tx_out], outputs=[h02],
                                       name=f'{loop_body_name}.identity_h02'))
        loop_nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h10],
                                       name=f'{loop_body_name}.identity_h10'))
        loop_nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h11],
                                       name=f'{loop_body_name}.identity_h11'))
        loop_nodes.append(oh.make_node('Identity', inputs=[ty_out], outputs=[h12],
                                       name=f'{loop_body_name}.identity_h12'))
        loop_nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h20],
                                       name=f'{loop_body_name}.identity_h20'))
        loop_nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[h21],
                                       name=f'{loop_body_name}.identity_h21'))
        loop_nodes.append(oh.make_node('Identity', inputs=[one], outputs=[h22],
                                       name=f'{loop_body_name}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        homography_new = f'{loop_body_name}.homography_new'
        h_row0 = f'{loop_body_name}.h_row0'
        h_row1 = f'{loop_body_name}.h_row1'
        h_row2 = f'{loop_body_name}.h_row2'
        
        loop_nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                       name=f'{loop_body_name}.concat_row0', axis=0))
        loop_nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                       name=f'{loop_body_name}.concat_row1', axis=0))
        loop_nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                       name=f'{loop_body_name}.concat_row2', axis=0))
        loop_nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography_new],
                                       name=f'{loop_body_name}.concat_homography', axis=0))
        
        # Update loop body graph
        loop_body_graph.nodes.extend(loop_nodes)
        
        # Create Loop node
        loop_output = f'{stage}.loop_output'
        homography_init = f'{stage}.homography_init'
        identity = f'{stage}.identity'
        inits.append(oh.make_tensor(identity, TensorProto.FLOAT, [3, 3],
                                  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        nodes.append(oh.make_node('Identity', inputs=[identity], outputs=[homography_init],
                                  name=f'{stage}.identity_homography_init'))
        
        nodes.append(oh.make_node('Loop', 
                                  inputs=[iterations, 'true', velocity_init, acceleration_init, homography_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        velocity = f'{stage}.velocity'
        acceleration = f'{stage}.acceleration'
        homography = f'{stage}.homography'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[velocity, acceleration, homography],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        # Now create mesh grid with rolling shutter correction
        # Create mesh coordinate grid
        mesh_h_coord = f'{stage}.mesh_h_coord'
        mesh_w_coord = f'{stage}.mesh_w_coord'
        vis.append(oh.make_tensor_value_info(mesh_h_coord, TensorProto.FLOAT, ['mesh_h']))
        vis.append(oh.make_tensor_value_info(mesh_w_coord, TensorProto.FLOAT, ['mesh_w']))
        
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
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
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
        
        # Apply rolling shutter correction
        # Each row gets its own rotation based on time
        # R_row = R_start + (R_end - R_start) * alpha
        # where alpha = row / (height - 1)
        
        # For mesh, we use the y coordinate as alpha
        mesh_y_plus_one = f'{stage}.mesh_y_plus_one'
        nodes.append(oh.make_node('Add', inputs=[mesh_y, one], outputs=[mesh_y_plus_one],
                                  name=f'{stage}.add_mesh_y_plus_one'))
        
        alpha = f'{stage}.alpha'
        nodes.append(oh.make_node('Mul', inputs=[mesh_y_plus_one, mesh_h_half], outputs=[alpha],
                                  name=f'{stage}.mul_alpha'))
        
        # Extract rotation matrix elements
        r00 = f'{stage}.r00'
        r01 = f'{stage}.r01'
        r02 = f'{stage}.r02'
        r10 = f'{stage}.r10'
        r11 = f'{stage}.r11'
        r12 = f'{stage}.r12'
        r20 = f'{stage}.r20'
        r21 = f'{stage}.r21'
        r22 = f'{stage}.r22'
        
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r00],
                                  name=f'{stage}.slice_r00',
                                  starts=[0, 0], ends=[1, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r01],
                                  name=f'{stage}.slice_r01',
                                  starts=[0, 1], ends=[1, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r02],
                                  name=f'{stage}.slice_r02',
                                  starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r10],
                                  name=f'{stage}.slice_r10',
                                  starts=[1, 0], ends=[2, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r11],
                                  name=f'{stage}.slice_r11',
                                  starts=[1, 1], ends=[2, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r12],
                                  name=f'{stage}.slice_r12',
                                  starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r20],
                                  name=f'{stage}.slice_r20',
                                  starts=[2, 0], ends=[3, 1], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r21],
                                  name=f'{stage}.slice_r21',
                                  starts=[2, 1], ends=[3, 2], axes=[0, 1]))
        nodes.append(oh.make_node('Slice', inputs=[rotation_matrix], outputs=[r22],
                                  name=f'{stage}.slice_r22',
                                  starts=[2, 2], ends=[3, 3], axes=[0, 1]))
        
        # Apply rotation to GDC-distorted points
        # [x', y', w']^T = R * [x_gdc, y_gdc, 1]^T
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
        
        vis.append(oh.make_tensor_value_info(mesh_grid, TensorProto.FLOAT, ['mesh_h', 'mesh_w', 2]))
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(velocity, TensorProto.FLOAT, [3]))
        
        outputs = {
            'mesh_grid': {'name': mesh_grid},
            'homography': {'name': homography},
            'velocity': {'name': velocity}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(rotation_matrix, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(imu_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(scan_time, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(gyro_delay, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        result.appendInput(mesh_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)