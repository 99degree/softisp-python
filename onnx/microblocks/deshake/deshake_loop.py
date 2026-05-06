from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeLoopBase(MicroblockBase):
    """
    DeshakeLoopBase (v0)
    -------------------
    COORDINATOR: Fuse GDC and Deshake into single coordinate grid.
    
    Position: Between algo and applier, fuses sensor data and GDC coefficients.
    
    Purpose: Fuse frame-based homography with IMU data and GDC coefficients,
             then generate a single coordinate grid for one-pass correction.
    
    Needs:
        - frame_homography [3,3] : Frame-based homography matrix
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - smoothing_factor [1] : Smoothing factor (0-1)
        - iterations [1] : Number of smoothing iterations
        - image_size [2] : [height, width] for grid generation

    Provides:
        - fused_grid [h,w,2] : Fused coordinate grid for GridSample
        - homography [3,3] : Refined homography matrix
        - velocity [3] : Estimated velocity [vx, vy, vr]
        - acceleration [3] : Estimated acceleration [ax, ay, ar]

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Fuse GDC + Deshake into coordinate grid
        - build_applier: Not used (applier handles application)

    Transformation Composition:
        P_final = T_deshake * T_gdc * P_source
        
        Where:
        - T_gdc: GDC distortion transformation (lens correction)
        - T_deshake: Deshake homography transformation (motion compensation)
        - P_source: Source pixel coordinates
        - P_final: Final pixel coordinates for sampling

    Loop Body:
        1. Fuse frame homography with IMU data
        2. Update velocity and acceleration estimates
        3. Apply smoothing filter
        4. Update state for next iteration
        5. Generate fused coordinate grid (GDC + Deshake)

    Complexity: ~50-60 ONNX nodes
    Use Case: Real-time fused GDC + Deshake correction
    """
    name = 'deshake_loop_base'
    family = 'deshake_loop_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop operator for iterative homography refinement and coordinate fusion.
        
        The loop maintains state (velocity, acceleration) across iterations,
        fuses frame-based homography with IMU data, and generates a fused
        coordinate grid combining GDC and Deshake corrections.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_homography = f'{upstream}.frame_homography'
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        smoothing_factor = f'{upstream}.smoothing_factor'
        iterations = f'{upstream}.iterations'
        image_size = f'{upstream}.image_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
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
        
        # Create loop body graph
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
        
        # Add loop body nodes
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
        
        # Update velocity: v_new = v_old + a * dt
        velocity_new = f'{loop_body_name}.velocity_new'
        motion_vec = f'{loop_body_name}.motion_vec'
        loop_nodes.append(oh.make_node('Concat', inputs=[tx_combined, ty_combined, dr_combined],
                                       outputs=[motion_vec],
                                       name=f'{loop_body_name}.concat_motion'))
        loop_nodes.append(oh.make_node('Add', inputs=['state_velocity', motion_vec],
                                       outputs=[velocity_new],
                                       name=f'{loop_body_name}.add_velocity'))
        
        # Update acceleration: a_new = v_new - v_old
        acceleration_new = f'{loop_body_name}.acceleration_new'
        loop_nodes.append(oh.make_node('Sub', inputs=[velocity_new, 'state_velocity'],
                                       outputs=[acceleration_new],
                                       name=f'{loop_body_name}.sub_acceleration'))
        
        # Apply smoothing to velocity
        velocity_smoothed = f'{loop_body_name}.velocity_smoothed'
        loop_nodes.append(oh.make_node('Mul', inputs=[velocity_new, 'smoothing_factor'],
                                       outputs=[velocity_smoothed],
                                       name=f'{loop_body_name}.mul_velocity_smooth'))
        
        # Apply smoothing to acceleration
        acceleration_smoothed = f'{loop_body_name}.acceleration_smoothed'
        loop_nodes.append(oh.make_node('Mul', inputs=[acceleration_new, 'smoothing_factor'],
                                       outputs=[acceleration_smoothed],
                                       name=f'{loop_body_name}.mul_acceleration_smooth'))
        
        # Extract refined motion from smoothed velocity
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
        # H = [1  0  tx]
        #     [0  1  ty]
        #     [0  0   1]
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
        
        # Now create fused coordinate grid (GDC + Deshake)
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
        
        # Stack into coordinate grid [h,w,2]
        identity_grid = f'{stage}.identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[identity_grid],
                                  name=f'{stage}.concat_identity_grid', axis=-1))
        
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
        
        # Extract x and y from identity grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        x_sq = f'{stage}.x_sq'
        y_sq = f'{stage}.y_sq'
        r_sq = f'{stage}.r_sq'
        nodes.append(oh.make_node('Mul', inputs=[x, x], outputs=[x_sq],
                                  name=f'{stage}.mul_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[y, y], outputs=[y_sq],
                                  name=f'{stage}.mul_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[x_sq, y_sq], outputs=[r_sq],
                                  name=f'{stage}.add_r_sq'))
        
        # Calculate radial distortion factor
        # r_dist = 1 + k1 * r^2 + k2 * r^4
        r4 = f'{stage}.r4'
        nodes.append(oh.make_node('Mul', inputs=[r_sq, r_sq], outputs=[r4],
                                  name=f'{stage}.mul_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r4], outputs=[k2_term],
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
        # x_tan = 2 * p1 * x * y + p2 * (r^2 + 2 * x^2)
        # y_tan = p1 * (r^2 + 2 * y^2) + 2 * p2 * x * y
        xy = f'{stage}.xy'
        nodes.append(oh.make_node('Mul', inputs=[x, y], outputs=[xy],
                                  name=f'{stage}.mul_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_y_sq], outputs=[r2_plus_2y2],
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
        # x_gdc = x * radial_factor + x_tan
        # y_gdc = y * radial_factor + y_tan
        x_radial = f'{stage}.x_radial'
        y_radial = f'{stage}.y_radial'
        nodes.append(oh.make_node('Mul', inputs=[x, radial_factor], outputs=[x_radial],
                                  name=f'{stage}.mul_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[y, radial_factor], outputs=[y_radial],
                                  name=f'{stage}.mul_y_radial'))
        
        x_gdc = f'{stage}.x_gdc'
        y_gdc = f'{stage}.y_gdc'
        nodes.append(oh.make_node('Add', inputs=[x_radial, x_tan], outputs=[x_gdc],
                                  name=f'{stage}.add_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[y_radial, y_tan], outputs=[y_gdc],
                                  name=f'{stage}.add_y_gdc'))
        
        # Stack into GDC grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_gdc, y_gdc], outputs=[gdc_grid],
                                  name=f'{stage}.concat_gdc_grid', axis=-1))
        
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
        
        # Apply Deshake homography to GDC-distorted points
        # [x', y', w']^T = H * [x_gdc, y_gdc, 1]^T
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y_gdc], outputs=[x_term2],
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
        nodes.append(oh.make_node('Mul', inputs=[h10, x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y_gdc], outputs=[y_term2],
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
        nodes.append(oh.make_node('Mul', inputs=[h20, x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y_gdc], outputs=[w_term2],
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
        
        # Stack into fused grid [h,w,2]
        fused_grid = f'{stage}.fused_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[fused_grid],
                                  name=f'{stage}.concat_fused_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(fused_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(velocity, TensorProto.FLOAT, [3]))
        vis.append(oh.make_tensor_value_info(acceleration, TensorProto.FLOAT, [3]))
        
        outputs = {
            'fused_grid': {'name': fused_grid},
            'homography': {'name': homography},
            'velocity': {'name': velocity},
            'acceleration': {'name': acceleration}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(imu_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)


class DeshakeLoopV1(MicroblockBase):
    """
    DeshakeLoopV1 (v1)
    -----------------
    COORDINATOR: Kalman filter-based homography refinement with GDC fusion.
    
    Position: Between algo and applier, fuses sensor data and GDC coefficients.
    
    Purpose: Kalman filter for optimal homography estimation with GDC fusion.
    
    Needs:
        - frame_homography [3,3] : Frame-based homography matrix
        - gdc_coeffs [4] : GDC coefficients [k1, k2, p1, p2]
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - process_noise [1] : Process noise covariance
        - measurement_noise [1] : Measurement noise covariance
        - iterations [1] : Number of filter iterations
        - image_size [2] : [height, width] for grid generation

    Provides:
        - fused_grid [h,w,2] : Fused coordinate grid for GridSample
        - homography [3,3] : Refined homography matrix
        - covariance [6] : State covariance matrix diagonal
        - innovation [3] : Innovation (measurement - prediction)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Create Loop with Kalman filter and GDC fusion
        - build_applier: Not used (applier handles application)

    Kalman Filter:
        - Predict: x_pred = A * x_prev + B * u
        - Update: x_new = x_pred + K * (z - H * x_pred)
        - K = P * H^T * (H * P * H^T + R)^-1

    Complexity: ~70-80 ONNX nodes
    Use Case: High-precision fused GDC + Deshake correction
    """
    name = 'deshake_loop_v1'
    family = 'deshake_loop_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop with Kalman filter for optimal homography estimation and GDC fusion.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_homography = f'{upstream}.frame_homography'
        gdc_coeffs = f'{upstream}.gdc_coeffs'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        process_noise = f'{upstream}.process_noise'
        measurement_noise = f'{upstream}.measurement_noise'
        iterations = f'{upstream}.iterations'
        image_size = f'{upstream}.image_size'
        
        # Extract image dimensions
        height = f'{stage}.height'
        width = f'{stage}.width'
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[height],
                                  name=f'{stage}.slice_height',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[image_size], outputs=[width],
                                  name=f'{stage}.slice_width',
                                  starts=[1], ends=[2], axes=[0]))
        
        # Initial state: position and velocity
        state_init = f'{stage}.state_init'
        covariance_init = f'{stage}.covariance_init'
        zero = f'{stage}.zero'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [6], 
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [6], 
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[state_init],
                                  name=f'{stage}.identity_state_init'))
        nodes.append(oh.make_node('Identity', inputs=[one], outputs=[covariance_init],
                                  name=f'{stage}.identity_covariance_init'))
        
        # Create loop body graph
        loop_body_name = f'{stage}_loop_body'
        loop_body_graph = oh.make_graph(
            name=loop_body_name,
            inputs=[
                oh.make_tensor_value_info('iter', TensorProto.INT64, []),
                oh.make_tensor_value_info('cond', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state', TensorProto.FLOAT, [6]),
                oh.make_tensor_value_info('covariance', TensorProto.FLOAT, [6]),
                oh.make_tensor_value_info('homography', TensorProto.FLOAT, [3, 3]),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state_out', TensorProto.FLOAT, [6]),
                oh.make_tensor_value_info('covariance_out', TensorProto.FLOAT, [6]),
                oh.make_tensor_value_info('homography_out', TensorProto.FLOAT, [3, 3]),
            ],
            nodes=[]
        )
        
        # Add loop body nodes (simplified Kalman filter)
        loop_nodes = []
        
        # Extract translation from homography
        tx = f'{loop_body_name}.tx'
        ty = f'{loop_body_name}.ty'
        loop_nodes.append(oh.make_node('Slice', inputs=['homography'],
                                       outputs=[tx],
                                       name=f'{loop_body_name}.slice_tx',
                                       starts=[0, 2], ends=[1, 3], axes=[0, 1]))
        loop_nodes.append(oh.make_node('Slice', inputs=['homography'],
                                       outputs=[ty],
                                       name=f'{loop_body_name}.slice_ty',
                                       starts=[1, 2], ends=[2, 3], axes=[0, 1]))
        
        # Predict step: x_pred = x_prev + v * dt
        # State: [tx, ty, dr, vx, vy, vr]
        position = f'{loop_body_name}.position'
        velocity = f'{loop_body_name}.velocity'
        loop_nodes.append(oh.make_node('Slice', inputs=['state'],
                                       outputs=[position],
                                       name=f'{loop_body_name}.slice_position',
                                       starts=[0], ends=[3], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=['state'],
                                       outputs=[velocity],
                                       name=f'{loop_body_name}.slice_velocity',
                                       starts=[3], ends=[6], axes=[0]))
        
        # Predict position
        position_pred = f'{loop_body_name}.position_pred'
        loop_nodes.append(oh.make_node('Add', inputs=[position, velocity],
                                       outputs=[position_pred],
                                       name=f'{loop_body_name}.add_position_pred'))
        
        # Update step: fuse with measurement
        # Innovation = measurement - prediction
        measurement = f'{loop_body_name}.measurement'
        loop_nodes.append(oh.make_node('Concat', inputs=[tx, ty, 'imu_dr'],
                                       outputs=[measurement],
                                       name=f'{loop_body_name}.concat_measurement'))
        
        innovation = f'{loop_body_name}.innovation'
        loop_nodes.append(oh.make_node('Sub', inputs=[measurement, position_pred],
                                       outputs=[innovation],
                                       name=f'{loop_body_name}.sub_innovation'))
        
        # Kalman gain (simplified)
        kalman_gain = f'{loop_body_name}.kalman_gain'
        loop_nodes.append(oh.make_node('Mul', inputs=[innovation, 'measurement_noise'],
                                       outputs=[kalman_gain],
                                       name=f'{loop_body_name}.mul_kalman_gain'))
        
        # Update state
        position_new = f'{loop_body_name}.position_new'
        loop_nodes.append(oh.make_node('Add', inputs=[position_pred, kalman_gain],
                                       outputs=[position_new],
                                       name=f'{loop_body_name}.add_position_new'))
        
        # Update velocity
        velocity_new = f'{loop_body_name}.velocity_new'
        loop_nodes.append(oh.make_node('Add', inputs=[velocity, kalman_gain],
                                       outputs=[velocity_new],
                                       name=f'{loop_body_name}.add_velocity_new'))
        
        # Combine into new state
        state_new = f'{loop_body_name}.state_new'
        loop_nodes.append(oh.make_node('Concat', inputs=[position_new, velocity_new],
                                       outputs=[state_new],
                                       name=f'{loop_body_name}.concat_state_new'))
        
        # Update covariance (simplified)
        covariance_new = f'{loop_body_name}.covariance_new'
        loop_nodes.append(oh.make_node('Mul', inputs=['covariance', 'process_noise'],
                                       outputs=[covariance_new],
                                       name=f'{loop_body_name}.mul_covariance'))
        
        # Create refined homography matrix
        tx_out = f'{loop_body_name}.tx_out'
        ty_out = f'{loop_body_name}.ty_out'
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[tx_out],
                                       name=f'{loop_body_name}.slice_tx_out',
                                       starts=[0], ends=[1], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[ty_out],
                                       name=f'{loop_body_name}.slice_ty_out',
                                       starts=[1], ends=[2], axes=[0]))
        
        # Create homography matrix
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
                                  inputs=[iterations, 'true', state_init, covariance_init, homography_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        state = f'{stage}.state'
        covariance = f'{stage}.covariance'
        homography = f'{stage}.homography'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[state, covariance, homography],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        # Now create fused coordinate grid (GDC + Deshake)
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
        
        # Stack into coordinate grid [h,w,2]
        identity_grid = f'{stage}.identity_grid'
        nodes.append(oh.make_node('Concat', inputs=[w_norm, h_norm], outputs=[identity_grid],
                                  name=f'{stage}.concat_identity_grid', axis=-1))
        
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
        
        # Extract x and y from identity grid
        x = f'{stage}.x'
        y = f'{stage}.y'
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[x],
                                  name=f'{stage}.slice_x',
                                  starts=[0], ends=[1], axes=[-1]))
        nodes.append(oh.make_node('Slice', inputs=[identity_grid], outputs=[y],
                                  name=f'{stage}.slice_y',
                                  starts=[1], ends=[2], axes=[-1]))
        
        # Calculate radius squared
        x_sq = f'{stage}.x_sq'
        y_sq = f'{stage}.y_sq'
        r_sq = f'{stage}.r_sq'
        nodes.append(oh.make_node('Mul', inputs=[x, x], outputs=[x_sq],
                                  name=f'{stage}.mul_x_sq'))
        nodes.append(oh.make_node('Mul', inputs=[y, y], outputs=[y_sq],
                                  name=f'{stage}.mul_y_sq'))
        nodes.append(oh.make_node('Add', inputs=[x_sq, y_sq], outputs=[r_sq],
                                  name=f'{stage}.add_r_sq'))
        
        # Calculate radial distortion factor
        r4 = f'{stage}.r4'
        nodes.append(oh.make_node('Mul', inputs=[r_sq, r_sq], outputs=[r4],
                                  name=f'{stage}.mul_r4'))
        
        k1_term = f'{stage}.k1_term'
        k2_term = f'{stage}.k2_term'
        nodes.append(oh.make_node('Mul', inputs=[k1, r_sq], outputs=[k1_term],
                                  name=f'{stage}.mul_k1_term'))
        nodes.append(oh.make_node('Mul', inputs=[k2, r4], outputs=[k2_term],
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
        xy = f'{stage}.xy'
        nodes.append(oh.make_node('Mul', inputs=[x, y], outputs=[xy],
                                  name=f'{stage}.mul_xy'))
        
        two = f'{stage}.two'
        inits.append(oh.make_tensor(two, TensorProto.FLOAT, [], [2.0]))
        
        two_xy = f'{stage}.two_xy'
        nodes.append(oh.make_node('Mul', inputs=[two, xy], outputs=[two_xy],
                                  name=f'{stage}.mul_two_xy'))
        
        two_x_sq = f'{stage}.two_x_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, x_sq], outputs=[two_x_sq],
                                  name=f'{stage}.mul_two_x_sq'))
        
        two_y_sq = f'{stage}.two_y_sq'
        nodes.append(oh.make_node('Mul', inputs=[two, y_sq], outputs=[two_y_sq],
                                  name=f'{stage}.mul_two_y_sq'))
        
        r2_plus_2x2 = f'{stage}.r2_plus_2x2'
        r2_plus_2y2 = f'{stage}.r2_plus_2y2'
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_x_sq], outputs=[r2_plus_2x2],
                                  name=f'{stage}.add_r2_plus_2x2'))
        nodes.append(oh.make_node('Add', inputs=[r_sq, two_y_sq], outputs=[r2_plus_2y2],
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
        x_radial = f'{stage}.x_radial'
        y_radial = f'{stage}.y_radial'
        nodes.append(oh.make_node('Mul', inputs=[x, radial_factor], outputs=[x_radial],
                                  name=f'{stage}.mul_x_radial'))
        nodes.append(oh.make_node('Mul', inputs=[y, radial_factor], outputs=[y_radial],
                                  name=f'{stage}.mul_y_radial'))
        
        x_gdc = f'{stage}.x_gdc'
        y_gdc = f'{stage}.y_gdc'
        nodes.append(oh.make_node('Add', inputs=[x_radial, x_tan], outputs=[x_gdc],
                                  name=f'{stage}.add_x_gdc'))
        nodes.append(oh.make_node('Add', inputs=[y_radial, y_tan], outputs=[y_gdc],
                                  name=f'{stage}.add_y_gdc'))
        
        # Stack into GDC grid [h,w,2]
        gdc_grid = f'{stage}.gdc_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_gdc, y_gdc], outputs=[gdc_grid],
                                  name=f'{stage}.concat_gdc_grid', axis=-1))
        
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
        
        # Apply Deshake homography to GDC-distorted points
        ones = f'{stage}.ones'
        inits.append(oh.make_tensor(ones, TensorProto.FLOAT, [], [1.0]))
        
        x_term1 = f'{stage}.x_term1'
        x_term2 = f'{stage}.x_term2'
        x_term3 = f'{stage}.x_term3'
        nodes.append(oh.make_node('Mul', inputs=[h00, x_gdc], outputs=[x_term1],
                                  name=f'{stage}.mul_x_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h01, y_gdc], outputs=[x_term2],
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
        nodes.append(oh.make_node('Mul', inputs=[h10, x_gdc], outputs=[y_term1],
                                  name=f'{stage}.mul_y_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h11, y_gdc], outputs=[y_term2],
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
        nodes.append(oh.make_node('Mul', inputs=[h20, x_gdc], outputs=[w_term1],
                                  name=f'{stage}.mul_w_term1'))
        nodes.append(oh.make_node('Mul', inputs=[h21, y_gdc], outputs=[w_term2],
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
        
        # Stack into fused grid [h,w,2]
        fused_grid = f'{stage}.fused_grid'
        nodes.append(oh.make_node('Concat', inputs=[x_norm, y_norm], outputs=[fused_grid],
                                  name=f'{stage}.concat_fused_grid', axis=-1))
        
        vis.append(oh.make_tensor_value_info(fused_grid, TensorProto.FLOAT, ['h', 'w', 2]))
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(covariance, TensorProto.FLOAT, [6]))
        
        outputs = {
            'fused_grid': {'name': fused_grid},
            'homography': {'name': homography},
            'covariance': {'name': covariance}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(gdc_coeffs, type=TensorProto.FLOAT, shape=[4])
        result.appendInput(imu_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(process_noise, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(measurement_noise, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        result.appendInput(image_size, type=TensorProto.INT64, shape=[2])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return super().build_coordinator(stage, prev_stages)

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)