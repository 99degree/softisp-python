from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeLoopBase(MicroblockBase):
    """
    DeshakeLoopBase (v0)
    -------------------
    CONTROL LOOP: Iterative homography refinement with sensor fusion.

    Position: Between algo and applier, fuses sensor data.
    
    Purpose: Fuse frame-based homography with IMU data, apply smoothing.
    
    Needs:
        - frame_homography [3,3] : Frame-based homography matrix
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - smoothing_factor [1] : Smoothing factor (0-1)
        - iterations [1] : Number of smoothing iterations

    Provides:
        - homography [3,3] : Refined homography matrix
        - velocity [3] : Estimated velocity [vx, vy, vr]
        - acceleration [3] : Estimated acceleration [ax, ay, ar]

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Creates Loop for sensor fusion and smoothing
        - build_applier: Not used (applier handles application)

    Loop Body:
        1. Fuse frame homography with IMU data
        2. Update velocity and acceleration estimates
        3. Apply smoothing filter
        4. Update state for next iteration

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]
        
        Where:
        - h00, h01, h10, h11: Rotation and scaling
        - h02, h12: Translation (tx, ty)
        - h20, h21: Perspective (shear)
        - h22: Scale (usually 1)

    Complexity: ~40-50 ONNX nodes
    Use Case: Real-time sensor fusion and smoothing
    """
    name = 'deshake_loop_base'
    family = 'deshake_loop_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop operator for iterative homography refinement.
        
        The loop maintains state (velocity, acceleration) across iterations
        and fuses frame-based homography with IMU data for robust estimation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_homography = f'{upstream}.frame_homography'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        smoothing_factor = f'{upstream}.smoothing_factor'
        iterations = f'{upstream}.iterations'
        
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
        # Simplified: v_new = v_old + motion
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
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(velocity, TensorProto.FLOAT, [3]))
        vis.append(oh.make_tensor_value_info(acceleration, TensorProto.FLOAT, [3]))
        
        outputs = {
            'homography': {'name': homography},
            'velocity': {'name': velocity},
            'acceleration': {'name': acceleration}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(imu_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)


class DeshakeLoopV1(MicroblockBase):
    """
    DeshakeLoopV1 (v1)
    -----------------
    CONTROL LOOP: Kalman filter-based homography refinement.

    Position: Between algo and applier, fuses sensor data.
    
    Purpose: Kalman filter for optimal homography estimation.
    
    Needs:
        - frame_homography [3,3] : Frame-based homography matrix
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - process_noise [1] : Process noise covariance
        - measurement_noise [1] : Measurement noise covariance
        - iterations [1] : Number of filter iterations

    Provides:
        - homography [3,3] : Refined homography matrix
        - covariance [9] : State covariance matrix diagonal
        - innovation [3] : Innovation (measurement - prediction)

    Behavior:
        - build_algo: Not used (algo extracts homography from frames)
        - build_coordinator: Creates Loop with Kalman filter
        - build_applier: Not used (applier handles application)

    Kalman Filter:
        - Predict: x_pred = A * x_prev + B * u
        - Update: x_new = x_pred + K * (z - H * x_pred)
        - K = P * H^T * (H * P * H^T + R)^-1

    Homography Matrix Format (OpenCV):
        H = [h00 h01 h02]
            [h10 h11 h12]
            [h20 h21 h22]

    Complexity: ~60-80 ONNX nodes
    Use Case: High-precision motion estimation
    """
    name = 'deshake_loop_v1'
    family = 'deshake_loop_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts homography from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop with Kalman filter for optimal homography estimation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_homography = f'{upstream}.frame_homography'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        process_noise = f'{upstream}.process_noise'
        measurement_noise = f'{upstream}.measurement_noise'
        iterations = f'{upstream}.iterations'
        
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
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(covariance, TensorProto.FLOAT, [6]))
        
        outputs = {
            'homography': {'name': homography},
            'covariance': {'name': covariance}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_homography, type=TensorProto.FLOAT, shape=[3, 3])
        result.appendInput(imu_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(imu_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(process_noise, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(measurement_noise, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(iterations, type=TensorProto.INT64, shape=[1])
        return result

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_coordinator(stage, prev_stages)