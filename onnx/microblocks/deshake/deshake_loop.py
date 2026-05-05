from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeLoopBase(MicroblockBase):
    """
    DeshakeLoopBase (v0)
    -------------------
    CONTROL LOOP: Iterative motion trace estimation and smoothing.

    Position: Between algo and applier, fuses sensor data.
    
    Purpose: Fuse frame-based motion with IMU data, apply smoothing.
    
    Needs:
        - frame_dx [1] : Frame-based horizontal translation
        - frame_dy [1] : Frame-based vertical translation
        - frame_dr [1] : Frame-based rotation angle
        - frame_dz [1] : Frame-based zoom/scale factor
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - smoothing_factor [1] : Smoothing factor (0-1)
        - iterations [1] : Number of smoothing iterations

    Provides:
        - dx [1] : Refined horizontal translation
        - dy [1] : Refined vertical translation
        - dr [1] : Refined rotation angle
        - dz [1] : Refined zoom/scale factor
        - velocity [4] : Estimated velocity [vx, vy, vr, vz]
        - acceleration [4] : Estimated acceleration [ax, ay, ar, az]

    Behavior:
        - build_algo: Not used (algo extracts motion from frames)
        - build_coordinator: Creates Loop for sensor fusion and smoothing
        - build_applier: Not used (applier handles application)

    Loop Body:
        1. Fuse frame motion with IMU data
        2. Update velocity and acceleration estimates
        3. Apply smoothing filter
        4. Update state for next iteration

    Complexity: ~40-50 ONNX nodes
    Use Case: Real-time sensor fusion and smoothing
    """
    name = 'deshake_loop_base'
    family = 'deshake_loop_base'
    version = 'v0'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts motion from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop operator for iterative motion trace estimation.
        
        The loop maintains state (velocity, acceleration) across iterations
        and fuses frame-based motion with IMU data for robust estimation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_dx = f'{upstream}.frame_dx'
        frame_dy = f'{upstream}.frame_dy'
        frame_dr = f'{upstream}.frame_dr'
        frame_dz = f'{upstream}.frame_dz'
        imu_dx = f'{upstream}.imu_dx'
        imu_dy = f'{upstream}.imu_dy'
        imu_dr = f'{upstream}.imu_dr'
        smoothing_factor = f'{upstream}.smoothing_factor'
        iterations = f'{upstream}.iterations'
        
        # Initial state: velocity and acceleration
        velocity_init = f'{stage}.velocity_init'
        acceleration_init = f'{stage}.acceleration_init'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [4], [0.0, 0.0, 0.0, 0.0]))
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
                oh.make_tensor_value_info('state_velocity', TensorProto.FLOAT, [4]),
                oh.make_tensor_value_info('state_acceleration', TensorProto.FLOAT, [4]),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('velocity_out', TensorProto.FLOAT, [4]),
                oh.make_tensor_value_info('acceleration_out', TensorProto.FLOAT, [4]),
                oh.make_tensor_value_info('dx_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dy_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dr_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dz_out', TensorProto.FLOAT, [1]),
            ],
            nodes=[]
        )
        
        # Add loop body nodes
        loop_nodes = []
        
        # Fuse frame motion with IMU data
        # Weighted combination: motion = alpha * frame + (1-alpha) * imu
        one = f'{loop_body_name}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        
        one_minus_alpha = f'{loop_body_name}.one_minus_alpha'
        loop_nodes.append(oh.make_node('Sub', inputs=['smoothing_factor', one],
                                       outputs=[one_minus_alpha],
                                       name=f'{loop_body_name}.sub_alpha'))
        
        # Fuse dx
        dx_fused = f'{loop_body_name}.dx_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=['frame_dx', 'smoothing_factor'],
                                       outputs=[dx_fused],
                                       name=f'{loop_body_name}.mul_dx_frame'))
        dx_imu = f'{loop_body_name}.dx_imu'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dx', one_minus_alpha],
                                       outputs=[dx_imu],
                                       name=f'{loop_body_name}.mul_dx_imu'))
        dx_combined = f'{loop_body_name}.dx_combined'
        loop_nodes.append(oh.make_node('Add', inputs=[dx_fused, dx_imu],
                                       outputs=[dx_combined],
                                       name=f'{loop_body_name}.add_dx'))
        
        # Fuse dy
        dy_fused = f'{loop_body_name}.dy_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=['frame_dy', 'smoothing_factor'],
                                       outputs=[dy_fused],
                                       name=f'{loop_body_name}.mul_dy_frame'))
        dy_imu = f'{loop_body_name}.dy_imu'
        loop_nodes.append(oh.make_node('Mul', inputs=['imu_dy', one_minus_alpha],
                                       outputs=[dy_imu],
                                       name=f'{loop_body_name}.mul_dy_imu'))
        dy_combined = f'{loop_body_name}.dy_combined'
        loop_nodes.append(oh.make_node('Add', inputs=[dy_fused, dy_imu],
                                       outputs=[dy_combined],
                                       name=f'{loop_body_name}.add_dy'))
        
        # Fuse dr
        dr_fused = f'{loop_body_name}.dr_fused'
        loop_nodes.append(oh.make_node('Mul', inputs=['frame_dr', 'smoothing_factor'],
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
        
        # dz (no IMU data for zoom)
        dz_combined = f'{loop_body_name}.dz_combined'
        loop_nodes.append(oh.make_node('Identity', inputs=['frame_dz'],
                                       outputs=[dz_combined],
                                       name=f'{loop_body_name}.identity_dz'))
        
        # Update velocity: v_new = v_old + a * dt
        # Simplified: v_new = v_old + motion
        velocity_new = f'{loop_body_name}.velocity_new'
        motion_vec = f'{loop_body_name}.motion_vec'
        loop_nodes.append(oh.make_node('Concat', inputs=[dx_combined, dy_combined, dr_combined, dz_combined],
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
        loop_nodes.append(oh.make_node('Mul', inputs=[velocity_new, smoothing_factor],
                                       outputs=[velocity_smoothed],
                                       name=f'{loop_body_name}.mul_velocity_smooth'))
        
        # Apply smoothing to acceleration
        acceleration_smoothed = f'{loop_body_name}.acceleration_smoothed'
        loop_nodes.append(oh.make_node('Mul', inputs=[acceleration_new, smoothing_factor],
                                       outputs=[acceleration_smoothed],
                                       name=f'{loop_body_name}.mul_acceleration_smooth'))
        
        # Extract refined motion from smoothed velocity
        dx_out = f'{loop_body_name}.dx_out'
        dy_out = f'{loop_body_name}.dy_out'
        dr_out = f'{loop_body_name}.dr_out'
        dz_out = f'{loop_body_name}.dz_out'
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[dx_out],
                                       name=f'{loop_body_name}.slice_dx',
                                       starts=[0], ends=[1], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[dy_out],
                                       name=f'{loop_body_name}.slice_dy',
                                       starts=[1], ends=[2], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[dr_out],
                                       name=f'{loop_body_name}.slice_dr',
                                       starts=[2], ends=[3], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[velocity_smoothed],
                                       outputs=[dz_out],
                                       name=f'{loop_body_name}.slice_dz',
                                       starts=[3], ends=[4], axes=[0]))
        
        # Update loop body graph
        loop_body_graph.nodes.extend(loop_nodes)
        
        # Create Loop node
        loop_output = f'{stage}.loop_output'
        nodes.append(oh.make_node('Loop', 
                                  inputs=[iterations, 'true', velocity_init, acceleration_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        velocity = f'{stage}.velocity'
        acceleration = f'{stage}.acceleration'
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        dr = f'{stage}.dr'
        dz = f'{stage}.dz'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[velocity, acceleration, dx, dy, dr, dz],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        vis.append(oh.make_tensor_value_info(dx, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dy, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dr, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dz, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(velocity, TensorProto.FLOAT, [4]))
        vis.append(oh.make_tensor_value_info(acceleration, TensorProto.FLOAT, [4]))
        
        outputs = {
            'dx': {'name': dx},
            'dy': {'name': dy},
            'dr': {'name': dr},
            'dz': {'name': dz},
            'velocity': {'name': velocity},
            'acceleration': {'name': acceleration}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dz, type=TensorProto.FLOAT, shape=[1])
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
    CONTROL LOOP: Kalman filter-based motion estimation.

    Position: Between algo and applier, fuses sensor data.
    
    Purpose: Kalman filter for optimal motion estimation.
    
    Needs:
        - frame_dx [1] : Frame-based horizontal translation
        - frame_dy [1] : Frame-based vertical translation
        - frame_dr [1] : Frame-based rotation angle
        - frame_dz [1] : Frame-based zoom/scale factor
        - imu_dx [1] : IMU horizontal acceleration
        - imu_dy [1] : IMU vertical acceleration
        - imu_dr [1] : IMU angular velocity
        - process_noise [1] : Process noise covariance
        - measurement_noise [1] : Measurement noise covariance
        - iterations [1] : Number of filter iterations

    Provides:
        - dx [1] : Refined horizontal translation
        - dy [1] : Refined vertical translation
        - dr [1] : Refined rotation angle
        - dz [1] : Refined zoom/scale factor
        - covariance [8] : State covariance matrix diagonal
        - innovation [4] : Innovation (measurement - prediction)

    Behavior:
        - build_algo: Not used (algo extracts motion from frames)
        - build_coordinator: Creates Loop with Kalman filter
        - build_applier: Not used (applier handles application)

    Kalman Filter:
        - Predict: x_pred = A * x_prev + B * u
        - Update: x_new = x_pred + K * (z - H * x_pred)
        - K = P * H^T * (H * P * H^T + R)^-1

    Complexity: ~60-80 ONNX nodes
    Use Case: High-precision motion estimation
    """
    name = 'deshake_loop_v1'
    family = 'deshake_loop_v1'
    version = 'v1'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Not used - algo extracts motion from frames.
        """
        return BuildResult({}, [], [], [])

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Create Loop with Kalman filter for optimal motion estimation.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        frame_dx = f'{upstream}.frame_dx'
        frame_dy = f'{upstream}.frame_dy'
        frame_dr = f'{upstream}.frame_dr'
        frame_dz = f'{upstream}.frame_dz'
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
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [8], 
                                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [8], 
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
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
                oh.make_tensor_value_info('state', TensorProto.FLOAT, [8]),
                oh.make_tensor_value_info('covariance', TensorProto.FLOAT, [8]),
            ],
            outputs=[
                oh.make_tensor_value_info('cond_out', TensorProto.BOOL, []),
                oh.make_tensor_value_info('state_out', TensorProto.FLOAT, [8]),
                oh.make_tensor_value_info('covariance_out', TensorProto.FLOAT, [8]),
                oh.make_tensor_value_info('dx_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dy_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dr_out', TensorProto.FLOAT, [1]),
                oh.make_tensor_value_info('dz_out', TensorProto.FLOAT, [1]),
            ],
            nodes=[]
        )
        
        # Add loop body nodes (simplified Kalman filter)
        loop_nodes = []
        
        # Predict step: x_pred = x_prev + v * dt
        # State: [x, y, r, z, vx, vy, vr, vz]
        position = f'{loop_body_name}.position'
        velocity = f'{loop_body_name}.velocity'
        loop_nodes.append(oh.make_node('Slice', inputs=['state'],
                                       outputs=[position],
                                       name=f'{loop_body_name}.slice_position',
                                       starts=[0], ends=[4], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=['state'],
                                       outputs=[velocity],
                                       name=f'{loop_body_name}.slice_velocity',
                                       starts=[4], ends=[8], axes=[0]))
        
        # Predict position
        position_pred = f'{loop_body_name}.position_pred'
        loop_nodes.append(oh.make_node('Add', inputs=[position, velocity],
                                       outputs=[position_pred],
                                       name=f'{loop_body_name}.add_position_pred'))
        
        # Update step: fuse with measurement
        # Innovation = measurement - prediction
        measurement = f'{loop_body_name}.measurement'
        loop_nodes.append(oh.make_node('Concat', inputs=['frame_dx', 'frame_dy', 'frame_dr', 'frame_dz'],
                                       outputs=[measurement],
                                       name=f'{loop_body_name}.concat_measurement'))
        
        innovation = f'{loop_body_name}.innovation'
        loop_nodes.append(oh.make_node('Sub', inputs=[measurement, position_pred],
                                       outputs=[innovation],
                                       name=f'{loop_body_name}.sub_innovation'))
        
        # Kalman gain (simplified)
        kalman_gain = f'{loop_body_name}.kalman_gain'
        loop_nodes.append(oh.make_node('Mul', inputs=[innovation, measurement_noise],
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
        loop_nodes.append(oh.make_node('Mul', inputs=['covariance', process_noise],
                                       outputs=[covariance_new],
                                       name=f'{loop_body_name}.mul_covariance'))
        
        # Extract outputs
        dx_out = f'{loop_body_name}.dx_out'
        dy_out = f'{loop_body_name}.dy_out'
        dr_out = f'{loop_body_name}.dr_out'
        dz_out = f'{loop_body_name}.dz_out'
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[dx_out],
                                       name=f'{loop_body_name}.slice_dx',
                                       starts=[0], ends=[1], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[dy_out],
                                       name=f'{loop_body_name}.slice_dy',
                                       starts=[1], ends=[2], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[dr_out],
                                       name=f'{loop_body_name}.slice_dr',
                                       starts=[2], ends=[3], axes=[0]))
        loop_nodes.append(oh.make_node('Slice', inputs=[position_new],
                                       outputs=[dz_out],
                                       name=f'{loop_body_name}.slice_dz',
                                       starts=[3], ends=[4], axes=[0]))
        
        # Update loop body graph
        loop_body_graph.nodes.extend(loop_nodes)
        
        # Create Loop node
        loop_output = f'{stage}.loop_output'
        nodes.append(oh.make_node('Loop', 
                                  inputs=[iterations, 'true', state_init, covariance_init],
                                  outputs=[loop_output],
                                  name=f'{stage}.loop',
                                  body=loop_body_graph))
        
        # Extract outputs from loop
        state = f'{stage}.state'
        covariance = f'{stage}.covariance'
        dx = f'{stage}.dx'
        dy = f'{stage}.dy'
        dr = f'{stage}.dr'
        dz = f'{stage}.dz'
        
        nodes.append(oh.make_node('Split', inputs=[loop_output],
                                  outputs=[state, covariance, dx, dy, dr, dz],
                                  name=f'{stage}.split_loop_output',
                                  axis=0))
        
        vis.append(oh.make_tensor_value_info(dx, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dy, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dr, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(dz, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(covariance, TensorProto.FLOAT, [8]))
        
        outputs = {
            'dx': {'name': dx},
            'dy': {'name': dy},
            'dr': {'name': dr},
            'dz': {'name': dz},
            'covariance': {'name': covariance}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(frame_dx, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dy, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dr, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(frame_dz, type=TensorProto.FLOAT, shape=[1])
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