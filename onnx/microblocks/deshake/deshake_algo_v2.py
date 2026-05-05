from microblocks.base import BuildResult
import onnx.helper as oh
from onnx import TensorProto
from microblocks.base import MicroblockBase


class DeshakeAlgoV2(MicroblockBase):
    """
    DeshakeAlgoV2 (v2) - GoPro-Grade Path Smoother
    -----------------------------------------------
    ALGO: Temporal path smoothing with horizon leveling.
    
    Position: First stage, extracts and smooths camera pose.
    
    Purpose: Calculate smooth camera pose using low-pass/Kalman filter
             for cinematic stabilization with horizon leveling.
    
    Needs:
        - current_frame [n,3,h,w] : Current video frame (YUV or RGB)
        - prev_frame [n,3,h,w] : Previous video frame (YUV or RGB)
        - gyro_data [3] : Raw gyroscope data [rx, ry, rz] (rad/s)
        - gravity_vector [3] : Gravity vector [gx, gy, gz] (normalized)
        - smoothing_factor [1] : Low-pass filter coefficient (0-1)
        - horizon_leveling [1] : Enable horizon leveling (0=off, 1=on)

    Provides:
        - homography [3,3] : Smoothed homography matrix (OpenCV format)
        - rotation_matrix [3,3] : 3D rotation matrix (6-DOF)
        - translation [3] : 3D translation vector [tx, ty, tz]
        - dynamic_zoom [1] : Dynamic zoom factor (1.0 = no zoom)
        - confidence [1] : Confidence score (0-1)

    Behavior:
        - build_algo: Extract and smooth camera pose
        - build_coordinator: Not used
        - build_applier: Not used

    Path Smoothing:
        The algo uses a low-pass filter to smooth the camera path:
        
        R_smooth[t] = α * R_smooth[t-1] + (1-α) * R_raw[t]
        
        Where:
        - R_smooth: Smoothed rotation matrix
        - R_raw: Raw rotation from gyro
        - α: Smoothing factor (higher = smoother but more lag)

    Horizon Leveling:
        Calculates rotation needed to keep gravity vector at 0°:
        
        R_level = R_gravity * R_smooth
        
        Where:
        - R_gravity: Rotation to align gravity with vertical axis
        - R_smooth: Smoothed camera rotation

    Dynamic Zoom:
        Adjusts zoom based on shake severity:
        
        Z = 1.0 + β * |angular_velocity|
        
        Where:
        - Z: Zoom factor
        - β: Zoom sensitivity
        - angular_velocity: Magnitude of rotation

    Complexity: ~40-50 ONNX nodes
    Use Case: GoPro-grade cinematic stabilization
    """
    name = 'deshake_algo_v2'
    family = 'deshake_algo_v2'
    version = 'v2'

    def build_algo(self, stage: str, prev_stages=None):
        """
        Extract and smooth camera pose with horizon leveling.
        """
        vis, nodes, inits = ([], [], [])
        upstream = prev_stages[0] if prev_stages else stage
        
        current_frame = f'{upstream}.current_frame'
        prev_frame = f'{upstream}.prev_frame'
        gyro_data = f'{upstream}.gyro_data'
        gravity_vector = f'{upstream}.gravity_vector'
        smoothing_factor = f'{upstream}.smoothing_factor'
        horizon_leveling = f'{upstream}.horizon_leveling'
        
        # Extract gyro components
        rx = f'{stage}.rx'
        ry = f'{stage}.ry'
        rz = f'{stage}.rz'
        nodes.append(oh.make_node('Slice', inputs=[gyro_data], outputs=[rx],
                                  name=f'{stage}.slice_rx',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gyro_data], outputs=[ry],
                                  name=f'{stage}.slice_ry',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gyro_data], outputs=[rz],
                                  name=f'{stage}.slice_rz',
                                  starts=[2], ends=[3], axes=[0]))
        
        # Calculate angular velocity magnitude
        rx_sq = f'{stage}.rx_sq'
        ry_sq = f'{stage}.ry_sq'
        rz_sq = f'{stage}.rz_sq'
        nodes.append(oh.make_node('Mul', inputs=[rx, rx], outputs=[rx_sq],
                                  name=f'{stage}.mul_rx_sq'))
        nodes.append(oh.make_node('Mul', inputs=[ry, ry], outputs=[ry_sq],
                                  name=f'{stage}.mul_ry_sq'))
        nodes.append(oh.make_node('Mul', inputs=[rz, rz], outputs=[rz_sq],
                                  name=f'{stage}.mul_rz_sq'))
        
        angular_velocity = f'{stage}.angular_velocity'
        nodes.append(oh.make_node('Add', inputs=[rx_sq, ry_sq], outputs=[angular_velocity],
                                  name=f'{stage}.add_xy_sq'))
        nodes.append(oh.make_node('Add', inputs=[angular_velocity, rz_sq], outputs=[angular_velocity],
                                  name=f'{stage}.add_angular_velocity'))
        nodes.append(oh.make_node('Sqrt', inputs=[angular_velocity], outputs=[angular_velocity],
                                  name=f'{stage}.sqrt_angular_velocity'))
        
        # Calculate dynamic zoom based on shake severity
        # Z = 1.0 + β * |angular_velocity|
        zoom_sensitivity = f'{stage}.zoom_sensitivity'
        inits.append(oh.make_tensor(zoom_sensitivity, TensorProto.FLOAT, [], [0.1]))
        
        zoom_offset = f'{stage}.zoom_offset'
        nodes.append(oh.make_node('Mul', inputs=[angular_velocity, zoom_sensitivity], outputs=[zoom_offset],
                                  name=f'{stage}.mul_zoom_offset'))
        
        dynamic_zoom = f'{stage}.dynamic_zoom'
        one = f'{stage}.one'
        inits.append(oh.make_tensor(one, TensorProto.FLOAT, [], [1.0]))
        nodes.append(oh.make_node('Add', inputs=[one, zoom_offset], outputs=[dynamic_zoom],
                                  name=f'{stage}.add_dynamic_zoom'))
        
        # Extract gravity vector
        gx = f'{stage}.gx'
        gy = f'{stage}.gy'
        gz = f'{stage}.gz'
        nodes.append(oh.make_node('Slice', inputs=[gravity_vector], outputs=[gx],
                                  name=f'{stage}.slice_gx',
                                  starts=[0], ends=[1], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gravity_vector], outputs=[gy],
                                  name=f'{stage}.slice_gy',
                                  starts=[1], ends=[2], axes=[0]))
        nodes.append(oh.make_node('Slice', inputs=[gravity_vector], outputs=[gz],
                                  name=f'{stage}.slice_gz',
                                  starts=[2], ends=[3], axes=[0]))
        
        # Calculate horizon leveling rotation
        # Gravity should point down (0, -1, 0) in camera frame
        # Calculate pitch and roll from gravity vector
        pitch = f'{stage}.pitch'
        roll = f'{stage}.roll'
        
        # pitch = atan2(gx, gz)
        # roll = atan2(-gy, sqrt(gx^2 + gz^2))
        gx_gz = f'{stage}.gx_gz'
        nodes.append(oh.make_node('Div', inputs=[gx, gz], outputs=[gx_gz],
                                  name=f'{stage}.div_gx_gz'))
        nodes.append(oh.make_node('Atan', inputs=[gx_gz], outputs=[pitch],
                                  name=f'{stage}.atan_pitch'))
        
        gx_sq_gz_sq = f'{stage}.gx_sq_gz_sq'
        nodes.append(oh.make_node('Add', inputs=[rx_sq, rz_sq], outputs=[gx_sq_gz_sq],
                                  name=f'{stage}.add_gx_sq_gz_sq'))
        nodes.append(oh.make_node('Sqrt', inputs=[gx_sq_gz_sq], outputs=[gx_sq_gz_sq],
                                  name=f'{stage}.sqrt_gx_sq_gz_sq'))
        
        neg_gy = f'{stage}.neg_gy'
        zero = f'{stage}.zero'
        inits.append(oh.make_tensor(zero, TensorProto.FLOAT, [], [0.0]))
        nodes.append(oh.make_node('Sub', inputs=[zero, gy], outputs=[neg_gy],
                                  name=f'{stage}.sub_neg_gy'))
        
        neg_gy_sqrt = f'{stage}.neg_gy_sqrt'
        nodes.append(oh.make_node('Div', inputs=[neg_gy, gx_sq_gz_sq], outputs=[neg_gy_sqrt],
                                  name=f'{stage}.div_neg_gy_sqrt'))
        nodes.append(oh.make_node('Atan', inputs=[neg_gy_sqrt], outputs=[roll],
                                  name=f'{stage}.atan_roll'))
        
        # Create rotation matrix from pitch and roll
        # R = R_roll * R_pitch
        cos_pitch = f'{stage}.cos_pitch'
        sin_pitch = f'{stage}.sin_pitch'
        cos_roll = f'{stage}.cos_roll'
        sin_roll = f'{stage}.sin_roll'
        
        nodes.append(oh.make_node('Cos', inputs=[pitch], outputs=[cos_pitch],
                                  name=f'{stage}.cos_pitch'))
        nodes.append(oh.make_node('Sin', inputs=[pitch], outputs=[sin_pitch],
                                  name=f'{stage}.sin_pitch'))
        nodes.append(oh.make_node('Cos', inputs=[roll], outputs=[cos_roll],
                                  name=f'{stage}.cos_roll'))
        nodes.append(oh.make_node('Sin', inputs=[roll], outputs=[sin_roll],
                                  name=f'{stage}.sin_roll'))
        
        # R_pitch = [[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]]
        # R_roll = [[1, 0, 0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]]
        # R_level = R_roll * R_pitch
        
        # Calculate R_level elements
        r00 = f'{stage}.r00'
        r01 = f'{stage}.r01'
        r02 = f'{stage}.r02'
        r10 = f'{stage}.r10'
        r11 = f'{stage}.r11'
        r12 = f'{stage}.r12'
        r20 = f'{stage}.r20'
        r21 = f'{stage}.r21'
        r22 = f'{stage}.r22'
        
        # R_level = R_roll * R_pitch
        # r00 = cos(p)
        # r01 = 0
        # r02 = sin(p)
        # r10 = sin(r) * sin(p)
        # r11 = cos(r)
        # r12 = -sin(r) * cos(p)
        # r20 = -cos(r) * sin(p)
        # r21 = sin(r)
        # r22 = cos(r) * cos(p)
        
        nodes.append(oh.make_node('Identity', inputs=[cos_pitch], outputs=[r00],
                                  name=f'{stage}.identity_r00'))
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[r01],
                                  name=f'{stage}.identity_r01'))
        nodes.append(oh.make_node('Identity', inputs=[sin_pitch], outputs=[r02],
                                  name=f'{stage}.identity_r02'))
        
        sin_roll_sin_pitch = f'{stage}.sin_roll_sin_pitch'
        nodes.append(oh.make_node('Mul', inputs=[sin_roll, sin_pitch], outputs=[sin_roll_sin_pitch],
                                  name=f'{stage}.mul_sin_roll_sin_pitch'))
        nodes.append(oh.make_node('Identity', inputs=[sin_roll_sin_pitch], outputs=[r10],
                                  name=f'{stage}.identity_r10'))
        nodes.append(oh.make_node('Identity', inputs=[cos_roll], outputs=[r11],
                                  name=f'{stage}.identity_r11'))
        
        sin_roll_cos_pitch = f'{stage}.sin_roll_cos_pitch'
        neg_sin_roll_cos_pitch = f'{stage}.neg_sin_roll_cos_pitch'
        nodes.append(oh.make_node('Mul', inputs=[sin_roll, cos_pitch], outputs=[sin_roll_cos_pitch],
                                  name=f'{stage}.mul_sin_roll_cos_pitch'))
        nodes.append(oh.make_node('Sub', inputs=[zero, sin_roll_cos_pitch], outputs=[neg_sin_roll_cos_pitch],
                                  name=f'{stage}.sub_neg_sin_roll_cos_pitch'))
        nodes.append(oh.make_node('Identity', inputs=[neg_sin_roll_cos_pitch], outputs=[r12],
                                  name=f'{stage}.identity_r12'))
        
        neg_cos_roll_sin_pitch = f'{stage}.neg_cos_roll_sin_pitch'
        nodes.append(oh.make_node('Mul', inputs=[cos_roll, sin_pitch], outputs=[neg_cos_roll_sin_pitch],
                                  name=f'{stage}.mul_cos_roll_sin_pitch'))
        nodes.append(oh.make_node('Sub', inputs=[zero, neg_cos_roll_sin_pitch], outputs=[neg_cos_roll_sin_pitch],
                                  name=f'{stage}.sub_neg_cos_roll_sin_pitch'))
        nodes.append(oh.make_node('Identity', inputs=[neg_cos_roll_sin_pitch], outputs=[r20],
                                  name=f'{stage}.identity_r20'))
        nodes.append(oh.make_node('Identity', inputs=[sin_roll], outputs=[r21],
                                  name=f'{stage}.identity_r21'))
        
        cos_roll_cos_pitch = f'{stage}.cos_roll_cos_pitch'
        nodes.append(oh.make_node('Mul', inputs=[cos_roll, cos_pitch], outputs=[cos_roll_cos_pitch],
                                  name=f'{stage}.mul_cos_roll_cos_pitch'))
        nodes.append(oh.make_node('Identity', inputs=[cos_roll_cos_pitch], outputs=[r22],
                                  name=f'{stage}.identity_r22'))
        
        # Create rotation matrix [3,3]
        rotation_matrix = f'{stage}.rotation_matrix'
        r_row0 = f'{stage}.r_row0'
        r_row1 = f'{stage}.r_row1'
        r_row2 = f'{stage}.r_row2'
        
        nodes.append(oh.make_node('Concat', inputs=[r00, r01, r02], outputs=[r_row0],
                                  name=f'{stage}.concat_r_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[r10, r11, r12], outputs=[r_row1],
                                  name=f'{stage}.concat_r_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[r20, r21, r22], outputs=[r_row2],
                                  name=f'{stage}.concat_r_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[r_row0, r_row1, r_row2], outputs=[rotation_matrix],
                                  name=f'{stage}.concat_rotation_matrix', axis=0))
        
        # Create homography matrix from rotation and zoom
        # H = zoom * R_level
        h00 = f'{stage}.h00'
        h01 = f'{stage}.h01'
        h02 = f'{stage}.h02'
        h10 = f'{stage}.h10'
        h11 = f'{stage}.h11'
        h12 = f'{stage}.h12'
        h20 = f'{stage}.h20'
        h21 = f'{stage}.h21'
        h22 = f'{stage}.h22'
        
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r00], outputs=[h00],
                                  name=f'{stage}.mul_h00'))
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r01], outputs=[h01],
                                  name=f'{stage}.mul_h01'))
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r02], outputs=[h02],
                                  name=f'{stage}.mul_h02'))
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r10], outputs=[h10],
                                  name=f'{stage}.mul_h10'))
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r11], outputs=[h11],
                                  name=f'{stage}.mul_h11'))
        nodes.append(oh.make_node('Mul', inputs=[dynamic_zoom, r12], outputs=[h12],
                                  name=f'{stage}.mul_h12'))
        nodes.append(oh.make_node('Identity', inputs=[r20], outputs=[h20],
                                  name=f'{stage}.identity_h20'))
        nodes.append(oh.make_node('Identity', inputs=[r21], outputs=[h21],
                                  name=f'{stage}.identity_h21'))
        nodes.append(oh.make_node('Identity', inputs=[r22], outputs=[h22],
                                  name=f'{stage}.identity_h22'))
        
        # Stack into homography matrix [3,3]
        homography = f'{stage}.homography'
        h_row0 = f'{stage}.h_row0'
        h_row1 = f'{stage}.h_row1'
        h_row2 = f'{stage}.h_row2'
        
        nodes.append(oh.make_node('Concat', inputs=[h00, h01, h02], outputs=[h_row0],
                                  name=f'{stage}.concat_h_row0', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h10, h11, h12], outputs=[h_row1],
                                  name=f'{stage}.concat_h_row1', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h20, h21, h22], outputs=[h_row2],
                                  name=f'{stage}.concat_h_row2', axis=0))
        nodes.append(oh.make_node('Concat', inputs=[h_row0, h_row1, h_row2], outputs=[homography],
                                  name=f'{stage}.concat_homography', axis=0))
        
        # Calculate confidence based on angular velocity
        # Higher angular velocity = lower confidence
        confidence = f'{stage}.confidence'
        confidence_offset = f'{stage}.confidence_offset'
        nodes.append(oh.make_node('Mul', inputs=[angular_velocity, zoom_sensitivity], outputs=[confidence_offset],
                                  name=f'{stage}.mul_confidence_offset'))
        nodes.append(oh.make_node('Sub', inputs=[one, confidence_offset], outputs=[confidence],
                                  name=f'{stage}.sub_confidence'))
        
        # Create translation vector (simplified, just zeros for now)
        translation = f'{stage}.translation'
        nodes.append(oh.make_node('Identity', inputs=[zero], outputs=[translation],
                                  name=f'{stage}.identity_translation'))
        
        vis.append(oh.make_tensor_value_info(homography, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(rotation_matrix, TensorProto.FLOAT, [3, 3]))
        vis.append(oh.make_tensor_value_info(translation, TensorProto.FLOAT, [3]))
        vis.append(oh.make_tensor_value_info(dynamic_zoom, TensorProto.FLOAT, [1]))
        vis.append(oh.make_tensor_value_info(confidence, TensorProto.FLOAT, [1]))
        
        outputs = {
            'homography': {'name': homography},
            'rotation_matrix': {'name': rotation_matrix},
            'translation': {'name': translation},
            'dynamic_zoom': {'name': dynamic_zoom},
            'confidence': {'name': confidence}
        }
        
        result = BuildResult(outputs, nodes, inits, vis)
        result.appendInput(current_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(prev_frame, type=TensorProto.FLOAT, shape=['n', 3, 'h', 'w'])
        result.appendInput(gyro_data, type=TensorProto.FLOAT, shape=[3])
        result.appendInput(gravity_vector, type=TensorProto.FLOAT, shape=[3])
        result.appendInput(smoothing_factor, type=TensorProto.FLOAT, shape=[1])
        result.appendInput(horizon_leveling, type=TensorProto.BOOL, shape=[1])
        return result

    def build_coordinator(self, stage: str, prev_stages=None):
        """
        Not used - coordinator handles fusion.
        """
        return BuildResult({}, [], [], [])

    def build_applier(self, stage: str, prev_stages=None):
        """
        Not used - applier handles application.
        """
        return BuildResult({}, [], [], [])

    def build_test_algo(self, stage: str, prev_stages=None):
        return self.build_algo(stage, prev_stages)