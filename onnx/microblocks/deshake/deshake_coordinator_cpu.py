"""
Stateful CPU Coordinator for Deshake

This module provides a stateful coordinator that maintains rotation history,
performs smoothing, and calculates correction matrices for the ONNX grid generator.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .rotation_wrapper import RotationState, RotationTrace


class DeshakeCoordinatorCPU:
    """
    Stateful CPU Coordinator for Deshake
    
    Maintains rotation history, performs smoothing, and calculates correction
    matrices for the ONNX grid generator. Handles sample rate mismatch between
    different sensors.
    
    Architecture:
    - Algo (NPU): 60 Hz (Frame rate)
    - IMU (CPU/Sensor): 200 Hz - 1000 Hz
    - Coordinator Trace: Interpolates to match frame rate
    """
    
    def __init__(self, 
                 max_history: int = 100,
                 smoothing_window: int = 5,
                 smoothing_method: str = 'exponential',
                 smoothing_alpha: float = 0.8):
        """
        Initialize DeshakeCoordinatorCPU.
        
        Args:
            max_history: Maximum number of samples to keep in history
            smoothing_window: Size of smoothing window (for moving average)
            smoothing_method: Smoothing method ('moving_average' or 'exponential')
            smoothing_alpha: Alpha parameter for exponential smoothing
        """
        self.rotation_trace = RotationTrace(max_history=max_history)
        self.smoothing_window = smoothing_window
        self.smoothing_method = smoothing_method
        self.smoothing_alpha = smoothing_alpha
        
        # Camera intrinsics
        self.focal_length = 1000.0  # Default focal length in pixels
        self.principal_point = (960.0, 540.0)  # Default principal point (cx, cy)
        
        # GDC coefficients
        self.gdc_coeffs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [k1, k2, p1, p2]
        
        # Current state
        self.current_rotation: Optional[RotationState] = None
        self.target_rotation: Optional[RotationState] = None
        self.correction_matrix = np.eye(3, dtype=np.float32)
        
        # Timing
        self.frame_time = 0.0
        self.shutter_time = 0.0167  # Default 60fps shutter time
        
    def set_camera_intrinsics(self, focal_length: float, cx: float, cy: float):
        """
        Set camera intrinsics.
        
        Args:
            focal_length: Focal length in pixels
            cx: Principal point x coordinate
            cy: Principal point y coordinate
        """
        self.focal_length = focal_length
        self.principal_point = (cx, cy)
    
    def set_gdc_coeffs(self, k1: float, k2: float, p1: float, p2: float):
        """
        Set GDC distortion coefficients.
        
        Args:
            k1: Radial distortion coefficient 1
            k2: Radial distortion coefficient 2
            p1: Tangential distortion coefficient 1
            p2: Tangential distortion coefficient 2
        """
        self.gdc_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)
    
    def add_rotation(self, 
                    data: np.ndarray, 
                    mode: str = 'quat',
                    timestamp: Optional[float] = None):
        """
        Add a rotation to the trace.
        
        Args:
            data: Rotation data (quaternion, euler, or matrix)
            mode: Input format ('quat', 'euler', or 'matrix')
            timestamp: Timestamp of the rotation (optional)
        """
        rotation = RotationState(data, mode=mode)
        
        if timestamp is not None:
            rotation.timestamp = timestamp
        
        self.rotation_trace.add(rotation, timestamp)
        self.current_rotation = rotation
    
    def add_imu_rotation(self, 
                         gyro_data: np.ndarray,
                         dt: float,
                         timestamp: Optional[float] = None):
        """
        Add IMU rotation (integrated from gyroscope data).
        
        Args:
            gyro_data: Gyroscope data [rx, ry, rz] (rad/s)
            dt: Time step (seconds)
            timestamp: Timestamp of the rotation (optional)
        """
        # Integrate gyro to get rotation
        # For small angles, rotation ≈ gyro * dt
        euler = gyro_data * dt
        
        if timestamp is None:
            timestamp = self.frame_time
        
        self.add_rotation(euler, mode='euler', timestamp=timestamp)
    
    def set_frame_timing(self, frame_time: float, shutter_time: float):
        """
        Set frame timing for center-of-exposure calculation.
        
        Args:
            frame_time: Frame capture time
            shutter_time: Shutter exposure time
        """
        self.frame_time = frame_time
        self.shutter_time = shutter_time
    
    def get_center_of_exposure_rotation(self) -> RotationState:
        """
        Get rotation at center of exposure for current frame.
        
        Returns:
            RotationState at center of exposure
        """
        return self.rotation_trace.get_center_of_exposure(
            self.frame_time, 
            self.shutter_time
        )
    
    def smooth_rotation_trace(self) -> list:
        """
        Smooth the rotation trace.
        
        Returns:
            List of smoothed RotationState
        """
        if self.smoothing_method == 'moving_average':
            return self.rotation_trace.smooth(
                window_size=self.smoothing_window,
                method='moving_average'
            )
        else:
            return self.rotation_trace.smooth(
                method='exponential'
            )
    
    def calculate_correction_matrix(self, 
                                  target_rotation: Optional[RotationState] = None) -> np.ndarray:
        """
        Calculate correction matrix to transform from current to target rotation.
        
        Args:
            target_rotation: Target rotation state (optional, uses identity if None)
            
        Returns:
            3x3 correction matrix
        """
        if target_rotation is None:
            # Use identity (no correction)
            target_rotation = RotationState(np.eye(3), mode='matrix')
        
        if len(self.rotation_trace) == 0:
            return np.eye(3, dtype=np.float32)
        
        # Get smoothed rotation at center of exposure
        smoothed_trace = self.smooth_rotation_trace()
        if len(smoothed_trace) == 0:
            current = self.rotation_trace.history[-1]
        else:
            current = smoothed_trace[-1]
        
        # Calculate correction: R_corr = R_target * R_current^(-1)
        R_current_inv = current.inverse()
        R_corr = target_rotation.matrix @ R_current_inv.matrix
        
        self.correction_matrix = R_corr
        return R_corr
    
    def calculate_horizon_leveling(self, gravity_vector: np.ndarray) -> np.ndarray:
        """
        Calculate horizon leveling rotation matrix.
        
        Args:
            gravity_vector: Gravity vector [gx, gy, gz] (normalized)
            
        Returns:
            3x3 rotation matrix for horizon leveling
        """
        gx, gy, gz = gravity_vector[0], gravity_vector[1], gravity_vector[2]
        
        # Calculate pitch and roll from gravity vector
        # pitch = atan2(gx, gz)
        # roll = atan2(-gy, sqrt(gx^2 + gz^2))
        
        pitch = np.arctan2(gx, gz)
        roll = np.arctan2(-gy, np.sqrt(gx*gx + gz*gz))
        
        # Create rotation matrices
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ], dtype=np.float32)
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ], dtype=np.float32)
        
        # Combined rotation: R_level = R_roll * R_pitch
        R_level = R_roll @ R_pitch
        
        return R_level
    
    def calculate_dynamic_zoom(self, angular_velocity: np.ndarray, 
                             zoom_sensitivity: float = 0.1) -> float:
        """
        Calculate dynamic zoom factor based on shake severity.
        
        Args:
            angular_velocity: Angular velocity vector [rx, ry, rz] (rad/s)
            zoom_sensitivity: Zoom sensitivity factor
            
        Returns:
            Zoom factor (1.0 = no zoom)
        """
        # Calculate magnitude of angular velocity
        magnitude = np.linalg.norm(angular_velocity)
        
        # Calculate zoom: Z = 1.0 + β * |angular_velocity|
        zoom = 1.0 + zoom_sensitivity * magnitude
        
        return zoom
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Get camera intrinsic matrix K.
        
        Returns:
            3x3 camera intrinsic matrix
        """
        cx, cy = self.principal_point
        K = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def get_decision_result(self) -> Dict[str, Any]:
        """
        Get the decision result for the ONNX grid generator.
        
        Returns:
            Dictionary containing:
            - correction_matrix: 3x3 correction matrix
            - camera_matrix: 3x3 camera intrinsic matrix
            - gdc_coeffs: GDC distortion coefficients [k1, k2, p1, p2]
            - dynamic_zoom: Dynamic zoom factor
            - horizon_leveling: Horizon leveling matrix (optional)
        """
        result = {
            'correction_matrix': self.correction_matrix,
            'camera_matrix': self.get_camera_matrix(),
            'gdc_coeffs': self.gdc_coeffs,
            'dynamic_zoom': 1.0
        }
        
        return result
    
    def reset(self):
        """Reset coordinator state."""
        self.rotation_trace = RotationTrace(max_history=self.rotation_trace.max_history)
        self.current_rotation = None
        self.target_rotation = None
        self.correction_matrix = np.eye(3, dtype=np.float32)
    
    def __repr__(self) -> str:
        return (f"DeshakeCoordinatorCPU("
                f"history={len(self.rotation_trace)}, "
                f"smoothing={self.smoothing_method}, "
                f"alpha={self.smoothing_alpha})")