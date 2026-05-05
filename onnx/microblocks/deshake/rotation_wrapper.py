"""
Universal Rotation Wrapper for Deshake Coordinator

This module provides a universal interface for handling different rotation formats
(Quaternions, Euler angles, Rotation Matrices) in the coordinator.
"""

import numpy as np
from typing import Union, List


class RotationState:
    """
    Abstract Rotation Wrapper for the Coordinator Trace
    
    Handles conversion between different rotation formats:
    - Quaternions (4 floats: [w, x, y, z])
    - Euler angles (3 floats: [roll, pitch, yaw])
    - Rotation matrices (3x3 matrix)
    
    All formats are converted to a unified 3x3 rotation matrix for processing.
    """
    
    def __init__(self, data: Union[np.ndarray, List[float]], mode: str = 'quat'):
        """
        Initialize RotationState with data in specified format.
        
        Args:
            data: Rotation data (quaternion, euler, or matrix)
            mode: Input format ('quat', 'euler', or 'matrix')
        """
        self.mode = mode
        
        if mode == 'quat':
            self.matrix = self.quat_to_matrix(data)
            self.quaternion = np.array(data, dtype=np.float32)
        elif mode == 'euler':
            self.matrix = self.euler_to_matrix(data)
            self.euler = np.array(data, dtype=np.float32)
        elif mode == 'matrix':
            self.matrix = np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"Unknown rotation mode: {mode}")
        
        # Store timestamp for interpolation
        self.timestamp = 0.0
    
    @staticmethod
    def quat_to_matrix(q: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Quaternion format: [w, x, y, z] (scalar first)
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        q = np.array(q, dtype=np.float32)
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.zeros((3, 3), dtype=np.float32)
        R[0, 0] = 1 - 2*y*y - 2*z*z
        R[0, 1] = 2*x*y - 2*z*w
        R[0, 2] = 2*x*z + 2*y*w
        R[1, 0] = 2*x*y + 2*z*w
        R[1, 1] = 1 - 2*x*x - 2*z*z
        R[1, 2] = 2*y*z - 2*x*w
        R[2, 0] = 2*x*z - 2*y*w
        R[2, 1] = 2*y*z + 2*x*w
        R[2, 2] = 1 - 2*x*x - 2*y*y
        
        return R
    
    @staticmethod
    def euler_to_matrix(euler: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Euler format: [roll, pitch, yaw] (in radians)
        
        Args:
            euler: Euler angles [roll, pitch, yaw]
            
        Returns:
            3x3 rotation matrix
        """
        euler = np.array(euler, dtype=np.float32)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]], dtype=np.float32)
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                         [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]], dtype=np.float32)
        
        # Combined rotation: R = R_z * R_y * R_x
        R = R_z @ R_y @ R_x
        
        return R
    
    @staticmethod
    def matrix_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        # Calculate quaternion from rotation matrix
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return np.array([w, x, y, z], dtype=np.float32)
    
    @staticmethod
    def matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to Euler angles.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Euler angles [roll, pitch, yaw] (in radians)
        """
        # Calculate Euler angles from rotation matrix
        # Using ZYX convention (yaw, pitch, roll)
        
        # Pitch (y-axis rotation)
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def slerp(self, other: 'RotationState', t: float) -> 'RotationState':
        """
        Spherical Linear Interpolation between two rotations.
        
        Args:
            other: Other RotationState to interpolate to
            t: Interpolation factor (0 to 1)
            
        Returns:
            New RotationState at interpolated position
        """
        # Convert both to quaternions
        q1 = self.matrix_to_quat(self.matrix)
        q2 = self.matrix_to_quat(other.matrix)
        
        # Calculate dot product
        dot = np.dot(q1, q2)
        
        # If dot is negative, negate one quaternion to take shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are too close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            result = result / np.linalg.norm(result)
            return RotationState(result, mode='quat')
        
        # Calculate angle
        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t
        
        # Calculate interpolated quaternion
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result = s0 * q1 + s1 * q2
        result = result / np.linalg.norm(result)
        
        return RotationState(result, mode='quat')
    
    def lerp(self, other: 'RotationState', t: float) -> 'RotationState':
        """
        Linear Interpolation between two rotations (for Euler angles).
        
        Args:
            other: Other RotationState to interpolate to
            t: Interpolation factor (0 to 1)
            
        Returns:
            New RotationState at interpolated position
        """
        # Convert both to Euler angles
        e1 = self.matrix_to_euler(self.matrix)
        e2 = self.matrix_to_euler(other.matrix)
        
        # Linear interpolation
        result = e1 + t * (e2 - e1)
        
        return RotationState(result, mode='euler')
    
    def inverse(self) -> 'RotationState':
        """
        Calculate inverse rotation.
        
        Returns:
            New RotationState with inverse rotation
        """
        R_inv = self.matrix.T  # For rotation matrices, inverse = transpose
        return RotationState(R_inv, mode='matrix')
    
    def compose(self, other: 'RotationState') -> 'RotationState':
        """
        Compose two rotations (R = R1 * R2).
        
        Args:
            other: Other RotationState to compose with
            
        Returns:
            New RotationState with composed rotation
        """
        R = self.matrix @ other.matrix
        return RotationState(R, mode='matrix')
    
    def __mul__(self, other: 'RotationState') -> 'RotationState':
        """Compose two rotations using * operator."""
        return self.compose(other)
    
    def __repr__(self) -> str:
        return f"RotationState(mode={self.mode}, matrix=\n{self.matrix})"


class RotationTrace:
    """
    Maintains a history of RotationState for smoothing and interpolation.
    
    Handles sample rate mismatch between different sensors:
    - Algo (NPU): 60 Hz (Frame rate)
    - IMU (CPU/Sensor): 200 Hz - 1000 Hz
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize RotationTrace.
        
        Args:
            max_history: Maximum number of samples to keep in history
        """
        self.max_history = max_history
        self.history: List[RotationState] = []
    
    def add(self, rotation: RotationState, timestamp: float = None):
        """
        Add a rotation to the trace.
        
        Args:
            rotation: RotationState to add
            timestamp: Timestamp of the rotation (optional)
        """
        if timestamp is not None:
            rotation.timestamp = timestamp
        
        self.history.append(rotation)
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_at_time(self, timestamp: float, interpolation: str = 'slerp') -> RotationState:
        """
        Get rotation at a specific timestamp using interpolation.
        
        Args:
            timestamp: Target timestamp
            interpolation: Interpolation method ('slerp' or 'lerp')
            
        Returns:
            RotationState at target timestamp
        """
        if len(self.history) == 0:
            raise ValueError("No rotation history available")
        
        if len(self.history) == 1:
            return self.history[0]
        
        # Find surrounding samples
        for i in range(len(self.history) - 1):
            if self.history[i].timestamp <= timestamp <= self.history[i + 1].timestamp:
                # Interpolate between samples
                t1 = self.history[i].timestamp
                t2 = self.history[i + 1].timestamp
                
                if t2 == t1:
                    return self.history[i]
                
                t = (timestamp - t1) / (t2 - t1)
                
                if interpolation == 'slerp':
                    return self.history[i].slerp(self.history[i + 1], t)
                else:
                    return self.history[i].lerp(self.history[i + 1], t)
        
        # If timestamp is outside range, return closest
        if timestamp < self.history[0].timestamp:
            return self.history[0]
        else:
            return self.history[-1]
    
    def get_center_of_exposure(self, frame_time: float, shutter_time: float) -> RotationState:
        """
        Get rotation at center of exposure for a frame.
        
        Args:
            frame_time: Frame capture time
            shutter_time: Shutter exposure time
            
        Returns:
            RotationState at center of exposure
        """
        center_time = frame_time + shutter_time / 2.0
        return self.get_at_time(center_time)
    
    def smooth(self, window_size: int = 5, method: str = 'moving_average') -> List[RotationState]:
        """
        Smooth the rotation trace.
        
        Args:
            window_size: Size of smoothing window
            method: Smoothing method ('moving_average' or 'exponential')
            
        Returns:
            List of smoothed RotationState
        """
        if len(self.history) == 0:
            return []
        
        if method == 'moving_average':
            return self._moving_average_smooth(window_size)
        elif method == 'exponential':
            return self._exponential_smooth(alpha=0.8)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def _moving_average_smooth(self, window_size: int) -> List[RotationState]:
        """Moving average smoothing."""
        smoothed = []
        
        for i in range(len(self.history)):
            start = max(0, i - window_size // 2)
            end = min(len(self.history), i + window_size // 2 + 1)
            
            # Average quaternions
            quats = [self.matrix_to_quat(self.history[j].matrix) for j in range(start, end)]
            avg_quat = np.mean(quats, axis=0)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)
            
            smoothed.append(RotationState(avg_quat, mode='quat'))
        
        return smoothed
    
    def _exponential_smooth(self, alpha: float = 0.8) -> List[RotationState]:
        """Exponential smoothing."""
        if len(self.history) == 0:
            return []
        
        smoothed = [self.history[0]]
        
        for i in range(1, len(self.history)):
            # Exponential smoothing: y[i] = alpha * x[i] + (1-alpha) * y[i-1]
            prev_quat = self.matrix_to_quat(smoothed[-1].matrix)
            curr_quat = self.matrix_to_quat(self.history[i].matrix)
            
            # Linear interpolation between previous and current
            smoothed_quat = alpha * curr_quat + (1 - alpha) * prev_quat
            smoothed_quat = smoothed_quat / np.linalg.norm(smoothed_quat)
            
            smoothed.append(RotationState(smoothed_quat, mode='quat'))
        
        return smoothed
    
    def get_correction_matrix(self, target: RotationState) -> np.ndarray:
        """
        Calculate correction matrix to transform from current to target rotation.
        
        Args:
            target: Target rotation state
            
        Returns:
            3x3 correction matrix
        """
        # Correction: R_corr = R_target * R_current^(-1)
        R_current_inv = self.history[-1].inverse()
        R_corr = target.matrix @ R_current_inv.matrix
        
        return R_corr
    
    def __len__(self) -> int:
        return len(self.history)
    
    def __repr__(self) -> str:
        return f"RotationTrace(count={len(self.history)}, max={self.max_history})"