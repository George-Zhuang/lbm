import numpy as np
from typing import Dict, Optional, Tuple

class KalmanFilter3D:
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        """
        Initialize 3D Kalman Filter
        
        Args:
            process_noise (float): Process noise covariance
            measurement_noise (float): Measurement noise covariance
        """
        # State vector: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.state_cov = np.eye(6) * 1000  # Initial state covariance
        
        # State transition matrix
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3)  # Relationship between position and velocity
        
        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        self.Q[3:6, 3:6] *= 10  # Larger noise for velocity
        
        # Measurement matrix (only measuring position)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise
        
        # Time step
        self.dt = 1.0
        
    def predict(self):
        """Prediction step"""
        # Update time step in state transition matrix
        self.F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.state_cov = self.F @ self.state_cov @ self.F.T + self.Q
        
    def update(self, measurement, visibility):
        """
        Update step
        
        Args:
            measurement (np.ndarray): 3D coordinate measurement [x, y, z]
            visibility (float): Visibility (0-1)
        """
        # Adjust measurement noise based on visibility
        adjusted_R = self.R / (visibility + 1e-6)  # Avoid division by zero
        
        # Calculate Kalman gain
        S = self.H @ self.state_cov @ self.H.T + adjusted_R
        K = self.state_cov @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        
        # Update covariance
        I = np.eye(6)
        self.state_cov = (I - K @ self.H) @ self.state_cov
        
    def get_position(self):
        """Get current position"""
        return self.state[0:3]
    
    def get_velocity(self):
        """Get current velocity"""
        return self.state[3:6]

class MultiPointKF3D:
    def __init__(self, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0,
                 visibility_threshold: float = 0.3,
                 reinit_visibility_threshold: float = 0.7,
                 lost_frames_threshold: int = 10):
        """
        Multi-point Kalman Filter Management System
        
        Args:
            process_noise (float): Process noise
            measurement_noise (float): Measurement noise
            visibility_threshold (float): Threshold for point loss detection
            reinit_visibility_threshold (float): Threshold for point reinitialization
            lost_frames_threshold (int): Number of consecutive frames for point loss detection
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.visibility_threshold = visibility_threshold
        self.reinit_visibility_threshold = reinit_visibility_threshold
        self.lost_frames_threshold = lost_frames_threshold
        
        # Store Kalman filters for each point
        self.kf_dict: Dict[int, KalmanFilter3D] = {}
        # Store lost frame count for each point
        self.lost_frames: Dict[int, int] = {}
        # Store status for each point (active/lost)
        self.point_status: Dict[int, str] = {}
        
    def update(self, point_id: int, measurement: np.ndarray, 
               visibility: float) -> None:
        """
        Update state for a specific point
        
        Args:
            point_id (int): Unique identifier for the point
            measurement (np.ndarray): 3D coordinate measurement [x, y, z]
            visibility (float): Visibility (0-1)
        """
        # Initialize Kalman filter for new points
        if point_id not in self.kf_dict:
            self._init_point(point_id, measurement)
            return
            
        # Get current point status
        status = self.point_status.get(point_id, "active")
        
        if status == "active":
            # Check if point should be marked as lost
            if visibility < self.visibility_threshold:
                self.lost_frames[point_id] = self.lost_frames.get(point_id, 0) + 1
                if self.lost_frames[point_id] >= self.lost_frames_threshold:
                    self.point_status[point_id] = "lost"
            else:
                # Normal update
                self.kf_dict[point_id].predict()
                self.kf_dict[point_id].update(measurement, visibility)
                self.lost_frames[point_id] = 0
                
        elif status == "lost":
            # Check if point should be reinitialized
            if visibility >= self.reinit_visibility_threshold:
                self._reinit_point(point_id, measurement)
                
    def _init_point(self, point_id: int, measurement: np.ndarray) -> None:
        """Initialize a new point"""
        kf = KalmanFilter3D(self.process_noise, self.measurement_noise)
        kf.state[0:3] = measurement  # Initialize position
        self.kf_dict[point_id] = kf
        self.lost_frames[point_id] = 0
        self.point_status[point_id] = "active"
        
    def _reinit_point(self, point_id: int, measurement: np.ndarray) -> None:
        """Reinitialize a lost point"""
        self._init_point(point_id, measurement)
        
    def get_point_position(self, point_id: int) -> Optional[np.ndarray]:
        """Get position for a specific point"""
        if point_id in self.kf_dict and self.point_status.get(point_id) == "active":
            return self.kf_dict[point_id].get_position()
        return None
        
    def get_point_velocity(self, point_id: int) -> Optional[np.ndarray]:
        """Get velocity for a specific point"""
        if point_id in self.kf_dict and self.point_status.get(point_id) == "active":
            return self.kf_dict[point_id].get_velocity()
        return None
        
    def get_all_active_points(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Get positions and velocities for all active points"""
        active_points = {}
        for point_id, status in self.point_status.items():
            if status == "active":
                position = self.get_point_position(point_id)
                velocity = self.get_point_velocity(point_id)
                if position is not None and velocity is not None:
                    active_points[point_id] = (position, velocity)
        return active_points
