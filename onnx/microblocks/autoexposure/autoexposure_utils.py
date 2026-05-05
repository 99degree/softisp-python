# autoexposure_utils.py
"""
Utility functions for auto exposure microblock
"""

import numpy as np


class AutoExposureError(Exception):
    """Base exception for auto exposure errors"""
    pass


class InputValidationError(AutoExposureError):
    """Exception for input validation errors"""
    pass


class ExposureCalculationError(AutoExposureError):
    """Exception for exposure calculation errors"""
    pass


class EdgeCaseHandler:
    """Handler for edge cases in exposure calculation"""
    
    def __init__(self, config):
        self.config = config
    
    def handle_all_black(self, brightness):
        """Handle all-black image case"""
        if brightness < self.config.min_valid_brightness:
            # Return maximum exposure for very dark images
            return self.config.max_ev, 2.0 ** self.config.max_ev
        return None, None
    
    def handle_all_white(self, brightness):
        """Handle all-white image case"""
        if brightness > self.config.max_valid_brightness:
            # Return minimum exposure for very bright images
            return self.config.min_ev, 2.0 ** self.config.min_ev
        return None, None
    
    def handle_invalid_brightness(self, brightness):
        """Handle invalid brightness values"""
        if brightness <= 0:
            raise ExposureCalculationError("Brightness must be positive")
        if brightness > 1.0:
            raise ExposureCalculationError("Brightness must be <= 1.0")
        return True
    
    def clip_exposure_value(self, ev):
        """Clip exposure value to valid range"""
        return max(self.config.min_ev, min(self.config.max_ev, ev))
    
    def safe_division(self, numerator, denominator, default=1.0):
        """Safe division with protection against division by zero"""
        if denominator == 0:
            return default
        return numerator / denominator


class InputValidator:
    """Validator for input images"""
    
    @staticmethod
    def validate_image_shape(image):
        """Validate image shape"""
        if image is None:
            raise InputValidationError("Input image cannot be None")
        
        if len(image.shape) != 4:
            raise InputValidationError(
                f"Input must be 4D tensor [batch, channels, height, width], "
                f"got {len(image.shape)}D tensor"
            )
        
        batch, channels, height, width = image.shape
        
        if batch != 1:
            raise InputValidationError(f"Batch size must be 1, got {batch}")
        
        if channels not in [1, 3, 4]:
            raise InputValidationError(
                f"Channels must be 1, 3, or 4, got {channels}"
            )
        
        if height <= 0 or width <= 0:
            raise InputValidationError(
                f"Height and width must be positive, got {height}x{width}"
            )
        
        return True
    
    @staticmethod
    def validate_value_range(image, min_val=0.0, max_val=1.0):
        """Validate image value range"""
        if np.any(image < min_val) or np.any(image > max_val):
            raise InputValidationError(
                f"Input values must be in range [{min_val}, {max_val}]"
            )
        return True
    
    @staticmethod
    def validate_config(config):
        """Validate configuration"""
        if config is None:
            raise InputValidationError("Configuration cannot be None")
        
        if config.target_brightness <= 0 or config.target_brightness > 1.0:
            raise InputValidationError(
                f"Target brightness must be in (0, 1], got {config.target_brightness}"
            )
        
        if config.min_ev >= config.max_ev:
            raise InputValidationError(
                f"min_ev must be less than max_ev, got {config.min_ev} >= {config.max_ev}"
            )
        
        if config.smoothing_factor < 0 or config.smoothing_factor > 1.0:
            raise InputValidationError(
                f"Smoothing factor must be in [0, 1], got {config.smoothing_factor}"
            )
        
        return True


class ExposureCalculator:
    """Calculator for exposure values"""
    
    def __init__(self, config):
        self.config = config
        self.edge_handler = EdgeCaseHandler(config)
    
    def calculate_exposure_value(self, current_brightness, target_brightness=None):
        """
        Calculate exposure value from current brightness
        
        EV = log2(target_brightness / current_brightness)
        """
        if target_brightness is None:
            target_brightness = self.config.target_brightness
        
        # Handle edge cases
        ev, gain = self.edge_handler.handle_all_black(current_brightness)
        if ev is not None:
            return ev, gain
        
        ev, gain = self.edge_handler.handle_all_white(current_brightness)
        if ev is not None:
            return ev, gain
        
        # Validate brightness
        self.edge_handler.handle_invalid_brightness(current_brightness)
        
        # Calculate exposure value
        # EV = log2(target / current)
        brightness_ratio = self.edge_handler.safe_division(
            target_brightness, current_brightness
        )
        
        # Calculate log2
        ev = np.log2(brightness_ratio)
        
        # Clip to valid range
        ev = self.edge_handler.clip_exposure_value(ev)
        
        # Calculate gain: gain = 2^EV
        gain = 2.0 ** ev
        
        return ev, gain
    
    def calculate_weighted_brightness(self, brightness_values, weights=None):
        """
        Calculate weighted brightness from multiple values
        
        Args:
            brightness_values: List/array of brightness values
            weights: Optional weights for each value
        
        Returns:
            Weighted brightness value
        """
        if weights is None:
            weights = self.config.zone_weights
        
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        weighted_brightness = np.sum(np.array(brightness_values) * weights)
        
        return weighted_brightness
    
    def apply_smoothing(self, current_ev, previous_ev=None):
        """
        Apply temporal smoothing to exposure value
        
        Args:
            current_ev: Current calculated exposure value
            previous_ev: Previous exposure value (for smoothing)
        
        Returns:
            Smoothed exposure value
        """
        if not self.config.enable_smoothing or previous_ev is None:
            return current_ev
        
        # Apply exponential smoothing
        smoothed_ev = (
            self.config.smoothing_factor * current_ev +
            (1 - self.config.smoothing_factor) * previous_ev
        )
        
        return smoothed_ev