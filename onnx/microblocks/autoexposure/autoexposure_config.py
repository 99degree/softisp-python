# autoexposure_config.py
"""
Configuration system for auto exposure microblock
"""

class AutoExposureConfig:
    """Configuration for auto exposure algorithms"""
    
    def __init__(self):
        # Exposure targets
        self.target_brightness = 0.18  # 18% gray card standard
        self.min_ev = -2.0  # Minimum exposure value
        self.max_ev = 2.0   # Maximum exposure value
        
        # Smoothing and filtering
        self.smoothing_factor = 0.1  # Temporal smoothing factor
        self.enable_smoothing = True
        
        # Zone weights for weighted brightness calculation
        self.center_weight = 0.7  # Weight for center zone
        self.zone_weights = [0.5, 0.7, 1.0, 0.7, 0.5]  # 5x5 zone weights
        
        # RGB-specific settings
        self.rgb_weights = [0.299, 0.587, 0.114]  # BT.601 luminance weights
        self.enable_color_aware = True
        
        # YUV-specific settings
        self.yuv_luminance_only = True  # Use only Y channel for exposure
        self.preserve_chrominance = True
        
        # Advanced features
        self.enable_histogram_analysis = False
        self.histogram_percentile = 50  # Median brightness
        self.enable_multi_zone = False
        
        # Error handling
        self.enable_edge_case_handling = True
        self.min_valid_brightness = 0.01
        self.max_valid_brightness = 0.99
        
        # Performance
        self.enable_optimization = True
        self.use_fast_approximation = False


class AutoExposureStatsConfig(AutoExposureConfig):
    """Configuration for statistics-based auto exposure"""
    
    def __init__(self):
        super().__init__()
        # RGB-specific defaults
        self.enable_color_aware = True
        self.rgb_weights = [0.299, 0.587, 0.114]  # BT.601 luminance weights
        self.enable_multi_zone = True
        self.zone_weights = [0.5, 0.7, 1.0, 0.7, 0.5]


class AutoExposureYUVConfig(AutoExposureConfig):
    """Configuration for YUV-based auto exposure"""
    
    def __init__(self):
        super().__init__()
        # YUV-specific defaults
        self.yuv_luminance_only = True
        self.preserve_chrominance = True
        self.enable_color_aware = False
        self.enable_multi_zone = False