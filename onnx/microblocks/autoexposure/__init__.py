# microblocks/autoexposure/__init__.py
try:
    from .autoexposure_algo import AutoExposureBase, AutoExposureStats, AutoExposureYUV
    __all__ = ['AutoExposureBase', 'AutoExposureStats', 'AutoExposureYUV']
except ImportError:
    # Fallback for when ONNX is not available
    class AutoExposureBase:
        name = "autoexposure"
        version = "v1"
        
        def build_algo(self, stage, prev_stages=None):
            return "Mock algorithm implementation"
            
        def build_applier(self, stage, prev_stages=None):
            return "Mock applier implementation"
            
        def build_coordinator(self, stage, prev_stages=None):
            return "Mock coordinator implementation"
            
        def build_test_algo(self, stage, prev_stages=None):
            return "Mock test implementation"
    
    class AutoExposureStats(AutoExposureBase):
        name = "autoexposure_stats"
        version = "v1"
    
    class AutoExposureYUV(AutoExposureBase):
        name = "autoexposure_yuv"
        version = "v1"
    
    __all__ = ['AutoExposureBase', 'AutoExposureStats', 'AutoExposureYUV']