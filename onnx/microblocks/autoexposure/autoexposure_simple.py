# autoexposure.py
# Simplified autoexposure implementation for testing without ONNX

class AutoExposureBase:
    """Auto Exposure Base Class - simplified for testing"""
    
    name = "autoexposure"
    version = "v1"
    
    def build_applier(self, stage, prev_stages=None):
        """Build auto exposure applier"""
        return "Mock BuildResult"

    def build_coordinator(self, stage, prev_stages=None):
        return "Mock coordinator"

    def build_algo(self, stage, prev_stages=None):
        """Build the algorithmic part of the auto exposure block."""
        return "Mock algorithm"

    def build_test_algo(self, stage, prev_stages=None):
        """Test algorithmic implementation of auto exposure."""
        return "Mock test algorithm"


# Version-specific classes
class AutoExposureV1(AutoExposureBase):
    name = "autoexposure"
    version = "v1"


class AutoExposurePassthrough(AutoExposureBase):
    name = "autoexposure"
    version = "v1-passthrough"