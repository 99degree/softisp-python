# autoexposure.py
from microblocks.base import BuildResult, MicroblockBase
import onnx.helper as oh
from onnx import TensorProto


def _n(stage, suffix):
    """Generate unique node name per stage"""
    return f"{stage}.{suffix}"



class AutoExposureBase(MicroblockBase):
    """Auto Exposure Base Class - DO NOT REGISTER - used as parent class"""
    
    # Use invalid name so this class won't be auto-registered
    name = None
    version = None
    
    def build_algo(self, stage, prev_stages=None):
        # Delegate to subclass implementation
        return super().build_algo(stage, prev_stages)


# Version-specific classes with proper implementations
class AutoExposureV1(AutoExposureBase):
    """Auto Exposure V1 - delegate to actual implementation"""
    # Don't override name/version - will be set by subclass
    pass


class AutoExposurePassthrough(AutoExposureBase):
    """Auto Exposure Passthrough - delegate to actual implementation"""
    pass