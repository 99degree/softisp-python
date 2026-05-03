#!/usr/bin/env python3

# Simple test script to verify autoexposure microblock
import sys
import os

# Add the current directory to the path so we can import microblocks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_autoexposure_implementation():
    """Test that our autoexposure microblock can be imported"""
    try:
        # Try to import the class definition
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        print("Autoexposure classes imported successfully")
        
        # Test creating instances
        ae_base = AutoExposureBase()
        ae_stats = AutoExposureStats()
        ae_yuv = AutoExposureYUV()
        
        print("Autoexposure class instances created successfully")
        
        # Check that the classes have the expected attributes
        print(f"AutoExposureStats name: {ae_stats.name}, version: {ae_stats.version}")
        print(f"AutoExposureYUV name: {ae_yuv.name}, version: {ae_yuv.version}")
        
        return True
        
    except Exception as e:
        print(f"Error testing autoexposure implementation: {e}")
        return False

if __name__ == "__main__":
    if test_autoexposure_implementation():
        print("SUCCESS: Autoexposure microblock implementation is valid")
    else:
        print("FAILURE: Autoexposure microblock implementation has issues")
        sys.exit(1)