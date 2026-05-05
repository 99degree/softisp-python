#!/usr/bin/env python3

"""
Test auto exposure utility classes without ONNX dependency
Tests configuration, error handling, and exposure calculator
"""

import sys
import os
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_classes():
    """Test configuration classes directly"""
    print("Testing configuration classes...")
    
    try:
        # Import config classes directly
        from microblocks.autoexposure.autoexposure_config import (
            AutoExposureConfig,
            AutoExposureStatsConfig,
            AutoExposureYUVConfig
        )
        
        # Test base configuration
        base_config = AutoExposureConfig()
        assert base_config.target_brightness == 0.18
        assert base_config.min_ev == -2.0
        assert base_config.max_ev == 2.0
        assert base_config.smoothing_factor == 0.1
        print("  ✓ Base configuration works")
        
        # Test stats configuration
        stats_config = AutoExposureStatsConfig()
        assert stats_config.enable_color_aware == True
        assert stats_config.rgb_weights == [0.299, 0.587, 0.114]
        print("  ✓ Stats configuration works")
        
        # Test YUV configuration
        yuv_config = AutoExposureYUVConfig()
        assert yuv_config.yuv_luminance_only == True
        assert yuv_config.preserve_chrominance == True
        print("  ✓ YUV configuration works")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utility_classes():
    """Test utility classes directly"""
    print("Testing utility classes...")
    
    try:
        # Import utility classes directly
        from microblocks.autoexposure.autoexposure_config import AutoExposureConfig
        from microblocks.autoexposure.autoexposure_utils import (
            ExposureCalculator,
            EdgeCaseHandler,
            InputValidator,
            AutoExposureError,
            InputValidationError,
            ExposureCalculationError
        )
        
        # Test exposure calculator
        config = AutoExposureConfig()
        calculator = ExposureCalculator(config)
        
        ev, gain = calculator.calculate_exposure_value(0.5, 0.18)
        expected_ev = np.log2(0.18 / 0.5)
        expected_gain = 2.0 ** expected_ev
        
        assert abs(ev - expected_ev) < 0.01
        assert abs(gain - expected_gain) < 0.01
        print(f"  ✓ Exposure calculator works (EV={ev:.3f}, gain={gain:.3f})")
        
        # Test edge case handler
        edge_handler = EdgeCaseHandler(config)
        ev, gain = edge_handler.handle_all_black(0.005)
        assert ev == config.max_ev
        print("  ✓ Edge case handler works")
        
        # Test input validator
        validator = InputValidator()
        try:
            validator.validate_image_shape(None)
            assert False, "Should raise error"
        except InputValidationError:
            pass
        print("  ✓ Input validator works")
        
        # Test exception classes
        try:
            raise AutoExposureError("Test error")
        except AutoExposureError as e:
            assert str(e) == "Test error"
        print("  ✓ Exception classes work")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exposure_logic():
    """Test exposure calculation logic"""
    print("Testing exposure calculation logic...")
    
    try:
        from microblocks.autoexposure.autoexposure_config import AutoExposureConfig
        from microblocks.autoexposure.autoexposure_utils import ExposureCalculator
        
        config = AutoExposureConfig()
        calculator = ExposureCalculator(config)
        
        # Test normal case
        ev, gain = calculator.calculate_exposure_value(0.5, 0.18)
        assert ev is not None
        assert gain is not None
        print(f"  ✓ Normal exposure: EV={ev:.3f}, gain={gain:.3f}")
        
        # Test dark image
        ev, gain = calculator.calculate_exposure_value(0.005, 0.18)
        assert ev == config.max_ev
        print(f"  ✓ Dark image: EV={ev:.3f}, gain={gain:.3f}")
        
        # Test bright image
        ev, gain = calculator.calculate_exposure_value(0.995, 0.18)
        assert ev == config.min_ev
        print(f"  ✓ Bright image: EV={ev:.3f}, gain={gain:.3f}")
        
        # Test weighted brightness
        brightness_values = [0.3, 0.5, 0.7]
        weights = [0.2, 0.6, 0.2]
        weighted = calculator.calculate_weighted_brightness(brightness_values, weights)
        expected = 0.3 * 0.2 + 0.5 * 0.6 + 0.7 * 0.2
        assert abs(weighted - expected) < 0.01
        print(f"  ✓ Weighted brightness: {weighted:.3f}")
        
        # Test smoothing
        current_ev = 1.0
        previous_ev = 0.5
        smoothed = calculator.apply_smoothing(current_ev, previous_ev)
        expected_smoothed = 0.1 * 1.0 + 0.9 * 0.5
        assert abs(smoothed - expected_smoothed) < 0.01
        print(f"  ✓ Temporal smoothing: {smoothed:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("AUTO EXPOSURE UTILITIES TEST (NO ONNX DEPENDENCY)")
    print("=" * 70)
    print()
    
    tests = [
        ("Configuration Classes", test_config_classes),
        ("Utility Classes", test_utility_classes),
        ("Exposure Logic", test_exposure_logic),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Print summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
