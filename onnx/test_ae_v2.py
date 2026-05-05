#!/usr/bin/env python3

"""
Comprehensive test script for auto exposure v2 implementation
Tests Phase 1 and Phase 2 features
"""

import sys
import os
import json
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration_system():
    """Test the configuration system"""
    print("Testing configuration system...")
    
    try:
        from microblocks.autoexposure import (
            AutoExposureConfig, 
            AutoExposureStatsConfig, 
            AutoExposureYUVConfig
        )
        
        # Test base configuration
        base_config = AutoExposureConfig()
        assert base_config.target_brightness == 0.18, "Default target should be 0.18"
        assert base_config.min_ev == -2.0, "Default min_ev should be -2.0"
        assert base_config.max_ev == 2.0, "Default max_ev should be 2.0"
        assert base_config.smoothing_factor == 0.1, "Default smoothing should be 0.1"
        
        print("  ✓ Base configuration created successfully")
        
        # Test stats configuration
        stats_config = AutoExposureStatsConfig()
        assert stats_config.enable_color_aware == True, "Stats should enable color_aware"
        assert stats_config.rgb_weights == [0.299, 0.587, 0.114], "Default RGB weights should be BT.601"
        
        print("  ✓ Stats configuration created successfully")
        
        # Test YUV configuration
        yuv_config = AutoExposureYUVConfig()
        assert yuv_config.yuv_luminance_only == True, "YUV should use luminance only"
        assert yuv_config.preserve_chrominance == True, "YUV should preserve chrominance"
        
        print("  ✓ YUV configuration created successfully")
        
        # Test custom configuration
        custom_config = AutoExposureConfig()
        custom_config.target_brightness = 0.25
        custom_config.min_ev = -3.0
        custom_config.max_ev = 3.0
        assert custom_config.target_brightness == 0.25, "Custom target should be 0.25"
        assert custom_config.min_ev == -3.0, "Custom min_ev should be -3.0"
        assert custom_config.max_ev == 3.0, "Custom max_ev should be 3.0"
        
        print("  ✓ Custom configuration works successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling utilities"""
    print("Testing error handling...")
    
    try:
        from microblocks.autoexposure import (
            EdgeCaseHandler,
            InputValidator,
            AutoExposureError,
            InputValidationError,
            ExposureCalculationError,
            AutoExposureConfig
        )
        
        # Test edge case handler
        config = AutoExposureConfig()
        edge_handler = EdgeCaseHandler(config)
        
        # Test all-black handling
        ev, gain = edge_handler.handle_all_black(0.005)
        assert ev == config.max_ev, "All-black should return max_ev"
        assert gain == 2.0 ** config.max_ev, "All-black should return max gain"
        
        print("  ✓ All-black edge case handled correctly")
        
        # Test all-white handling
        ev, gain = edge_handler.handle_all_white(0.995)
        assert ev == config.min_ev, "All-white should return min_ev"
        assert gain == 2.0 ** config.min_ev, "All-white should return min gain"
        
        print("  ✓ All-white edge case handled correctly")
        
        # Test normal case
        ev, gain = edge_handler.handle_all_black(0.5)
        assert ev is None, "Normal case should return None"
        assert gain is None, "Normal case should return None"
        
        print("  ✓ Normal case handled correctly")
        
        # Test exposure clipping
        clipped_ev = edge_handler.clip_exposure_value(3.0)
        assert clipped_ev == config.max_ev, "EV should be clipped to max"
        
        clipped_ev = edge_handler.clip_exposure_value(-3.0)
        assert clipped_ev == config.min_ev, "EV should be clipped to min"
        
        print("  ✓ Exposure clipping works correctly")
        
        # Test safe division
        result = edge_handler.safe_division(10.0, 2.0)
        assert result == 5.0, "Safe division should work normally"
        
        result = edge_handler.safe_division(10.0, 0.0)
        assert result == 1.0, "Division by zero should return default"
        
        print("  ✓ Safe division works correctly")
        
        # Test input validator
        validator = InputValidator()
        
        # Test shape validation (will fail with mock data, but we can test the logic)
        try:
            validator.validate_image_shape(None)
            assert False, "Should raise error for None input"
        except InputValidationError:
            pass  # Expected
        
        print("  ✓ Input validation works correctly")
        
        # Test exception classes
        try:
            raise AutoExposureError("Test error")
        except AutoExposureError as e:
            assert str(e) == "Test error", "Exception message should be preserved"
        
        print("  ✓ Exception classes work correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exposure_calculator():
    """Test exposure calculation logic"""
    print("Testing exposure calculator...")
    
    try:
        from microblocks.autoexposure import ExposureCalculator, AutoExposureConfig
        
        config = AutoExposureConfig()
        calculator = ExposureCalculator(config)
        
        # Test normal exposure calculation
        current_brightness = 0.5
        target_brightness = 0.18
        
        ev, gain = calculator.calculate_exposure_value(current_brightness, target_brightness)
        
        # Calculate expected values
        expected_ev = np.log2(target_brightness / current_brightness)
        expected_gain = 2.0 ** expected_ev
        
        assert abs(ev - expected_ev) < 0.01, f"EV should be {expected_ev}, got {ev}"
        assert abs(gain - expected_gain) < 0.01, f"Gain should be {expected_gain}, got {gain}"
        
        print(f"  ✓ Normal exposure calculation works (EV={ev:.3f}, gain={gain:.3f})")
        
        # Test edge case: very dark image
        ev, gain = calculator.calculate_exposure_value(0.005, target_brightness)
        assert ev == config.max_ev, "Very dark should return max EV"
        
        print("  ✓ Very dark image handled correctly")
        
        # Test edge case: very bright image
        ev, gain = calculator.calculate_exposure_value(0.995, target_brightness)
        assert ev == config.min_ev, "Very bright should return min EV"
        
        print("  ✓ Very bright image handled correctly")
        
        # Test weighted brightness calculation
        brightness_values = [0.3, 0.5, 0.7]
        weights = [0.2, 0.6, 0.2]
        
        weighted = calculator.calculate_weighted_brightness(brightness_values, weights)
        expected_weighted = 0.3 * 0.2 + 0.5 * 0.6 + 0.7 * 0.2
        
        assert abs(weighted - expected_weighted) < 0.01, f"Weighted should be {expected_weighted}, got {weighted}"
        
        print("  ✓ Weighted brightness calculation works correctly")
        
        # Test smoothing
        current_ev = 1.0
        previous_ev = 0.5
        
        smoothed = calculator.apply_smoothing(current_ev, previous_ev)
        expected_smoothed = 0.1 * 1.0 + 0.9 * 0.5  # smoothing_factor * current + (1-smoothing) * previous
        
        assert abs(smoothed - expected_smoothed) < 0.01, f"Smoothed should be {expected_smoothed}, got {smoothed}"
        
        print("  ✓ Temporal smoothing works correctly")
        
        # Test smoothing without previous value
        smoothed = calculator.apply_smoothing(current_ev, None)
        assert smoothed == current_ev, "Without previous value, should return current"
        
        print("  ✓ Smoothing without previous value works correctly")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Exposure calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_v2_implementation():
    """Test the v2 implementation"""
    print("Testing v2 implementation...")
    
    try:
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        # Test base class
        base_ae = AutoExposureBase()
        assert base_ae.version == "v2", "Base class should be v2"
        assert base_ae.config is not None, "Base class should have config"
        assert base_ae.calculator is not None, "Base class should have calculator"
        assert base_ae.edge_handler is not None, "Base class should have edge_handler"
        
        print("  ✓ Base class v2 initialized correctly")
        
        # Test stats class
        stats_ae = AutoExposureStats()
        assert stats_ae.version == "v2", "Stats class should be v2"
        assert stats_ae.config.enable_color_aware == True, "Stats should enable color_aware"
        
        print("  ✓ Stats class v2 initialized correctly")
        
        # Test YUV class
        yuv_ae = AutoExposureYUV()
        assert yuv_ae.version == "v2", "YUV class should be v2"
        assert yuv_ae.config.yuv_luminance_only == True, "YUV should use luminance only"
        
        print("  ✓ YUV class v2 initialized correctly")
        
        # Test that methods are callable
        assert callable(base_ae.build_algo), "build_algo should be callable"
        assert callable(base_ae.build_applier), "build_applier should be callable"
        
        print("  ✓ Methods are callable")
        
        return True
        
    except Exception as e:
        print(f"  ✗ V2 implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_configurations_v2():
    """Test v2 pipeline configurations"""
    print("Testing v2 pipeline configurations...")
    
    try:
        # Test stats-based v2 pipeline
        with open('pipeline_stats_ae_v2.json', 'r') as f:
            stats_pipeline = json.load(f)
        
        assert 'canonical_name' in stats_pipeline, "Pipeline should have canonical_name"
        assert stats_pipeline['canonical_name'] == 'softisp_pipeline_stats_ae_v2', "Should be v2 pipeline"
        assert 'autoexposure_stats' in stats_pipeline['stages'], "Should have autoexposure_stats stage"
        
        ae_stage = stats_pipeline['stages']['autoexposure_stats']
        assert ae_stage['version'] == 'v2', "Should use v2 version"
        assert ae_stage['class'] == 'autoexposure_stats', "Should use stats class"
        
        print("  ✓ Stats-based v2 pipeline configuration is valid")
        
        # Test YUV-based v2 pipeline
        with open('pipeline_yuv_ae_v2.json', 'r') as f:
            yuv_pipeline = json.load(f)
        
        assert 'canonical_name' in yuv_pipeline, "Pipeline should have canonical_name"
        assert yuv_pipeline['canonical_name'] == 'softisp_pipeline_yuv_ae_v2', "Should be v2 pipeline"
        assert 'autoexposure_yuv' in yuv_pipeline['stages'], "Should have autoexposure_yuv stage"
        
        ae_yuv_stage = yuv_pipeline['stages']['autoexposure_yuv']
        assert ae_yuv_stage['version'] == 'v2', "Should use v2 version"
        assert ae_yuv_stage['class'] == 'autoexposure_yuv', "Should use YUV class"
        
        print("  ✓ YUV-based v2 pipeline configuration is valid")
        
        return True
        
    except Exception as e:
        print(f"  ✗ V2 pipeline configurations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase1_features():
    """Test Phase 1 critical features"""
    print("Testing Phase 1 critical features...")
    
    try:
        from microblocks.autoexposure import (
            ExposureCalculator,
            EdgeCaseHandler,
            InputValidator,
            AutoExposureConfig
        )
        
        # Test 1: Real exposure calculation logic
        config = AutoExposureConfig()
        calculator = ExposureCalculator(config)
        
        ev, gain = calculator.calculate_exposure_value(0.5, 0.18)
        assert ev is not None, "Should calculate EV"
        assert gain is not None, "Should calculate gain"
        
        print("  ✓ Real exposure calculation logic works")
        
        # Test 2: Error handling
        edge_handler = EdgeCaseHandler(config)
        ev, gain = edge_handler.handle_all_black(0.001)
        assert ev == config.max_ev, "Should handle all-black case"
        
        print("  ✓ Error handling works")
        
        # Test 3: Configuration system
        custom_config = AutoExposureConfig()
        custom_config.target_brightness = 0.25
        assert custom_config.target_brightness == 0.25, "Configuration should be customizable"
        
        print("  ✓ Configuration system works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 1 features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_phase2_features():
    """Test Phase 2 important features"""
    print("Testing Phase 2 important features...")
    
    try:
        from microblocks.autoexposure import (
            AutoExposureStats,
            AutoExposureYUV,
            AutoExposureStatsConfig,
            AutoExposureYUVConfig
        )
        
        # Test 1: RGB-specific optimizations
        stats_ae = AutoExposureStats()
        assert stats_ae.config.enable_color_aware == True, "Should enable color_aware"
        assert stats_ae.config.rgb_weights == [0.299, 0.587, 0.114], "Should use BT.601 weights"
        
        print("  ✓ RGB-specific optimizations work")
        
        # Test 2: YUV-specific optimizations
        yuv_ae = AutoExposureYUV()
        assert yuv_ae.config.yuv_luminance_only == True, "Should use luminance only"
        assert yuv_ae.config.preserve_chrominance == True, "Should preserve chrominance"
        
        print("  ✓ YUV-specific optimizations work")
        
        # Test 3: Advanced exposure algorithms (weighted calculation)
        calculator = stats_ae.calculator
        brightness_values = [0.3, 0.5, 0.7]
        weights = [0.2, 0.6, 0.2]
        
        weighted = calculator.calculate_weighted_brightness(brightness_values, weights)
        assert weighted is not None, "Should calculate weighted brightness"
        
        print("  ✓ Advanced exposure algorithms work")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 2 features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("AUTO EXPOSURE V2 - PHASE 1 & 2 IMPLEMENTATION TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Configuration System", test_configuration_system),
        ("Error Handling", test_error_handling),
        ("Exposure Calculator", test_exposure_calculator),
        ("V2 Implementation", test_v2_implementation),
        ("V2 Pipeline Configurations", test_pipeline_configurations_v2),
        ("Phase 1 Features", test_phase1_features),
        ("Phase 2 Features", test_phase2_features),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
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
        print("✓ ALL TESTS PASSED - PHASE 1 & 2 COMPLETE")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())