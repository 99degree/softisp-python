#!/usr/bin/env python3

"""
Test all auto exposure algorithms using importlib to bypass ONNX dependency
"""

import sys
import os
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a Python module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_class_attributes():
    """Test that all classes have correct attributes"""
    print("Testing class attributes...")
    
    try:
        ae_v2 = load_module_from_file(
            "ae_v2",
            "microblocks/autoexposure/autoexposure_v2.py"
        )
        
        # Test Simple
        simple = ae_v2.AutoExposureSimple
        assert simple.name == "autoexposure_simple"
        assert simple.family == "autoexposure_simple"
        assert simple.version == "v2"
        assert simple.target_brightness == 0.18
        assert simple.min_ev == -2.0
        assert simple.max_ev == 2.0
        print("  ✓ AutoExposureSimple attributes correct")
        
        # Test Stats
        stats = ae_v2.AutoExposureStats
        assert stats.name == "autoexposure_stats"
        assert stats.family == "autoexposure_stats"
        assert stats.version == "v2"
        assert stats.target_brightness == 0.18
        assert stats.rgb_weights == [0.299, 0.587, 0.114]
        print("  ✓ AutoExposureStats attributes correct")
        
        # Test YUV
        yuv = ae_v2.AutoExposureYUV
        assert yuv.name == "autoexposure_yuv"
        assert yuv.family == "autoexposure_yuv"
        assert yuv.version == "v2"
        assert yuv.target_brightness == 0.18
        print("  ✓ AutoExposureYUV attributes correct")
        
        # Test Histogram
        hist = ae_v2.AutoExposureHistogram
        assert hist.name == "autoexposure_histogram"
        assert hist.family == "autoexposure_histogram"
        assert hist.version == "v2"
        assert hist.histogram_bins == 256
        assert hist.percentile == 50
        print("  ✓ AutoExposureHistogram attributes correct")
        
        # Test MultiZone
        multi = ae_v2.AutoExposureMultiZone
        assert multi.name == "autoexposure_multizone"
        assert multi.family == "autoexposure_multizone"
        assert multi.version == "v2"
        assert multi.zone_weights == [0.5, 0.7, 1.0, 0.7, 0.5]
        print("  ✓ AutoExposureMultiZone attributes correct")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_existence():
    """Test that all classes have required methods"""
    print("Testing method existence...")
    
    try:
        ae_v2 = load_module_from_file(
            "ae_v2_methods",
            "microblocks/autoexposure/autoexposure_v2.py"
        )
        
        classes = [
            ae_v2.AutoExposureSimple,
            ae_v2.AutoExposureStats,
            ae_v2.AutoExposureYUV,
            ae_v2.AutoExposureHistogram,
            ae_v2.AutoExposureMultiZone
        ]
        
        for cls in classes:
            assert hasattr(cls, 'build_algo'), f"{cls.name} missing build_algo"
            assert hasattr(cls, 'build_applier'), f"{cls.name} missing build_applier"
            assert hasattr(cls, 'build_coordinator'), f"{cls.name} missing build_coordinator"
            assert hasattr(cls, 'build_test_algo'), f"{cls.name} missing build_test_algo"
            assert callable(cls.build_algo), f"{cls.name} build_algo not callable"
            assert callable(cls.build_applier), f"{cls.name} build_applier not callable"
            print(f"  ✓ {cls.name} has all required methods")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_family_uniqueness():
    """Test that all classes have unique family names"""
    print("Testing family name uniqueness...")
    
    try:
        ae_v2 = load_module_from_file(
            "ae_v2_family",
            "microblocks/autoexposure/autoexposure_v2.py"
        )
        
        families = [
            ae_v2.AutoExposureSimple.family,
            ae_v2.AutoExposureStats.family,
            ae_v2.AutoExposureYUV.family,
            ae_v2.AutoExposureHistogram.family,
            ae_v2.AutoExposureMultiZone.family
        ]
        
        assert len(families) == len(set(families)), "Family names are not unique"
        print(f"  ✓ All {len(families)} classes have unique family names")
        
        for family in families:
            print(f"    - {family}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_defaults():
    """Test that all classes have sensible default parameters"""
    print("Testing parameter defaults...")
    
    try:
        ae_v2 = load_module_from_file(
            "ae_v2_params",
            "microblocks/autoexposure/autoexposure_v2.py"
        )
        
        classes = [
            ("AutoExposureSimple", ae_v2.AutoExposureSimple),
            ("AutoExposureStats", ae_v2.AutoExposureStats),
            ("AutoExposureYUV", ae_v2.AutoExposureYUV),
            ("AutoExposureHistogram", ae_v2.AutoExposureHistogram),
            ("AutoExposureMultiZone", ae_v2.AutoExposureMultiZone)
        ]
        
        for name, cls in classes:
            # Check common parameters
            assert hasattr(cls, 'target_brightness'), f"{name} missing target_brightness"
            assert hasattr(cls, 'min_ev'), f"{name} missing min_ev"
            assert hasattr(cls, 'max_ev'), f"{name} missing max_ev"
            
            # Check values are sensible
            assert 0 < cls.target_brightness < 1, f"{name} target_brightness out of range"
            assert cls.min_ev < 0, f"{name} min_ev should be negative"
            assert cls.max_ev > 0, f"{name} max_ev should be positive"
            assert cls.min_ev < cls.max_ev, f"{name} min_ev should be less than max_ev"
            
            print(f"  ✓ {name} has valid parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_no_init_required():
    """Test that classes work without __init__"""
    print("Testing no __init__ required...")
    
    try:
        ae_v2 = load_module_from_file(
            "ae_v2_noinit",
            "microblocks/autoexposure/autoexposure_v2.py"
        )
        
        # Test that we can access class attributes without instantiation
        simple = ae_v2.AutoExposureSimple
        assert simple.target_brightness == 0.18
        assert simple.min_ev == -2.0
        assert simple.max_ev == 2.0
        print("  ✓ AutoExposureSimple works without instantiation")
        
        stats = ae_v2.AutoExposureStats
        assert stats.target_brightness == 0.18
        assert stats.rgb_weights == [0.299, 0.587, 0.114]
        print("  ✓ AutoExposureStats works without instantiation")
        
        yuv = ae_v2.AutoExposureYUV
        assert yuv.target_brightness == 0.18
        print("  ✓ AutoExposureYUV works without instantiation")
        
        hist = ae_v2.AutoExposureHistogram
        assert hist.histogram_bins == 256
        print("  ✓ AutoExposureHistogram works without instantiation")
        
        multi = ae_v2.AutoExposureMultiZone
        assert multi.zone_weights == [0.5, 0.7, 1.0, 0.7, 0.5]
        print("  ✓ AutoExposureMultiZone works without instantiation")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exports():
    """Test that __init__.py exports all classes"""
    print("Testing __init__.py exports...")
    
    try:
        ae_init = load_module_from_file(
            "ae_init",
            "microblocks/autoexposure/__init__.py"
        )
        
        expected_exports = [
            'AutoExposureSimple',
            'AutoExposureStats',
            'AutoExposureYUV',
            'AutoExposureHistogram',
            'AutoExposureMultiZone'
        ]
        
        for export in expected_exports:
            assert hasattr(ae_init, export), f"Missing export: {export}"
            print(f"  ✓ {export} is exported")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("AUTO EXPOSURE ALGORITHMS TEST")
    print("=" * 70)
    print()
    
    tests = [
        ("Class Attributes", test_class_attributes),
        ("Method Existence", test_method_existence),
        ("Family Uniqueness", test_family_uniqueness),
        ("Parameter Defaults", test_parameter_defaults),
        ("No __init__ Required", test_no_init_required),
        ("Exports", test_exports),
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
        print()
        print("Available Algorithms:")
        print("  1. AutoExposureSimple (LOW computation)")
        print("  2. AutoExposureStats (MEDIUM computation)")
        print("  3. AutoExposureYUV (MEDIUM computation)")
        print("  4. AutoExposureHistogram (HIGH computation)")
        print("  5. AutoExposureMultiZone (HIGH computation)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
