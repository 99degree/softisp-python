#!/usr/bin/env python3

"""
Comprehensive test script for auto exposure microblock correctness
Tests the structure, method signatures, and expected outputs without requiring ONNX
"""

import sys
import os
import json

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_class_structure():
    """Test that the classes have the correct structure"""
    print("Testing class structure...")
    
    try:
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        # Test base class
        assert hasattr(AutoExposureBase, 'name'), "AutoExposureBase missing 'name' attribute"
        assert hasattr(AutoExposureBase, 'version'), "AutoExposureBase missing 'version' attribute"
        assert hasattr(AutoExposureBase, 'build_algo'), "AutoExposureBase missing 'build_algo' method"
        assert hasattr(AutoExposureBase, 'build_applier'), "AutoExposureBase missing 'build_applier' method"
        assert hasattr(AutoExposureBase, 'build_coordinator'), "AutoExposureBase missing 'build_coordinator' method"
        assert hasattr(AutoExposureBase, 'build_test_algo'), "AutoExposureBase missing 'build_test_algo' method"
        
        # Test stats class
        assert hasattr(AutoExposureStats, 'name'), "AutoExposureStats missing 'name' attribute"
        assert hasattr(AutoExposureStats, 'version'), "AutoExposureStats missing 'version' attribute"
        assert AutoExposureStats.name == "autoexposure_stats", f"AutoExposureStats name should be 'autoexposure_stats', got '{AutoExposureStats.name}'"
        assert AutoExposureStats.version == "v1", f"AutoExposureStats version should be 'v1', got '{AutoExposureStats.version}'"
        
        # Test YUV class
        assert hasattr(AutoExposureYUV, 'name'), "AutoExposureYUV missing 'name' attribute"
        assert hasattr(AutoExposureYUV, 'version'), "AutoExposureYUV missing 'version' attribute"
        assert AutoExposureYUV.name == "autoexposure_yuv", f"AutoExposureYUV name should be 'autoexposure_yuv', got '{AutoExposureYUV.name}'"
        assert AutoExposureYUV.version == "v1", f"AutoExposureYUV version should be 'v1', got '{AutoExposureYUV.version}'"
        
        print("✓ Class structure test passed")
        return True
        
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False

def test_method_signatures():
    """Test that the methods have correct signatures"""
    print("Testing method signatures...")
    
    try:
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        # Create instances
        ae_base = AutoExposureBase()
        ae_stats = AutoExposureStats()
        ae_yuv = AutoExposureYUV()
        
        # Test that methods are callable
        assert callable(ae_base.build_algo), "build_algo should be callable"
        assert callable(ae_base.build_applier), "build_applier should be callable"
        assert callable(ae_base.build_coordinator), "build_coordinator should be callable"
        assert callable(ae_base.build_test_algo), "build_test_algo should be callable"
        
        # Test that methods accept expected parameters
        # We can't test the actual execution without ONNX, but we can test the signatures
        import inspect
        
        # Check build_algo signature
        algo_sig = inspect.signature(ae_base.build_algo)
        assert 'stage' in algo_sig.parameters, "build_algo should have 'stage' parameter"
        assert 'prev_stages' in algo_sig.parameters, "build_algo should have 'prev_stages' parameter"
        
        # Check build_applier signature
        applier_sig = inspect.signature(ae_base.build_applier)
        assert 'stage' in applier_sig.parameters, "build_applier should have 'stage' parameter"
        assert 'prev_stages' in applier_sig.parameters, "build_applier should have 'prev_stages' parameter"
        
        print("✓ Method signatures test passed")
        return True
        
    except Exception as e:
        print(f"✗ Method signatures test failed: {e}")
        return False

def test_pipeline_configurations():
    """Test that the pipeline configurations are valid"""
    print("Testing pipeline configurations...")
    
    try:
        # Test stats-based pipeline
        with open('pipeline_stats_ae.json', 'r') as f:
            stats_pipeline = json.load(f)
        
        assert 'canonical_name' in stats_pipeline, "Pipeline should have canonical_name"
        assert 'stages' in stats_pipeline, "Pipeline should have stages"
        assert 'autoexposure_stats' in stats_pipeline['stages'], "Pipeline should have autoexposure_stats stage"
        
        # Check autoexposure_stats stage configuration
        ae_stage = stats_pipeline['stages']['autoexposure_stats']
        assert ae_stage['class'] == 'autoexposure_stats', "Class should be autoexposure_stats"
        assert ae_stage['version'] == 'v1', "Version should be v1"
        assert 'demosaic' in ae_stage['inputs'], "Should depend on demosaic stage"
        
        # Test YUV-based pipeline
        with open('pipeline_yuv_ae.json', 'r') as f:
            yuv_pipeline = json.load(f)
        
        assert 'canonical_name' in yuv_pipeline, "Pipeline should have canonical_name"
        assert 'stages' in yuv_pipeline, "Pipeline should have stages"
        assert 'autoexposure_yuv' in yuv_pipeline['stages'], "Pipeline should have autoexposure_yuv stage"
        
        # Check autoexposure_yuv stage configuration
        ae_yuv_stage = yuv_pipeline['stages']['autoexposure_yuv']
        assert ae_yuv_stage['class'] == 'autoexposure_yuv', "Class should be autoexposure_yuv"
        assert ae_yuv_stage['version'] == 'v1', "Version should be v1"
        assert 'yuv' in ae_yuv_stage['inputs'], "Should depend on yuv stage"
        
        print("✓ Pipeline configurations test passed")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline configurations test failed: {e}")
        return False

def test_expected_outputs():
    """Test that the expected outputs are correctly defined"""
    print("Testing expected outputs...")
    
    try:
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        # Test that classes can be instantiated
        ae_base = AutoExposureBase()
        ae_stats = AutoExposureStats()
        ae_yuv = AutoExposureYUV()
        
        # Test that methods return something (even if it's a mock)
        result_algo = ae_base.build_algo("test_stage", ["prev_stage"])
        assert result_algo is not None, "build_algo should return a result"
        
        result_applier = ae_base.build_applier("test_stage", ["prev_stage"])
        assert result_applier is not None, "build_applier should return a result"
        
        print("✓ Expected outputs test passed")
        return True
        
    except Exception as e:
        print(f"✗ Expected outputs test failed: {e}")
        return False

def test_architecture_compliance():
    """Test that the implementation follows the ISP architecture"""
    print("Testing architecture compliance...")
    
    try:
        from microblocks.autoexposure import AutoExposureBase, AutoExposureStats, AutoExposureYUV
        
        # Test that both approaches inherit from the same base
        assert issubclass(AutoExposureStats, AutoExposureBase), "AutoExposureStats should inherit from AutoExposureBase"
        assert issubclass(AutoExposureYUV, AutoExposureBase), "AutoExposureYUV should inherit from AutoExposureBase"
        
        # Test that both have the same applier interface
        ae_stats = AutoExposureStats()
        ae_yuv = AutoExposureYUV()
        
        # Both should have the same method signatures
        import inspect
        stats_applier_sig = inspect.signature(ae_stats.build_applier)
        yuv_applier_sig = inspect.signature(ae_yuv.build_applier)
        
        assert str(stats_applier_sig) == str(yuv_applier_sig), "Both approaches should have the same applier signature"
        
        print("✓ Architecture compliance test passed")
        return True
        
    except Exception as e:
        print(f"✗ Architecture compliance test failed: {e}")
        return False

def test_documentation():
    """Test that documentation files exist and are valid"""
    print("Testing documentation...")
    
    try:
        # Check that README exists
        assert os.path.exists('microblocks/autoexposure/README.md'), "README.md should exist"
        
        # Check that implementation documentation exists
        assert os.path.exists('AUTOEXPOSURE_IMPLEMENTATION.md'), "AUTOEXPOSURE_IMPLEMENTATION.md should exist"
        
        # Check that summary exists
        assert os.path.exists('IMPLEMENTATION_SUMMARY.txt'), "IMPLEMENTATION_SUMMARY.txt should exist"
        
        # Read and validate README content
        with open('microblocks/autoexposure/README.md', 'r') as f:
            readme_content = f.read()
        
        assert 'autoexposure_stats' in readme_content, "README should mention autoexposure_stats"
        assert 'autoexposure_yuv' in readme_content, "README should mention autoexposure_yuv"
        assert 'Algorithm' in readme_content, "README should mention Algorithm"
        assert 'Applier' in readme_content, "README should mention Applier"
        
        print("✓ Documentation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 70)
    print("AUTO EXPOSURE MICROBLOCK CORRECTNESS TESTS")
    print("=" * 70)
    print()
    
    tests = [
        ("Class Structure", test_class_structure),
        ("Method Signatures", test_method_signatures),
        ("Pipeline Configurations", test_pipeline_configurations),
        ("Expected Outputs", test_expected_outputs),
        ("Architecture Compliance", test_architecture_compliance),
        ("Documentation", test_documentation),
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
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())