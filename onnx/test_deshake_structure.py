#!/usr/bin/env python3
"""
Deshake Microblock Structure Test

Tests the structure of all deshake microblock implementations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_file_existence():
    """Test that all required files exist."""
    print("Testing file existence...")
    
    files = [
        'microblocks/deshake/__init__.py',
        'microblocks/deshake/deshake.py',
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} does not exist")
            return False
    
    return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    files = [
        'microblocks/deshake/__init__.py',
        'microblocks/deshake/deshake.py',
    ]
    
    for file in files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✓ {file} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {file} has syntax error: {e}")
            return False
    
    return True


def test_class_definitions():
    """Test that all classes are defined."""
    print("\nTesting class definitions...")
    
    try:
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        
        classes = {
            'DeshakeBase': DeshakeBase,
            'DeshakeV1': DeshakeV1,
            'DeshakeV2': DeshakeV2,
        }
        
        for name, cls in classes.items():
            print(f"  ✓ {name} is defined")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_class_attributes():
    """Test that all classes have required attributes."""
    print("\nTesting class attributes...")
    
    try:
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        
        classes = {
            'DeshakeBase': DeshakeBase,
            'DeshakeV1': DeshakeV1,
            'DeshakeV2': DeshakeV2,
        }
        
        required_attrs = ['name', 'family', 'version']
        
        for name, cls in classes.items():
            missing = [attr for attr in required_attrs if not hasattr(cls, attr)]
            if missing:
                print(f"  ✗ {name} missing attributes: {missing}")
                return False
            else:
                print(f"  ✓ {name} has all required attributes")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_method_definitions():
    """Test that all classes have required methods."""
    print("\nTesting method definitions...")
    
    try:
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        
        classes = {
            'DeshakeBase': DeshakeBase,
            'DeshakeV1': DeshakeV1,
            'DeshakeV2': DeshakeV2,
        }
        
        required_methods = ['build_algo', 'build_applier', 'build_coordinator', 'build_test_algo']
        
        for name, cls in classes.items():
            missing = [method for method in required_methods if not hasattr(cls, method)]
            if missing:
                print(f"  ✗ {name} missing methods: {missing}")
                return False
            else:
                print(f"  ✓ {name} has all required methods")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_family_name_uniqueness():
    """Test that all classes have unique family names."""
    print("\nTesting family name uniqueness...")
    
    try:
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        
        classes = {
            'DeshakeBase': DeshakeBase,
            'DeshakeV1': DeshakeV1,
            'DeshakeV2': DeshakeV2,
        }
        
        families = {}
        for name, cls in classes.items():
            family = cls.family
            print(f"  ✓ {name} has family: {family}")
            if family in families:
                print(f"  ✗ Family name '{family}' is not unique!")
                return False
            families[family] = name
        
        print(f"  ✓ All {len(families)} family names are unique")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_exports():
    """Test that __init__.py exports all classes."""
    print("\nTesting __init__.py exports...")
    
    try:
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        from microblocks.deshake import __all__ as exports
        
        expected = ['DeshakeBase', 'DeshakeV1', 'DeshakeV2']
        
        for name in expected:
            if name in exports:
                print(f"  ✓ {name} is in __all__")
            else:
                print(f"  ✗ {name} is not in __all__")
                return False
        
        print(f"  ✓ All {len(expected)} classes in __all__")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("DESHAKE MICROBLOCK STRUCTURE TEST")
    print("=" * 70)
    
    tests = [
        test_file_existence,
        test_python_syntax,
        test_class_definitions,
        test_class_attributes,
        test_method_definitions,
        test_family_name_uniqueness,
        test_exports,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "File Existence",
        "Python Syntax",
        "Class Definitions",
        "Class Attributes",
        "Method Definitions",
        "Family Names",
        "Exports",
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<50} {status}")
    
    total = len(results)
    passed = sum(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        
        print("\nAvailable Deshake Algorithms:")
        from microblocks.deshake import DeshakeBase, DeshakeV1, DeshakeV2
        
        algorithms = [
            (DeshakeBase, "Global motion compensation"),
            (DeshakeV1, "Grid-based motion compensation"),
            (DeshakeV2, "Feature-based with RANSAC"),
        ]
        
        for i, (cls, description) in enumerate(algorithms, 1):
            print(f"  {i}. {cls.name} (family: {cls.family})")
            print(f"     {description}")
        
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())