#!/usr/bin/env python3

"""
Test LSC Displacement microblock structure without importing
"""

import sys
import os
import ast

def test_file_exists():
    """Test that all required files exist"""
    print("Testing file existence...")
    
    files = [
        "microblocks/lens/__init__.py",
        "microblocks/lens/lens_lcs_displacement.py",
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            return False
    
    return True

def test_python_syntax():
    """Test that Python files have valid syntax"""
    print("Testing Python syntax...")
    
    files = [
        "microblocks/lens/__init__.py",
        "microblocks/lens/lens_lcs_displacement.py",
    ]
    
    for file in files:
        try:
            with open(file, 'r') as f:
                code = f.read()
            ast.parse(code)
            print(f"  ✓ {file} has valid syntax")
        except SyntaxError as e:
            print(f"  ✗ {file} has syntax error: {e}")
            return False
    
    return True

def test_class_definitions():
    """Test that all classes are defined in the files"""
    print("Testing class definitions...")
    
    with open("microblocks/lens/lens_lcs_displacement.py", 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
    
    expected_classes = [
        "LensLCSDisplacementBase",
        "LensLCSDisplacementV1",
        "LensLCSDisplacementV2"
    ]
    
    for cls_name in expected_classes:
        if cls_name in classes:
            print(f"  ✓ {cls_name} is defined")
        else:
            print(f"  ✗ {cls_name} is missing")
            return False
    
    return True

def test_class_attributes():
    """Test that classes have required attributes"""
    print("Testing class attributes...")
    
    with open("microblocks/lens/lens_lcs_displacement.py", 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
    
    expected_attrs = {
        "LensLCSDisplacementBase": ["name", "family", "version"],
        "LensLCSDisplacementV1": ["name", "family", "version"],
        "LensLCSDisplacementV2": ["name", "family", "version"]
    }
    
    for cls_name, attrs in expected_attrs.items():
        if cls_name not in classes:
            print(f"  ✗ {cls_name} not found")
            continue
        
        found_attrs = set()
        for node in ast.walk(classes[cls_name]):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found_attrs.add(target.id)
        
        missing = set(attrs) - found_attrs
        if missing:
            print(f"  ✗ {cls_name} missing attributes: {missing}")
            return False
        else:
            print(f"  ✓ {cls_name} has all required attributes")
    
    return True

def test_method_definitions():
    """Test that classes have required methods"""
    print("Testing method definitions...")
    
    with open("microblocks/lens/lens_lcs_displacement.py", 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
    
    expected_methods = ["build_algo", "build_applier", "build_coordinator", "build_test_algo"]
    
    for cls_name in ["LensLCSDisplacementBase", "LensLCSDisplacementV1", "LensLCSDisplacementV2"]:
        if cls_name not in classes:
            print(f"  ✗ {cls_name} not found")
            continue
        
        found_methods = set()
        for node in ast.walk(classes[cls_name]):
            if isinstance(node, ast.FunctionDef):
                found_methods.add(node.name)
        
        missing = set(expected_methods) - found_methods
        if missing:
            print(f"  ✗ {cls_name} missing methods: {missing}")
            return False
        else:
            print(f"  ✓ {cls_name} has all required methods")
    
    return True

def test_family_names():
    """Test that family names are unique"""
    print("Testing family name uniqueness...")
    
    with open("microblocks/lens/lens_lcs_displacement.py", 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    families = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "family":
                            if isinstance(child.value, ast.Constant):
                                families[node.name] = child.value.value
    
    expected_families = {
        "LensLCSDisplacementBase": "lens_lcs_displacement_base",
        "LensLCSDisplacementV1": "lens_lcs_displacement_v1",
        "LensLCSDisplacementV2": "lens_lcs_displacement_v2"
    }
    
    for cls_name, expected_family in expected_families.items():
        if cls_name not in families:
            print(f"  ✗ {cls_name} missing family attribute")
            return False
        if families[cls_name] != expected_family:
            print(f"  ✗ {cls_name} has wrong family: {families[cls_name]} (expected {expected_family})")
            return False
        print(f"  ✓ {cls_name} has family: {families[cls_name]}")
    
    # Check uniqueness
    family_values = list(families.values())
    if len(family_values) != len(set(family_values)):
        print(f"  ✗ Family names are not unique")
        return False
    
    print(f"  ✓ All {len(families)} family names are unique")
    return True

def test_exports():
    """Test that __init__.py exports all classes"""
    print("Testing __init__.py exports...")
    
    with open("microblocks/lens/__init__.py", 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Find imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "lens_lcs_displacement" in node.module:
                for alias in node.names:
                    imports.append(alias.name)
    
    expected_exports = [
        "LensLCSDisplacementBase",
        "LensLCSDisplacementV1",
        "LensLCSDisplacementV2"
    ]
    
    for export in expected_exports:
        if export in imports:
            print(f"  ✓ {export} is imported")
        else:
            print(f"  ✗ {export} is not imported")
            return False
    
    # Check __all__
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        all_exports = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
                        for export in expected_exports:
                            if export not in all_exports:
                                print(f"  ✗ {export} not in __all__")
                                return False
                        print(f"  ✓ All {len(expected_exports)} classes in __all__")
                        return True
    
    print(f"  ⚠ __all__ not found")
    return True

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("LCS DISPLACEMENT MICROBLOCK STRUCTURE TEST")
    print("=" * 70)
    print()
    
    tests = [
        ("File Existence", test_file_exists),
        ("Python Syntax", test_python_syntax),
        ("Class Definitions", test_class_definitions),
        ("Class Attributes", test_class_attributes),
        ("Method Definitions", test_method_definitions),
        ("Family Names", test_family_names),
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
        print("Available LCS Displacement Algorithms:")
        print("  1. LensLCSDisplacementBase (family: lens_lcs_displacement_base)")
        print("  2. LensLCSDisplacementV1 (family: lens_lcs_displacement_v1)")
        print("  3. LensLCSDisplacementV2 (family: lens_lcs_displacement_v2)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
