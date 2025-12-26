#!/usr/bin/env python
"""
Integration test script to verify all core radipop_utils imports work correctly.

This script tests that all required dependencies are properly installed and
that the core modules can be imported without errors.

Usage:
    python test_imports.py
"""

import sys

def test_core_imports():
    """Test core module imports that should always work."""
    print("Testing core module imports...")
    
    try:
        import radipop_utils
        print("  ✓ radipop_utils")
    except ImportError as e:
        print(f"  ✗ radipop_utils: {e}")
        return False
    
    try:
        import radipop_utils.visualization
        print("  ✓ radipop_utils.visualization")
    except ImportError as e:
        print(f"  ✗ radipop_utils.visualization: {e}")
        return False
    
    try:
        import radipop_utils.features
        print("  ✓ radipop_utils.features")
    except ImportError as e:
        print(f"  ✗ radipop_utils.features: {e}")
        return False
    
    try:
        import radipop_utils.inference
        print("  ✓ radipop_utils.inference")
    except ImportError as e:
        print(f"  ✗ radipop_utils.inference: {e}")
        return False
    
    try:
        import radipop_utils.data
        print("  ✓ radipop_utils.data")
    except ImportError as e:
        print(f"  ✗ radipop_utils.data: {e}")
        return False
    
    try:
        import radipop_utils.utils
        print("  ✓ radipop_utils.utils")
    except ImportError as e:
        print(f"  ✗ radipop_utils.utils: {e}")
        return False
    
    return True

def test_optional_imports():
    """Test optional module imports (torch/TotalSegmentator)."""
    print("\nTesting optional module imports...")
    
    torch_available = False
    try:
        import totalsegmentator
        import totalsegmentator.python_api
        print("  ✓ totalsegmentator (optional)")
        torch_available = True
    except ImportError as e:
        print(f"  ⊘ totalsegmentator (optional, not installed): {e}")
    
    try:
        import radipop_utils.inference_via_total_segmentor
        if torch_available:
            print("  ✓ radipop_utils.inference_via_total_segmentor")
        else:
            print("  ⊘ radipop_utils.inference_via_total_segmentor (optional, requires torch)")
    except ImportError as e:
        print(f"  ⊘ radipop_utils.inference_via_total_segmentor (optional): {e}")
    
    return True  # Optional imports don't fail the test

def test_dicom2nifti_import():
    """Test optional dicom2nifti import."""
    print("\nTesting optional dicom2nifti import...")
    
    try:
        import dicom2nifti
        print("  ✓ dicom2nifti (optional)")
    except ImportError as e:
        print(f"  ⊘ dicom2nifti (optional, not installed): {e}")
    
    return True  # Optional import doesn't fail the test

def main():
    """Run all import tests."""
    print("=" * 60)
    print("RADIPOP Utils - Import Test")
    print("=" * 60)
    print()
    
    core_success = test_core_imports()
    test_optional_imports()
    test_dicom2nifti_import()
    
    print()
    print("=" * 60)
    if core_success:
        print("✓ All core imports successful!")
        print("\nNote: Optional dependencies (torch, dicom2nifti) are not required")
        print("      for core functionality.")
        return 0
    else:
        print("✗ Some core imports failed!")
        print("\nPlease check your installation and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

