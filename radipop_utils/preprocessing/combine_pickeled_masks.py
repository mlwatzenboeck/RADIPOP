from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import numpy as np

try:
    import pydicom  # type: ignore
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk  # type: ignore
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False


def _extract_dcm_number(dcm_name: str) -> int:
    """
    Extract the number from a DICOM filename like "IM-0001-0001.dcm".
    Returns the number before .dcm (e.g., 1 for "IM-0001-0001.dcm").
    """
    match = re.search(r"IM-\d+-(\d+)\.dcm$", dcm_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract number from DICOM filename: {dcm_name}")


def _extract_mask_number(mask_filename: str) -> int:
    """
    Extract the number from a mask filename like "0.p" or "123.p".
    """
    match = re.search(r"^(\d+)\.p$", mask_filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract number from mask filename: {mask_filename}")


def _read_dicom_metadata(dcm_path: Path) -> Tuple[Optional[Tuple[float, float, float]], 
                                                   Optional[Tuple[float, float, float]],
                                                   Optional[np.ndarray]]:
    """
    Read DICOM file and extract spacing, origin, and direction.
    
    Returns:
        spacing: (x, y, z) spacing in mm, or None if not available
        origin: (x, y, z) origin in mm, or None if not available
        direction: 3x3 direction matrix, or None if not available
    """
    if not PYDICOM_AVAILABLE:
        return None, None, None
    
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    except Exception as e:
        print(f"Warning: Could not read DICOM file {dcm_path}: {e}")
        return None, None, None
    
    # Extract spacing
    spacing = None
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    slice_thickness = getattr(ds, "SliceThickness", None)
    spacing_between_slices = getattr(ds, "SpacingBetweenSlices", None)
    
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        # PixelSpacing is (row, column) = (y, x) in DICOM
        # For SimpleITK, we need (x, y, z)
        x_spacing = float(pixel_spacing[1])  # column spacing
        y_spacing = float(pixel_spacing[0])  # row spacing
        
        # Z spacing: prefer SpacingBetweenSlices, then SliceThickness
        if spacing_between_slices is not None:
            z_spacing = float(spacing_between_slices)
        elif slice_thickness is not None:
            z_spacing = float(slice_thickness)
        else:
            z_spacing = 1.0  # default
        
        spacing = (x_spacing, y_spacing, z_spacing)
    
    # Extract origin
    origin = None
    image_position_patient = getattr(ds, "ImagePositionPatient", None)
    if image_position_patient is not None and len(image_position_patient) >= 3:
        origin = (
            float(image_position_patient[0]),
            float(image_position_patient[1]),
            float(image_position_patient[2])
        )
    
    # Extract direction
    direction = None
    image_orientation_patient = getattr(ds, "ImageOrientationPatient", None)
    if image_orientation_patient is not None and len(image_orientation_patient) >= 6:
        # ImageOrientationPatient is (row_x, row_y, row_z, col_x, col_y, col_z)
        # We need to construct a 3x3 direction matrix
        row_cosines = np.array([
            float(image_orientation_patient[0]),
            float(image_orientation_patient[1]),
            float(image_orientation_patient[2])
        ])
        col_cosines = np.array([
            float(image_orientation_patient[3]),
            float(image_orientation_patient[4]),
            float(image_orientation_patient[5])
        ])
        # Slice normal is cross product
        slice_normal = np.cross(row_cosines, col_cosines)
        
        # Direction matrix: [col_x, row_x, slice_x, col_y, row_y, slice_y, col_z, row_z, slice_z]
        direction = np.array([
            col_cosines[0], row_cosines[0], slice_normal[0],
            col_cosines[1], row_cosines[1], slice_normal[1],
            col_cosines[2], row_cosines[2], slice_normal[2]
        ])
    
    return spacing, origin, direction


def combine_pickled_masks(folder_name: str) -> None:
    """
    Combine pickle mask files into NIfTI files organized by acquisition time.
    
    For each folder in {folder_name}/by_acqtime/, loads all .p files from masks/
    subfolder, validates consecutive numbering, and combines them into a single
    NIfTI file with metadata from DICOM files.
    """
    if not SIMPLEITK_AVAILABLE:
        raise ImportError(
            "SimpleITK is required for combining masks. "
            "Install it with: pip install SimpleITK"
        )
    
    root = Path(folder_name).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    
    by_acqtime_dir = root / "by_acqtime"
    if not by_acqtime_dir.is_dir():
        raise NotADirectoryError(
            f"by_acqtime directory not found in {root}. "
            "Run dcm_seperation.py first."
        )
    
    # Get all acquisition time folders
    acq_time_folders = [
        d for d in by_acqtime_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    
    if len(acq_time_folders) == 0:
        raise RuntimeError(f"No acquisition time folders found in: {by_acqtime_dir}")
    
    print(f"Found {len(acq_time_folders)} acquisition time folder(s)")
    
    for acq_folder in acq_time_folders:
        print(f"\nProcessing {acq_folder.name}...")
        
        metadata_csv = acq_folder / "metadata.csv"
        masks_dir = acq_folder / "masks"
        
        if not metadata_csv.exists():
            print(f"Warning: metadata.csv not found in {acq_folder}, skipping...")
            continue
        
        if not masks_dir.exists() or not masks_dir.is_dir():
            print(f"Warning: masks/ directory not found in {acq_folder}, skipping...")
            continue
        
        # Read metadata.csv to get expected DICOM files
        with metadata_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) == 0:
            print(f"Warning: metadata.csv is empty in {acq_folder}, skipping...")
            continue
        
        # Extract expected DICOM numbers (with off-by-one correction)
        expected_dcm_indices = []
        dcm_paths = {}
        for row in rows:
            dcm_name = row.get("dcm_name", "")
            if dcm_name:
                try:
                    dcm_number = _extract_dcm_number(dcm_name)
                    dcm_index = dcm_number + 1  # off-by-one correction (DICOM number 288 corresponds to file 289)
                    expected_dcm_indices.append(dcm_index)
                    
                    # Store path to DICOM file for metadata extraction
                    dcm_src = row.get("dcm_src", "")
                    if dcm_src:
                        dcm_paths[dcm_index] = Path(dcm_src)
                except (ValueError, KeyError):
                    continue
        
        if len(expected_dcm_indices) == 0:
            print(f"Warning: No valid DICOM files found in metadata.csv, skipping...")
            continue
        
        expected_dcm_indices = sorted(set(expected_dcm_indices))
        min_expected = min(expected_dcm_indices)
        max_expected = max(expected_dcm_indices)
        
        # Load all .p files from masks/ directory
        mask_files = [
            f for f in masks_dir.iterdir()
            if f.is_file() and f.suffix == ".p" and not f.name.startswith(".")
        ]
        
        if len(mask_files) == 0:
            print(f"Warning: No .p files found in {masks_dir}, skipping...")
            continue
        
        # Extract numbers from mask filenames
        mask_indices = []
        mask_files_dict = {}
        for mask_file in mask_files:
            try:
                idx = _extract_mask_number(mask_file.name)
                mask_indices.append(idx)
                mask_files_dict[idx] = mask_file
            except ValueError:
                print(f"Warning: Skipping invalid mask filename: {mask_file.name}")
                continue
        
        if len(mask_indices) == 0:
            print(f"Warning: No valid mask files found, skipping...")
            continue
        
        mask_indices = sorted(mask_indices)
        min_mask = min(mask_indices)
        max_mask = max(mask_indices)
        
        # Check for missing beginning/end files (compared to DICOM files)
        missing_beginning = []
        missing_end = []
        if min_mask > min_expected:
            missing_beginning = list(range(min_expected, min_mask))
            print(f"Warning: Missing mask files at beginning: {missing_beginning}")
        if max_mask < max_expected:
            missing_end = list(range(max_mask + 1, max_expected + 1))
            print(f"Warning: Missing mask files at end: {missing_end}")
        
        # Validate consecutive numbering (no gaps)
        expected_mask_indices = list(range(min_mask, max_mask + 1))
        if mask_indices != expected_mask_indices:
            missing_in_middle = set(expected_mask_indices) - set(mask_indices)
            raise RuntimeError(
                f"Mask files are not consecutive. Missing indices: {sorted(missing_in_middle)}. "
                f"Found indices: {mask_indices}"
            )
        
        print(f"Found {len(mask_indices)} consecutive mask files (indices {min_mask} to {max_mask})")
        
        # Load all mask files
        masks = []
        for idx in mask_indices:
            mask_file = mask_files_dict[idx]
            try:
                with mask_file.open("rb") as f:
                    mask = pickle.load(f)
                masks.append(mask)
            except Exception as e:
                raise RuntimeError(f"Error loading mask file {mask_file}: {e}")
        
        # Stack masks into 3D array
        mask_array = np.stack(masks, axis=0)
        print(f"Combined mask shape: {mask_array.shape}")
        
        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(mask_array)
        
        # Try to extract metadata from a DICOM file
        # Use the first available DICOM file for metadata
        dcm_path_for_metadata = None
        for idx in mask_indices:
            if idx in dcm_paths and dcm_paths[idx].exists():
                dcm_path_for_metadata = dcm_paths[idx]
                break
        
        if dcm_path_for_metadata is None:
            # Try any DICOM file from the metadata
            for dcm_path in dcm_paths.values():
                if dcm_path.exists():
                    dcm_path_for_metadata = dcm_path
                    break
        
        if dcm_path_for_metadata is not None:
            spacing, origin, direction = _read_dicom_metadata(dcm_path_for_metadata)
            
            if spacing is not None:
                sitk_image.SetSpacing(spacing)
                print(f"Set spacing: {spacing}")
            else:
                # Default spacing
                sitk_image.SetSpacing((1.0, 1.0, 1.0))
                print("Warning: Could not extract spacing from DICOM, using default (1.0, 1.0, 1.0)")
            
            if origin is not None:
                sitk_image.SetOrigin(origin)
                print(f"Set origin: {origin}")
            else:
                sitk_image.SetOrigin((0.0, 0.0, 0.0))
                print("Warning: Could not extract origin from DICOM, using default (0.0, 0.0, 0.0)")
            
            if direction is not None:
                sitk_image.SetDirection(direction.flatten().tolist())
                print(f"Set direction matrix")
            else:
                # Identity matrix
                sitk_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                print("Warning: Could not extract direction from DICOM, using identity matrix")
        else:
            print("Warning: No DICOM file found for metadata extraction, using defaults")
            sitk_image.SetSpacing((1.0, 1.0, 1.0))
            sitk_image.SetOrigin((0.0, 0.0, 0.0))
            sitk_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        
        # Save as NIfTI
        output_path = masks_dir / "mask.nii.gz"
        sitk.WriteImage(sitk_image, str(output_path))
        print(f"Saved combined mask to: {output_path}")
    
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine pickle mask files into NIfTI files"
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="Path to the folder containing the DICOM files",
        required=True
    )
    args = parser.parse_args()
    combine_pickled_masks(folder_name=args.folder_name)

