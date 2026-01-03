import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from typing import Union

import radipop_utils.features


def combine_liver_spleen_masks(liver_mask_path: Union[Path, str], 
                                spleen_mask_path: Union[Path, str],
                                output_mask_path: Union[Path, str],
                                verbose: bool = True) -> None:
    """
    Combine separate liver and spleen mask files into a single mask.
    
    Args:
        liver_mask_path: Path to liver mask NIfTI file (binary mask, non-zero = liver)
        spleen_mask_path: Path to spleen mask NIfTI file (binary mask, non-zero = spleen)
        output_mask_path: Path where combined mask will be saved (1=liver, 2=spleen, 0=background)
        verbose: If True, print progress messages
    """
    liver_mask_path = Path(liver_mask_path)
    spleen_mask_path = Path(spleen_mask_path)
    output_mask_path = Path(output_mask_path)
    
    if not liver_mask_path.exists():
        raise FileNotFoundError(f"Liver mask not found: {liver_mask_path}")
    if not spleen_mask_path.exists():
        raise FileNotFoundError(f"Spleen mask not found: {spleen_mask_path}")
    
    if verbose:
        print(f"Loading liver mask: {liver_mask_path}")
    liver_mask_sitk = sitk.ReadImage(str(liver_mask_path))
    liver_array = sitk.GetArrayFromImage(liver_mask_sitk)
    
    if verbose:
        print(f"Loading spleen mask: {spleen_mask_path}")
    spleen_mask_sitk = sitk.ReadImage(str(spleen_mask_path))
    spleen_array = sitk.GetArrayFromImage(spleen_mask_sitk)
    
    # Check that arrays have the same shape
    if liver_array.shape != spleen_array.shape:
        raise ValueError(
            f"Masks have different shapes: liver {liver_array.shape} vs spleen {spleen_array.shape}"
        )
    
    # Convert to binary masks (non-zero -> 1)
    liver_binary = np.where(liver_array > 0, 1, 0).astype(np.uint8)
    spleen_binary = np.where(spleen_array > 0, 1, 0).astype(np.uint8)
    
    # Check for overlap
    overlap = np.logical_and(liver_binary, spleen_binary)
    if np.any(overlap):
        n_overlap = np.sum(overlap)
        if verbose:
            print(f"Warning: {n_overlap} voxels overlap between liver and spleen masks")
            print(f"  Overlap regions will be assigned to spleen (label 2)")
    
    # Combine masks: liver=1, spleen=2
    # Start with liver mask (label 1), then overwrite with spleen mask (label 2)
    # This ensures spleen takes precedence in overlap regions
    combined_array = liver_binary.astype(np.uint8).copy()
    combined_array[spleen_binary > 0] = 2
    
    # Verify the combined mask has only expected values
    unique_values = np.unique(combined_array)
    expected_values = {0, 1, 2}
    if not set(unique_values).issubset(expected_values):
        raise ValueError(f"Combined mask has unexpected values: {unique_values}")
    
    # Create SimpleITK image from array
    combined_mask_sitk = sitk.GetImageFromArray(combined_array)
    # Copy metadata from liver mask (spacing, origin, direction)
    combined_mask_sitk.CopyInformation(liver_mask_sitk)
    
    # Save combined mask
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving combined mask to: {output_mask_path}")
        print(f"  Values in combined mask: {unique_values}")
        print(f"  Liver voxels (1): {np.sum(combined_array == 1)}")
        print(f"  Spleen voxels (2): {np.sum(combined_array == 2)}")
        print(f"  Background voxels (0): {np.sum(combined_array == 0)}")
    
    sitk.WriteImage(combined_mask_sitk, str(output_mask_path))


def main_function():
    parser = argparse.ArgumentParser(
        description="Combine liver and spleen masks from TotalSegmentator and extract radiomics features."
    )
    
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Path to the base folder containing by_acqtime directory."
    )
    
    parser.add_argument(
        "--fe_settings",
        type=str,
        required=True,
        help="Path to the radiomics feature extraction settings YAML file."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where extracted features will be saved."
    )
    
    parser.add_argument(
        "--window_location_middle",
        type=float,
        default=50,
        help="Position (midpoint) of the intensity window. (Default = 50 HU -> soft tissue CT window.)"
    )
    
    parser.add_argument(
        "--window_width",
        type=float,
        default=500,
        help="Width of the intensity window. (Default = 500 HU -> soft tissue CT window.)"
    )
    
    parser.add_argument(
        "--skip_mask_combination",
        action="store_true",
        help="Skip mask combination step (assumes combined masks already exist)."
    )
    
    parser.add_argument(
        "--skip_feature_extraction",
        action="store_true",
        help="Skip feature extraction step."
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output."
    )
    
    args = parser.parse_args()
    
    # Convert to Path and resolve
    base_folder = Path(args.base_folder).expanduser().resolve()
    fe_settings_path = Path(args.fe_settings).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    if not base_folder.is_dir():
        raise NotADirectoryError(f"Base folder does not exist: {base_folder}")
    
    if not fe_settings_path.exists():
        raise FileNotFoundError(f"Feature extraction settings file not found: {fe_settings_path}")
    
    # Find by_acqtime folder
    by_acqtime_folder = base_folder / "by_acqtime"
    if not by_acqtime_folder.exists():
        raise FileNotFoundError(
            f"by_acqtime folder not found: {by_acqtime_folder}\n"
            "Did you run the DICOM separation and dcm2nii pipeline first?"
        )
    
    # Get all acquisition time subfolders
    acq_folders = [d for d in by_acqtime_folder.iterdir() if d.is_dir() and d.name != "metadata_all.csv"]
    
    if not acq_folders:
        raise RuntimeError(f"No acquisition time subfolders found in {by_acqtime_folder}")
    
    if not args.quiet:
        print(f"Found {len(acq_folders)} acquisition time folder(s):")
        for acq_folder in acq_folders:
            print(f"  - {acq_folder.name}")
        print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each acquisition time folder
    for acq_folder in tqdm(acq_folders, desc="Processing acquisition folders", disable=args.quiet):
        if not args.quiet:
            print(f"\nProcessing: {acq_folder.name}")
        
        # Find the NIfTI conversion output folder (should be acq_folder / acq_folder.name)
        nii_output_folder = acq_folder / acq_folder.name
        if not nii_output_folder.exists():
            # Try to find any subfolder with base.nii.gz
            nii_output_folder = None
            for subfolder in acq_folder.iterdir():
                if subfolder.is_dir():
                    if (subfolder / "base.nii.gz").exists():
                        nii_output_folder = subfolder
                        break
            
            if nii_output_folder is None:
                raise FileNotFoundError(
                    f"Could not find NIfTI conversion output folder in {acq_folder}.\n"
                    "Expected structure: {acq_folder}/{acq_folder.name}/base.nii.gz"
                )
        
        base_nii_file = nii_output_folder / "base.nii.gz"
        if not base_nii_file.exists():
            raise FileNotFoundError(f"base.nii.gz not found: {base_nii_file}")
        
        seg_out_folder = nii_output_folder / "seg_out"
        liver_mask_file = seg_out_folder / "liver.nii.gz"
        spleen_mask_file = seg_out_folder / "spleen.nii.gz"
        combined_mask_file = nii_output_folder / "mask_combined.nii.gz"
        
        # Step 1: Combine masks
        if not args.skip_mask_combination:
            if not args.quiet:
                print(f"  Step 1: Combining liver and spleen masks for {acq_folder.name}")
            
            if not liver_mask_file.exists():
                raise FileNotFoundError(f"Liver mask not found: {liver_mask_file}")
            if not spleen_mask_file.exists():
                raise FileNotFoundError(f"Spleen mask not found: {spleen_mask_file}")
            
            combine_liver_spleen_masks(
                liver_mask_path=liver_mask_file,
                spleen_mask_path=spleen_mask_file,
                output_mask_path=combined_mask_file,
                verbose=not args.quiet
            )
            
            if not args.quiet:
                print(f"  Step 1 completed: Combined mask saved to {combined_mask_file}\n")
        else:
            if not combined_mask_file.exists():
                raise FileNotFoundError(
                    f"Combined mask not found: {combined_mask_file}\n"
                    "Did you run the mask combination step? (Remove --skip_mask_combination flag if needed)."
                )
            if not args.quiet:
                print(f"  Step 1: Skipping mask combination (--skip_mask_combination flag set).\n")
        
        # Step 2: Extract features
        if not args.skip_feature_extraction:
            if not args.quiet:
                print(f"  Step 2: Extracting radiomics features for {acq_folder.name}")
            
            # Use acquisition folder name as patient ID
            patient_id = acq_folder.name
            
            radipop_utils.features.extract_and_save_features_from_nii(
                patientid=patient_id,
                image_loc=base_nii_file,
                mask_loc=combined_mask_file,
                output_dir=output_dir,
                fe_settings_path=fe_settings_path,
                tissue_class_dct={"liver": 1, "spleen": 2},  # Combined mask: 1=liver, 2=spleen
                check_existence=True,
                verbose=not args.quiet,
                window_location_middle=args.window_location_middle,
                window_width=args.window_width
            )
            
            if not args.quiet:
                print(f"  Step 2 completed: Features extracted for {patient_id}\n")
        else:
            if not args.quiet:
                print(f"  Step 2: Skipping feature extraction (--skip_feature_extraction flag set).\n")
    
    if not args.quiet:
        print("\n" + "="*60)
        print("Mask combination and feature extraction completed successfully!")
        print(f"Features saved to: {output_dir}")
        print("="*60)


if __name__ == "__main__":
    main_function()

