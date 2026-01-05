import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import traceback


def main_function():
    """
    Unified pipeline that processes all patient folders from a base directory.
    
    For each patient folder (e.g., FINAL.11, FINAL.38):
    1. Separates DICOMs by acquisition time
    2. Converts each to NIfTI
    3. Runs TotalSegmentator
    4. Runs contrast phase estimator
    5. Combines masks
    6. Extracts radiomics features
    """
    parser = argparse.ArgumentParser(
        description="Full pipeline: Process all patient folders - separate DICOMs, convert to NIfTI, segment, and extract features."
    )
    
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Base folder containing patient folders (e.g., FINAL.11, FINAL.38)."
    )
    
    parser.add_argument(
        "--fe_settings",
        type=str,
        required=True,
        help="Path to the radiomics feature extraction settings YAML file."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for TotalSegmentator (default: cpu). Use 'cuda' for GPU."
    )
    
    parser.add_argument(
        "--roi_subset",
        type=str,
        nargs="+",
        default=["liver", "spleen"],
        help="ROI subset for TotalSegmentator (default: liver spleen)."
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
        "--skip_separation",
        action="store_true",
        help="Skip DICOM separation step (assumes by_acqtime folders already exist)."
    )
    
    parser.add_argument(
        "--skip_dcm2nii",
        action="store_true",
        help="Skip dcm2nii conversion step (assumes NIfTI files already exist)."
    )
    
    parser.add_argument(
        "--skip_totalseg",
        action="store_true",
        help="Skip TotalSegmentator step."
    )
    
    parser.add_argument(
        "--skip_contrast",
        action="store_true",
        help="Skip contrast phase estimation step."
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
    
    parser.add_argument(
        "--skip_if_looks_finished",
        action="store_true",
        default=True,
        help="Skip processing if feature files already exist (default: True)."
    )
    
    parser.add_argument(
        "--no-skip_if_looks_finished",
        dest="skip_if_looks_finished",
        action="store_false",
        help="Process all patients even if feature files already exist (overrides --skip_if_looks_finished)."
    )
    
    args = parser.parse_args()
    
    # Convert to Path and resolve
    base_folder = Path(args.base_folder).expanduser().resolve()
    
    if not base_folder.is_dir():
        raise NotADirectoryError(f"Base folder does not exist: {base_folder}")
    
    # Find all patient folders (directories in base_folder)
    patient_folders = [d for d in base_folder.iterdir() if d.is_dir()]
    
    if not patient_folders:
        raise RuntimeError(f"No patient folders found in {base_folder}")
    
    # Filter out any hidden directories or by_acqtime folders if they exist
    patient_folders = [d for d in patient_folders if not d.name.startswith('.') and d.name != 'by_acqtime']
    
    if not patient_folders:
        raise RuntimeError(f"No valid patient folders found in {base_folder}")
    
    patient_folders.sort()  # Sort for consistent processing order
    
    if not args.quiet:
        print(f"Found {len(patient_folders)} patient folder(s):")
        for pf in patient_folders:
            print(f"  - {pf.name}")
        print()
    
    # Process each patient folder
    failed_patients = []
    skipped_patients = []
    
    for patient_folder in tqdm(patient_folders, desc="Processing patients", disable=args.quiet):
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"Processing patient: {patient_folder.name}")
            print(f"{'='*60}")
        
        # Check if we should skip this patient (if feature files already exist)
        if args.skip_if_looks_finished:
            by_acqtime_folder = patient_folder / "by_acqtime"
            if by_acqtime_folder.exists():
                # Check all acquisition folders for feature files
                acq_folders = [d for d in by_acqtime_folder.iterdir() if d.is_dir() and d.name != "metadata_all.csv"]
                all_finished = True
                
                for acq_folder in acq_folders:
                    # Features are saved to acq_folder / acq_folder.name / Features_*.xlsx
                    feature_folder = acq_folder / acq_folder.name
                    liver_features = feature_folder / "Features_liver.xlsx"
                    spleen_features = feature_folder / "Features_spleen.xlsx"
                    
                    if not (liver_features.exists() and spleen_features.exists()):
                        all_finished = False
                        break
                
                if all_finished and len(acq_folders) > 0:
                    if not args.quiet:
                        print(f"Skipping {patient_folder.name}: Feature files already exist for all acquisitions.")
                    skipped_patients.append(patient_folder.name)
                    continue
        
        # Create error log file path
        error_log_file = patient_folder / "pipeline_error.log"
        
        try:
            # Step 1: Run DICOM separation, dcm2nii, TotalSegmentator pipeline
            if not args.quiet:
                print(f"\nStep 1: Running DICOM separation, conversion, and segmentation pipeline...")
            
            # Build command for pipeline script (use the entry point command)
            pipeline_cmd = [
                "radipop_pipeline_dcm_separation_dcm2nii_totalseg",
                "--base_folder", str(patient_folder),
                "--device", args.device,
                "--roi_subset"] + args.roi_subset
            
            if args.skip_separation:
                pipeline_cmd.append("--skip_separation")
            if args.skip_dcm2nii:
                pipeline_cmd.append("--skip_dcm2nii")
            if args.skip_totalseg:
                pipeline_cmd.append("--skip_totalseg")
            if args.skip_contrast:
                pipeline_cmd.append("--skip_contrast")
            if args.quiet:
                pipeline_cmd.append("--quiet")
            
            result = subprocess.run(
                pipeline_cmd, 
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = (
                    f"Pipeline script failed for {patient_folder.name} with return code {result.returncode}\n"
                    f"Command: {' '.join(pipeline_cmd)}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}\n"
                )
                with open(error_log_file, "w") as f:
                    f.write(f"Error occurred at: {datetime.now().isoformat()}\n")
                    f.write(error_msg)
                if not args.quiet:
                    print(f"ERROR: {error_msg}")
                failed_patients.append((patient_folder.name, "Step 1: DICOM separation/conversion/segmentation"))
                continue  # Skip to next patient
            
            # Step 2: Run mask combination and feature extraction
            if not args.quiet:
                print(f"\nStep 2: Running mask combination and feature extraction...")
            
            # Build command for combine_masks script (use the entry point command)
            combine_cmd = [
                "radipop_combine_masks_and_extract_features",
                "--base_folder", str(patient_folder),
                "--fe_settings", args.fe_settings,
                "--window_location_middle", str(args.window_location_middle),
                "--window_width", str(args.window_width)
            ]
            
            if args.skip_mask_combination:
                combine_cmd.append("--skip_mask_combination")
            if args.skip_feature_extraction:
                combine_cmd.append("--skip_feature_extraction")
            if args.quiet:
                combine_cmd.append("--quiet")
            
            result = subprocess.run(
                combine_cmd, 
                check=False,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = (
                    f"Combine masks script failed for {patient_folder.name} with return code {result.returncode}\n"
                    f"Command: {' '.join(combine_cmd)}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}\n"
                )
                with open(error_log_file, "w") as f:
                    f.write(f"Error occurred at: {datetime.now().isoformat()}\n")
                    f.write(error_msg)
                if not args.quiet:
                    print(f"ERROR: {error_msg}")
                failed_patients.append((patient_folder.name, "Step 2: Mask combination/feature extraction"))
                continue  # Skip to next patient
            
            # Success - remove error log if it exists from a previous run
            if error_log_file.exists():
                error_log_file.unlink()
                
        except Exception as e:
            # Catch any unexpected exceptions
            error_msg = (
                f"Unexpected error processing {patient_folder.name}:\n"
                f"{type(e).__name__}: {str(e)}\n"
                f"\nTraceback:\n{traceback.format_exc()}\n"
            )
            with open(error_log_file, "w") as f:
                f.write(f"Error occurred at: {datetime.now().isoformat()}\n")
                f.write(error_msg)
            if not args.quiet:
                print(f"ERROR: {error_msg}")
            failed_patients.append((patient_folder.name, f"Unexpected error: {type(e).__name__}"))
            continue  # Skip to next patient
    
    if not args.quiet:
        print("\n" + "="*60)
        processed_count = len(patient_folders) - len(skipped_patients)
        success_count = processed_count - len(failed_patients)
        
        if skipped_patients:
            print(f"Skipped {len(skipped_patients)} patient(s) (features already exist):")
            for patient_name in skipped_patients:
                print(f"  - {patient_name}")
            print()
        
        if failed_patients:
            print(f"Pipeline completed with {len(failed_patients)} failed patient(s) out of {processed_count} processed:")
            for patient_name, error_type in failed_patients:
                print(f"  - {patient_name}: {error_type}")
                print(f"    Error log: {base_folder / patient_name / 'pipeline_error.log'}")
        else:
            print(f"Full pipeline completed successfully!")
            print(f"  Processed: {success_count}/{processed_count} patients")
            if skipped_patients:
                print(f"  Skipped: {len(skipped_patients)} patients (already finished)")
        print("="*60)


if __name__ == "__main__":
    main_function()

