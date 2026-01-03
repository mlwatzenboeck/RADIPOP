import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import radipop_utils.utils
from radipop_utils.preprocessing.dcm_seperation import separate_and_link_by_acquisition_time


def main_function():
    parser = argparse.ArgumentParser(
        description="Pipeline: Separate DICOMs by acquisition time, convert to NIfTI, run TotalSegmentator and contrast estimator."
    )
    
    parser.add_argument(
        "--base_folder",
        type=str,
        required=True,
        help="Path to the base folder containing DICOM files."
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
        "--quiet",
        action="store_true",
        help="Suppress verbose output."
    )
    
    args = parser.parse_args()
    
    # Convert to Path and resolve
    base_folder = Path(args.base_folder).expanduser().resolve()
    
    if not base_folder.is_dir():
        raise NotADirectoryError(f"Base folder does not exist: {base_folder}")
    
    # Step 1: Separate DICOM files by acquisition time
    if not args.skip_separation:
        if not args.quiet:
            print(f"Step 1: Separating DICOM files by acquisition time in {base_folder}")
        separate_and_link_by_acquisition_time(str(base_folder))
        if not args.quiet:
            print("Step 1 completed: DICOM separation done.\n")
    else:
        if not args.quiet:
            print("Step 1: Skipping DICOM separation (--skip_separation flag set).\n")
    
    # Find by_acqtime folder
    by_acqtime_folder = base_folder / "by_acqtime"
    if not by_acqtime_folder.exists():
        raise FileNotFoundError(
            f"by_acqtime folder not found: {by_acqtime_folder}\n"
            "Did you run the separation step? (Remove --skip_separation flag if needed)."
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
    
    # Process each acquisition time folder
    for acq_folder in tqdm(acq_folders, desc="Processing acquisition folders", disable=args.quiet):
        if not args.quiet:
            print(f"\nProcessing: {acq_folder.name}")
        
        # Step 2: Convert DICOM to NIfTI
        if not args.skip_dcm2nii:
            if not args.quiet:
                print(f"  Step 2: Converting DICOM to NIfTI for {acq_folder.name}")
            
            # dcm2nii creates a subfolder with the out_id and places base.nii.gz there
            # We'll use the acquisition time as the out_id
            out_id = acq_folder.name
            
            radipop_utils.utils.dcm2nii(
                dicom_folder=acq_folder,
                output_folder=acq_folder,  # Output in the same folder
                out_id=out_id,
                verbose=not args.quiet
            )
            
            # The output will be in acq_folder / out_id / base.nii.gz
            nii_output_folder = acq_folder / out_id
            nii_file = nii_output_folder / "base.nii.gz"
            
            if not nii_file.exists():
                raise FileNotFoundError(f"Expected NIfTI file not found: {nii_file}")
            
            if not args.quiet:
                print(f"  Step 2 completed: NIfTI conversion done. Output: {nii_file}\n")
        else:
            # Try to find existing NIfTI file
            # Look for subfolders that might contain base.nii.gz
            nii_file = None
            nii_output_folder = None
            for subfolder in acq_folder.iterdir():
                if subfolder.is_dir():
                    potential_nii = subfolder / "base.nii.gz"
                    if potential_nii.exists():
                        nii_file = potential_nii
                        nii_output_folder = subfolder
                        break
            
            if nii_file is None:
                # Try directly in acq_folder
                potential_nii = acq_folder / "base.nii.gz"
                if potential_nii.exists():
                    nii_file = potential_nii
                    nii_output_folder = acq_folder
            
            if nii_file is None:
                raise FileNotFoundError(
                    f"Could not find base.nii.gz in {acq_folder} or its subfolders.\n"
                    "Did you run the dcm2nii step? (Remove --skip_dcm2nii flag if needed)."
                )
            
            if not args.quiet:
                print(f"  Step 2: Skipping dcm2nii (--skip_dcm2nii flag set). Using existing: {nii_file}\n")
        
        # Ensure we have the nii_file and output_folder set
        if nii_output_folder is None:
            nii_output_folder = nii_file.parent
        
        # Step 3: Run TotalSegmentator
        if not args.skip_totalseg:
            if not args.quiet:
                print(f"  Step 3: Running TotalSegmentator for {acq_folder.name}")
            
            seg_output_folder = nii_output_folder / "seg_out"
            seg_output_folder.mkdir(parents=True, exist_ok=True)
            
            # Build TotalSegmentator command
            # --roi_subset accepts multiple values as separate arguments
            cmd = [
                "TotalSegmentator",
                "-i", str(nii_file),
                "-o", str(seg_output_folder),
                "--roi_subset"
            ]
            # Add ROI subset values as separate arguments
            cmd.extend(args.roi_subset)
            cmd.extend(["--device", args.device])
            
            if not args.quiet:
                print(f"  Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=not args.quiet,
                    text=True
                )
                if not args.quiet:
                    print(f"  Step 3 completed: TotalSegmentator done. Output: {seg_output_folder}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running TotalSegmentator: {e}")
                if e.stderr:
                    print(f"stderr: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError(
                    "TotalSegmentator command not found. "
                    "Make sure TotalSegmentator is installed and in your PATH. "
                    "Install with: pip install TotalSegmentator"
                )
        else:
            if not args.quiet:
                print(f"  Step 3: Skipping TotalSegmentator (--skip_totalseg flag set).\n")
        
        # Step 4: Run contrast phase estimator
        if not args.skip_contrast:
            if not args.quiet:
                print(f"  Step 4: Running contrast phase estimator for {acq_folder.name}")
            
            contrast_output_file = nii_output_folder / "contrast_phase.json"
            
            cmd = [
                "totalseg_get_phase",
                "-i", str(nii_file),
                "-o", str(contrast_output_file)
            ]
            
            if not args.quiet:
                print(f"  Running: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=not args.quiet,
                    text=True
                )
                if not args.quiet:
                    print(f"  Step 4 completed: Contrast phase estimation done. Output: {contrast_output_file}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error running totalseg_get_phase: {e}")
                if e.stderr:
                    print(f"stderr: {e.stderr}")
                raise
            except FileNotFoundError:
                raise RuntimeError(
                    "totalseg_get_phase command not found. "
                    "Make sure TotalSegmentator is installed and in your PATH. "
                    "Install with: pip install TotalSegmentator"
                )
        else:
            if not args.quiet:
                print(f"  Step 4: Skipping contrast phase estimation (--skip_contrast flag set).\n")
    
    if not args.quiet:
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)


if __name__ == "__main__":
    main_function()

