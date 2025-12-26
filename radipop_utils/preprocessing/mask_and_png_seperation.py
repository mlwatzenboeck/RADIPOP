from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List
import argparse


def _extract_dcm_number(dcm_name: str) -> int:
    """
    Extract the number from a DICOM filename like "IM-0001-0001.dcm".
    Returns the number before .dcm (e.g., 1 for "IM-0001-0001.dcm").
    """
    # Pattern: IM-XXXX-YYYY.dcm where YYYY is the number we want
    match = re.search(r"IM-\d+-(\d+)\.dcm$", dcm_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract number from DICOM filename: {dcm_name}")


def safe_symlink(src: Path, dst: Path) -> None:
    """Create a symlink if the destination doesn't exist."""
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src, start=dst.parent)
    os.symlink(rel, dst)


def separate_and_link_pngs_and_masks(folder_name: str) -> None:
    """
    Create symlinks for PNG and mask files organized by acquisition time.
    
    For each folder in {folder_name}/by_acqtime/, reads metadata.csv and creates
    symlinks from the original folder to pngs/ and masks/ subfolders.
    """
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
    
    missing_pngs: Dict[str, List[str]] = {}
    missing_masks: Dict[str, List[str]] = {}
    
    for acq_folder in acq_time_folders:
        metadata_csv = acq_folder / "metadata.csv"
        if not metadata_csv.exists():
            print(f"Warning: metadata.csv not found in {acq_folder}, skipping...")
            continue
        
        # Read metadata.csv
        with metadata_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if len(rows) == 0:
            print(f"Warning: metadata.csv is empty in {acq_folder}, skipping...")
            continue
        
        # Create subfolders
        pngs_dir = acq_folder / "pngs"
        masks_dir = acq_folder / "masks"
        pngs_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        missing_pngs[acq_folder.name] = []
        missing_masks[acq_folder.name] = []
        
        # Process each row
        for row in rows:
            dcm_name = row.get("dcm_name", "")
            if not dcm_name:
                continue
            
            try:
                dcm_number = _extract_dcm_number(dcm_name)
                # Apply off-by-one correction (DICOM number 288 corresponds to file 289.png)
                ##  WHY??? IT should be the other way arround.... TODO   BUG 
                # Should be 0.png   corresponds to ...-0001.dcm  
                file_index = dcm_number + 1
                
                # Source files in the original folder
                png_src = root / f"{file_index}.png"
                mask_src = root / f"{file_index}.p"
                
                # Destination symlinks
                png_dst = pngs_dir / f"{file_index}.png"
                mask_dst = masks_dir / f"{file_index}.p"
                
                # Create symlinks if source files exist
                if png_src.exists():
                    safe_symlink(png_src, png_dst)
                else:
                    missing_pngs[acq_folder.name].append(f"{file_index}.png")
                
                if mask_src.exists():
                    safe_symlink(mask_src, mask_dst)
                else:
                    missing_masks[acq_folder.name].append(f"{file_index}.p")
            
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not process {dcm_name} in {acq_folder}: {e}")
                continue
        
        print(f"Processed {acq_folder.name}: created symlinks in pngs/ and masks/")
    
    # Print summary of missing files
    print("\n" + "="*60)
    print("Summary of missing files:")
    print("="*60)
    
    total_missing_pngs = sum(len(files) for files in missing_pngs.values())
    total_missing_masks = sum(len(files) for files in missing_masks.values())
    
    if total_missing_pngs == 0 and total_missing_masks == 0:
        print("All files found and linked successfully!")
    else:
        if total_missing_pngs > 0:
            print(f"\nMissing PNG files (total: {total_missing_pngs}):")
            for folder, files in missing_pngs.items():
                if files:
                    print(f"  {folder}: {len(files)} file(s) - {', '.join(files[:5])}"
                          + (f" ... and {len(files)-5} more" if len(files) > 5 else ""))
        
        if total_missing_masks > 0:
            print(f"\nMissing mask files (total: {total_missing_masks}):")
            for folder, files in missing_masks.items():
                if files:
                    print(f"  {folder}: {len(files)} file(s) - {', '.join(files[:5])}"
                          + (f" ... and {len(files)-5} more" if len(files) > 5 else ""))
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate and link PNG and mask files by acquisition time"
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="Path to the folder containing the DICOM files",
        required=True
    )
    args = parser.parse_args()
    separate_and_link_pngs_and_masks(folder_name=args.folder_name)

