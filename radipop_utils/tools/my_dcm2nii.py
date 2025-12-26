import argparse
from pathlib import Path
import radipop_utils.utils


def main_function():
    parser = argparse.ArgumentParser(
        description="Convert DICOM files in a folder to NIfTI format."
    )
    
    parser.add_argument(
        "--dicom_folder",
        type=str,
        required=True,
        help="Path to the folder containing DICOM files."
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output folder where the converted NIfTI files will be saved."
    )
    
    parser.add_argument(
        "--out_id",
        type=str,
        default=None,
        help="Identifier for the output folder. If None, the patient ID extracted from DICOM files will be used."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print conversion details. (Default: True)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (overrides --verbose)."
    )
    
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    dicom_folder = Path(args.dicom_folder)
    output_folder = Path(args.output_folder)
    
    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    
    # Call the dcm2nii function
    radipop_utils.utils.dcm2nii(
        dicom_folder=dicom_folder,
        output_folder=output_folder,
        out_id=args.out_id,
        verbose=verbose
    )
    
    print("DICOM to NIfTI conversion completed!")


if __name__ == "__main__":
    main_function()

