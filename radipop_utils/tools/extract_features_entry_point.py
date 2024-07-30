import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import dotenv_values, find_dotenv
import radipop_utils
import radipop_utils.features
from typing import Union

path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent

# Load environment variables as dictionary
config = dotenv_values(find_dotenv())


def main_function():
    parser = argparse.ArgumentParser(description="Extract and save radiomics features from NIfTI images.")
    
    parser.add_argument("--patientid", type=str, help="Name of the newly created output folder")

    parser.add_argument("--image", type=str, help="Path to image in .nii format")

    parser.add_argument("--masks", type=str, help="Path to masks in .nii format; in mask 1=liver; 2=spleen; 0=other")
    
    parser.add_argument("--output_dir", type=str, help="Directory where the extracted features will be saved.")
    
    parser.add_argument("--fe_settings", type=Path, 
                        default=RADIPOP_PACKAGE_ROOT / "yaml" / "exampleCT.yaml",
                        help="Path to the radiomics feature extraction settings file.")
    
    parser.add_argument("--window_location_middle", type=float, 
                    default=50,
                    help="Position (middpoint) of the intesity window. (Default = 50 HU -> soft tissue CT window.)")
    
    parser.add_argument("--window_width", type=float, 
                    default=500,
                    help="Width of the intesity window. (Default = 500 HU -> soft tissue CT window.)")
    
    args = parser.parse_args()
    print(args)
    
    tissue_class_dct = {"liver": 1, "spleen": 2}
   
    radipop_utils.features.extract_and_save_features_from_nii(args.patientid,
                                                                args.image,
                                                                args.masks,
                                                                output_dir=args.output_dir,
                                                                fe_settings_path=args.fe_settings,
                                                                tissue_class_dct=tissue_class_dct,
                                                                check_existence=True,
                                                                verbose=True,
                                                                window_location_middle=args.window_location_middle, 
                                                                window_width = args.window_width)
    print("Done with feature extraction!")



if __name__ == "__main__":
    main_function()