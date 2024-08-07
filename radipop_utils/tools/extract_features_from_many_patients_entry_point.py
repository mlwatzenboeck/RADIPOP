import os
import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import dotenv_values, find_dotenv
import radipop_utils
import radipop_utils.features
from typing import Union
from pprint import pprint

path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent

# Load environment variables as dictionary
config = dotenv_values(find_dotenv())

def extraction_loop(images_and_mask_paths_file: Union[Path, str], output_dir: Union[Path, str], fe_settings: Union[Path, str],
                    window_location_middle=50, window_width=500, use_png_range=False):
    df = pd.read_excel(images_and_mask_paths_file)
    assert "images" in df.columns
    assert "masks" in df.columns
    assert "id" in df.columns

    os.makedirs(output_dir, exist_ok=True)

    tissue_class_dct = {"liver": 1, "spleen": 2}

    for i in tqdm(range(len(df))):
        image_loc = df.loc[i, "images"]
        mask_loc = df.loc[i, "masks"]
        patientid = df.loc[i, "id"]

        radipop_utils.features.extract_and_save_features_from_nii(patientid,
                                                                  image_loc,
                                                                  mask_loc,
                                                                  output_dir=output_dir,
                                                                  fe_settings_path=fe_settings,
                                                                  tissue_class_dct=tissue_class_dct,
                                                                  check_existence=True,
                                                                  verbose=True,
                                                                  window_location_middle=window_location_middle, 
                                                                  window_width = window_width, 
                                                                  use_png_range=use_png_range)

    print("Done with feature extraction!")


def main_function():
    parser = argparse.ArgumentParser(description="Extract and save radiomics features from NIfTI images (paths provided as an xlsx file.)")
    
    DATA_ROOT_DIRECTORY = Path(config["DATA_ROOT_DIRECTORY"])

    parser.add_argument("--images_and_mask_paths_file", type=Path, 
                        default=RADIPOP_PACKAGE_ROOT / "data" / "file_paths_and_hvpg_data.xlsx",
                        help="Path to the Excel file containing image and mask paths and patient IDs.")
    
    parser.add_argument("--output_dir", type=Path, 
                        default=DATA_ROOT_DIRECTORY / "radiomics" / "Dataset125_LSS" / "radipop",
                        help="Directory where the extracted features will be saved.")
    
    parser.add_argument("--fe_settings", type=Path, 
                        default=RADIPOP_PACKAGE_ROOT / "yaml" / "exampleCT.yaml",
                        help="Path to the radiomics feature extraction settings file.")
    
    parser.add_argument("--window_location_middle", type=float, 
                    default=50,
                    help="Position (middpoint) of the intesity window. (Default = 50 HU -> soft tissue CT window.)")
    
    parser.add_argument("--window_width", type=float, 
                    default=500,
                    help="Width of the intesity window. (Default = 500 HU -> soft tissue CT window.)")    

    parser.add_argument("--use_png_range", action="store_true", 
                        help="Use out_range [0,255] instead of [0.0, 1.0] and store as uint8.")
    
    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    print()
    
    extraction_loop(args.images_and_mask_paths_file, args.output_dir, args.fe_settings, 
                    window_location_middle=args.window_location_middle, 
                    window_width = args.window_width, 
                    use_png_range=args.use_png_range)
    
    # copy fe_settings file to output_dir
    try:
        shutil.copy(args.fe_settings, args.output_dir)
        print(f"Settings file {args.fe_settings} copied to {args.output_dir}")
    except Exception as e:
        print(f"An error occurred while copying the settings file: {e}")


if __name__ == "__main__":
    main_function()