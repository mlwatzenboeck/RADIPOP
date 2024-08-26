import os
import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Union
from pprint import pprint

# from dotenv import dotenv_values, find_dotenv

import radipop_utils
import radipop_utils.features
import radipop_utils.data
import radipop_utils.utils

import datetime

# path = Path(os.path.abspath(radipop_utils.__file__))
# RADIPOP_PACKAGE_ROOT = path.parent.parent

# # Load environment variables as dictionary
# config = dotenv_values(find_dotenv())



def main_function():
    parser = argparse.ArgumentParser(description="Extract and save radiomics features from NIfTI images (paths provided as an xlsx file.)")
    
    # DATA_ROOT_DIRECTORY = Path(config["DATA_ROOT_DIRECTORY"])

    parser.add_argument("--images_and_mask_paths_file", type=Path, 
                        help="Path to the Excel file containing image and mask paths and patient IDs.")
    
    parser.add_argument("--radiomics_dir", type=Path, 
                        help="Directory where features are collected from.")
    
    parser.add_argument("--output_dir", type=Path, 
                        default=None,
                        help="Directory where the extracted features will be saved. (Default: radiomics_dir)")
    
    parser.add_argument("--output_prefix", type=str, 
                        default="df",
                        help="Prefix for the output CSV file names. (Default: df)")
    
    parser.add_argument("--features_regex_filter", type=str,
                        default="^liver|^spleen",
                        help="Regex pattern to filter the features columns. (Default: '^liver|^spleen')")
    
    args = parser.parse_args()
    
    args_dict = vars(args)
    print(f"Running: '{Path(__file__).name}' with the following arguments:")
    print("---------------")
    pprint(args_dict)
    if args.output_dir is None:
        args.output_dir = args.radiomics_dir
        print("  ->  Output directory is set to radiomics_dir: ", args.output_dir)
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    dst = args.output_dir / f"args_settings__combine_radiomics_with_scalar_target_and_split_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)
    
    print()
    
    df = radipop_utils.data.load_HVPG_values_and_radiomics(paths_and_hvpg_data_file = args.images_and_mask_paths_file, 
                                                                              radiomics_dir = args.radiomics_dir)
    
    df_Tr, df_iTs, df_eTs = radipop_utils.data.split_radiomics_data(df, featurs_regex_filter = args.features_regex_filter)
    
    
    print(f"{len(df_Tr)} training samples, {len(df_iTs)} internal test samples, {len(df_eTs)} external test samples.")
    print("Number of radiomics features: ", len(df_Tr.filter(regex=args.features_regex_filter).columns))
    
    save_path = args.output_dir
    df_Tr.to_csv(save_path / f"{args.output_prefix}_Tr.csv", index=False)
    print(f"Data {args.output_prefix}_Tr.csv saved to: {save_path / f'{args.output_prefix}_Tr.csv'}")
        
    df_iTs.to_csv(save_path / f"{args.output_prefix}_iTs.csv", index=False)
    print(f"Data {args.output_prefix}_iTs.csv saved to: {save_path / f'{args.output_prefix}_iTs.csv'}")
    
    df_eTs.to_csv(save_path / f"{args.output_prefix}_eTs.csv", index=False)
    print(f"Data {args.output_prefix}_eTs.csv saved to: {save_path / f'{args.output_prefix}_eTs.csv'}")
    

if __name__ == "__main__":
    main_function()