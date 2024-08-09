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
    
    args = parser.parse_args()
    
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    if args.output_dir is None:
        args.output_dir = args.radiomics_dir
        print("  ->  Output directory is set to radiomics_dir: ", args.output_dir)
    
    print()
    
    df = radipop_utils.data.load_HVPG_values_and_radiomics(paths_and_hvpg_data_file = args.images_and_mask_paths_file, 
                                                                              radiomics_dir = args.radiomics_dir)
    
    df_Tr, df_iTs, df_eTs = radipop_utils.data.split_radiomics_data(df)
    print(f"{len(df_Tr)} training samples, {len(df_iTs)} internal test samples, {len(df_eTs)} external test samples.")
    
    save_path = args.output_dir
    df_Tr.to_csv(save_path / f"{args.output_prefix}_Tr.csv", index=False)
    print(f"Data {args.output_prefix}_Tr.csv saved to: {save_path / f'{args.output_prefix}_Tr.csv'}")
        
    df_iTs.to_csv(save_path / f"{args.output_prefix}_iTs.csv", index=False)
    print(f"Data {args.output_prefix}_iTs.csv saved to: {save_path / f'{args.output_prefix}_iTs.csv'}")
    
    df_eTs.to_csv(save_path / f"{args.output_prefix}_eTs.csv", index=False)
    print(f"Data {args.output_prefix}_eTs.csv saved to: {save_path / f'{args.output_prefix}_eTs.csv'}")
    

if __name__ == "__main__":
    main_function()