
import os
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import dotenv_values, find_dotenv
import radipop_utils
import radipop_utils.features
from typing import Union
import radipop_utils.utils
from tqdm import tqdm
import yaml
import SimpleITK as sitk
import numpy as np 
import argparse
import pprint

path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent

# # Load environment variables as dictionary
# config = dotenv_values(find_dotenv())

def get_reasonable_binwith(df_file_names: pd.DataFrame, 
                           frac=0.1,
                           verbose=False,
                           fe_setting: dict = {
                               "binWidth": 12,
                               "resampledPixelSpacing": [2.0, 0.71875, 0.71875],
                               "interpolator": sitk.sitkBSpline,
                               "padDistance": 10,
                           },
                           windowing_dct : Union[None, dict] = None):
    res = []
    shapes = []
    tissue_class_dct = {"liver": 1, "spleen": 2}
    df = df_file_names.sample(frac=frac, random_state=1).reset_index(drop=True)
    for idx in tqdm(range(len(df))):
        image_loc = df.loc[idx, "images"]
        mask_loc = df.loc[idx, "masks"]

        mask = sitk.ReadImage(mask_loc)
        if windowing_dct == None: 
            img = sitk.ReadImage(image_loc)
        else:
            # windowing_dct = dict(out_range = [0,255], 
            #                      wl = 60, 
            #                      ww = 600, 
            #                      dtype = np.uint8)
            # TODO 
            img_transformed = radipop_utils.features.convert_and_extract_from_nii(image_loc, **windowing_dct)      
        
        s = sitk.GetArrayFromImage(img).shape
        if verbose:
            print(f"shape = ", s)
        shapes.append(s)
        for tissue_type in tissue_class_dct.keys():
            r = radipop_utils.utils.suggest_radiomics_binwidth(img, mask == tissue_class_dct[tissue_type],
                                                               settings=fe_setting,
                                                               verbose=verbose,
                                                               return_range_instead=True
                                                               )
            res.append(r)
    return np.array(res), shapes




def main_function():
    parser = argparse.ArgumentParser(description="Read the radiomics file you intend to use and suggest an estimate for the binwidth")

    parser.add_argument("--images_and_mask_paths_file", type=Path, 
                        default=RADIPOP_PACKAGE_ROOT / "data" / "file_paths_and_hvpg_data.xlsx",
                        help="Path to the Excel file containing image and mask paths and patient IDs.")
    
    parser.add_argument("--fe_settings", type=Path, 
                        default=RADIPOP_PACKAGE_ROOT / "yaml" / "radiomics_fe_setttings_CT_no_preprocessing.yaml",
                        help="Path to the radiomics feature extraction settings file.")
    
    parser.add_argument("--frac", type=float, 
                    default=1.0,
                    help="Use only a random fraction of the images for estimating the binwidth.")
    
    args = parser.parse_args()
    
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint.pprint(args_dict)
    print()
    
    # ranges, shapes = get_reasonable_binwith(df)
    with open(args.fe_settings) as f:
        settings = yaml.safe_load(f)
        
    df = pd.read_excel(args.images_and_mask_paths_file)

    windowing_dct = dict(out_range = [0,255], 
                         wl = 60, 
                         ww = 600, 
                         dtype = np.float64)
    # windowing_dct = dict(out_range = [0,255], 
    #                      wl = 60, 
    #                      ww = 600, 
    #                      dtype = np.uint8)
    ranges, shapes = get_reasonable_binwith(df, frac=args.frac, fe_setting=settings["setting"], verbose=True, windowing_dct=windowing_dct) 

    print(f"{shapes =}", "\n")
    # range / binwidth shoudl be 16 - 128:
    
    ranges = np.sort(ranges)
    print(f"sorted {ranges =}")
    
    
    suggestion1 = ranges[0] / 16.0
    suggestion2 = ranges[-1] / 128
    suggestion3 = ranges.mean() / ((128 + 16)/2)
    suggestion4 = np.median(ranges) / ((128 + 16)/2)
    
    
    suggested_binwidth = suggestion1
    print(f"{suggested_binwidth = } for the settings in {args.fe_settings}.")
    print("This leads to range / binwidth = ", ranges / suggested_binwidth)
    # we cant make everyone happy, but lets try: 
    print(f"With this bin width the ratio: range/binwidth is < 16 for {sum(ranges / suggested_binwidth < 16)}/{int(len(df) * args.frac)}  and > 128 for {sum(ranges / suggested_binwidth > 128)}/{int(len(df) * args.frac)}.")


if __name__ == "__main__":
    main_function()