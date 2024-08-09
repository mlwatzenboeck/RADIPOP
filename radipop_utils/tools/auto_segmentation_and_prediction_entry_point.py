# imports 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from pathlib import Path
from typing import Union, Optional

import totalsegmentator
import totalsegmentator.python_api
import argparse
from pprint import pprint

import radipop_utils 
import radipop_utils.visualization
import radipop_utils.features
import radipop_utils.utils
import radipop_utils.data
import radipop_utils.inference
import radipop_utils.inference_via_total_segmentor

import datetime


# needed for loading the settings for the radiomics feature extractor 
path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent

# load user/ system specific env variables:
from dotenv import dotenv_values, find_dotenv
config = dotenv_values(find_dotenv())  # load environment variables as dictionary
DATA_ROOT_DIRECTORY = Path(config["DATA_ROOT_DIRECTORY"])

# These all need to match with the loaded model
fe_settings_path = RADIPOP_PACKAGE_ROOT / "yaml" / "radiomics_fe_setttings_CT_no_preprocessing_spacing_222.yaml"
model_dir = DATA_ROOT_DIRECTORY / "radiomics" / "Dataset125_LSS" / "regression" / "radipop_222"

def main_function():
    parser = argparse.ArgumentParser(
    description='Run autosegmentation, extract radiomics and prediction on a single patient.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--image_path', type=str, help='location of the input image (can be DICOM folder or or NIFTI file).')
    parser.add_argument('--output_folder', type=str, default=None, help='If and in which folder the created segmentation mask should be stored (default: %(default)s -> no storage).')

    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst = args.output_dir / f"args_settings__auto_segmentation_and_prediction_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)
    
    if args.image_path.endswith(".nii") or args.image_path.endswith(".nii.gz"):
        dicom = False
    else: 
        dicom = True
    
    radipop_utils.inference_via_total_segmentor.inference_via_total_segmentor(image_loc=args.image_path, 
                                                                              dicom=dicom, 
                                                                              output_folder=args.output_folder, 
                                                                              model_dir=model_dir,                # not exposed in the entry point for now 
                                                                              fe_settings_path=fe_settings_path   # not exposed in the entry point for now  
                                                                              )
    print("Done with auto segmentation and HVPG prediction!")
    return None



if __name__ == "__main__":
    # dst = "/home/cwatzenboeck/tmp/radipop_patientdev"   
    
    # # # convert from dictom to nifti
    # input_path_dicom = Path("/home/cwatzenboeck/data/project_segmenation_ls/2024_muw_additional_DICOMs/V.313 horos")
    
    main_function()
    
