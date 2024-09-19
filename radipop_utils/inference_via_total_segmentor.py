# imports 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

import nibabel as nib
from tqdm import tqdm
from typing import Union
import tempfile
import shutil

import totalsegmentator
import totalsegmentator.python_api

import radipop_utils 
import radipop_utils.visualization
import radipop_utils.features
import radipop_utils.utils
import radipop_utils.data
import radipop_utils.inference


# needed for loading the settings for the radiomics feature extractor 
path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent

# load user/ system specific env variables:
# from dotenv import dotenv_values, find_dotenv
# config = dotenv_values(find_dotenv())  # load environment variables as dictionary

# path to the data. You will (likely need to change this)
# DATA_ROOT_DIRECTORY = Path(config["DATA_ROOT_DIRECTORY"])


# # These all need to match with the loaded model
# fe_settings_path = RADIPOP_PACKAGE_ROOT / "yaml" / "radiomics_fe_setttings_CT_no_preprocessing_spacing_222.yaml"
# model_dir = DATA_ROOT_DIRECTORY / "radiomics" / "Dataset125_LSS" / "regression" / "radipop_222"

def inference_via_total_segmentor(image_loc: Union[Path, str], 
                                  fe_settings_path: Union[Path, str, None] = None, 
                                  model_dir: Union[Path, str, None] = None, 
                                  dicom=False, 
                                  output_folder : Union[str, None] = None):
    
    if fe_settings_path == None:
        fe_settings_path = radipop_utils.inference.load_fe_settings_from_package()
        print(f"Using default settings for radiomics feature extraction: {fe_settings_path}")    
    fe_settings_path = Path(fe_settings_path)
    
    
    # These must match the settings used for training the model
    window_location_middle = 50
    window_width = 500

    with tempfile.TemporaryDirectory() as tmp_wd_path:
        tmp_wd_path = Path(str(tmp_wd_path))

        subfolder_name = "tmp"
        if dicom:
            radipop_utils.utils.dcm2nii(image_loc, tmp_wd_path, out_id = subfolder_name, verbose=True)
            image_loc = tmp_wd_path / subfolder_name / "base.nii.gz"
        else: 
            image_loc = image_loc
        

        # auto segmentation
        os.makedirs(tmp_wd_path / subfolder_name, exist_ok=True)
        mask_loc = tmp_wd_path / subfolder_name / "mask_ts.nii.gz"
        print("Loading volume: ", image_loc)
        input_img = nib.load(image_loc)
        output_img = totalsegmentator.python_api.totalsegmentator(input_img)
        nib.save(output_img, mask_loc)
        print("Saved mask to: ", mask_loc)
            
        # extract radiomics features
        tissue_class_dct = {"liver": 5, "spleen": 1}
        mask = sitk.ReadImage(mask_loc)
        img = radipop_utils.features.convert_and_extract_from_nii(image_loc, out_range=[0, 1.0], wl=window_location_middle, ww=window_width, dtype=np.float64)  
        radiomics_dataframes = {}   
        for tissue_type in tqdm(tissue_class_dct.keys()):
            print(f"Extracting features for tissue type: {tissue_type}")
            features_df = radipop_utils.features.extract_radiomics_features(img, mask == tissue_class_dct[tissue_type], fe_settings_path)
            radiomics_dataframes[tissue_type] = features_df
        for tissue_type, features_df in radiomics_dataframes.items():   
            out_file = tmp_wd_path / subfolder_name  / f"Features_{tissue_type}.csv"
            features_df.to_csv(out_file)
            print(f"Saved features to: {out_file}")    
        dfc = radipop_utils.data.combined_radiomics_features(radiomics_dataframes)
        
        # Create a new mask with only the relevant tissue classes
        new_tissue_class_dct = {"liver": 1, "spleen": 2}
        mask = sitk.ReadImage(mask_loc)
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array_liver = np.where(mask_array == tissue_class_dct["liver"], 1, 0)
        mask_array_spleen = np.where(mask_array == tissue_class_dct["spleen"], 1, 0)
        mask_array_combined = (mask_array_liver * new_tissue_class_dct["liver"] +
                            mask_array_spleen * new_tissue_class_dct["spleen"])
        assert np.all(np.unique(mask_array_combined) == [0, 1, 2]), "Error in combining the masks. Overlapping values?"

        mask_combined = sitk.GetImageFromArray(mask_array_combined)
        mask_combined.CopyInformation(mask)
        mask_combined_loc = tmp_wd_path / subfolder_name / "mask_liver_and_spleen.nii.gz"

        # Use SimpleITK to save the image
        sitk.WriteImage(mask_combined, str(mask_combined_loc))
        print("Saved combined mask to: ", mask_combined_loc)
        


        # load_models_and_params
        if model_dir != None:
            model_dir = Path(model_dir)
            loaded_models, _, _ = radipop_utils.inference.load_models_and_params(model_dir = model_dir)
            model = loaded_models["RF"]
            print(f"Loaded model RF from: {model_dir}")
        else:
            model = radipop_utils.inference.load_model_from_package()
            print(f"Loaded model package")

        # load normalization_df and scaler
        # normalization_df = radipop_utils.data.load_normalization_df(model_dir)
        # scaler = radipop_utils.data.make_scaler_from_normalization_df(normalization_df)
        # X = scaler.transform(dfc)

        # predict
        y_pred = model.predict(dfc)
        
    
        df = pd.DataFrame({"HVPG": y_pred})
        df.to_csv(tmp_wd_path / subfolder_name / "HVPG_prediction.csv", index=False)
        

        if output_folder is not None:
            output_folder = Path(output_folder)
            print("Copying results to: ", output_folder)
            print(f"Consider running \n >>> itksnap -g {output_folder}/base.nii.gz -s {output_folder}/mask_liver_and_spleen.nii.gz \nor a similar viewer to inspect the segmenation.")
            os.makedirs(output_folder, exist_ok=True)
            
            # Copy the contents of tmp_wd_path / subfolder_name into output_folder
            src_folder = tmp_wd_path / subfolder_name
            for item in src_folder.iterdir():
                s = item
                d = output_folder / item.name
                if item.is_dir():
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
            if dicom:
                src = tmp_wd_path / subfolder_name / "base.nii.gz"
                dst = output_folder  # No 'radipop_results' subfolder
                # shutil.copy(src, dst)  # Already copied with copytree above
            else:
                dst = output_folder / "base.nii.gz"
                # Create a relative symlink
                os.symlink(
                    os.path.relpath(
                        image_loc,
                        output_folder
                    ),
                    dst
                )
                
    print(f"Predicted HVPG value: {y_pred[0]:.2f} mmHg")
                
    return y_pred[0]