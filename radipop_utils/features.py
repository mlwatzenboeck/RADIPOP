import os
import pydicom
from pathlib import Path
import nibabel as nib
import dicom2nifti
import numpy as np
from matplotlib import pyplot as plt
import importlib
import pickle
import SimpleITK as sitk
import SimpleITK # for type hinting
from typing import List, Union, Dict
from radiomics import featureextractor as extractor
import pandas as pd
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
from PIL import Image



def extract_radiomics_features(image: SimpleITK.SimpleITK.Image,
                               binary_mask: SimpleITK.SimpleITK.Image,
                               fe_settings_path: Union[str, Path]) -> pd.DataFrame:
    """
    Extracts radiomics features from a given medical image using a specified mask and feature extraction settings.

    Args:
        image (SimpleITK.SimpleITK.Image): The medical image from which to extract radiomics features.
        binary_mask (SimpleITK.SimpleITK.Image): The binary mask image where 1 indicates the region of interest.
        fe_settings_path (Union[str, Path]): Path to the radiomics feature extraction settings file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted radiomics features.
        
    Raises:
        AssertionError: If the mask is not binary.
        
    Example:
        features = extract_radiomics_features(image, binary_mask, 'path/to/settings.yaml')
    """
    
    set_of_unique_elements = set(np.unique(sitk.GetArrayFromImage((binary_mask))))
    assert set_of_unique_elements == {0, 1}, f"A binary mask is expected but mask contains: {set_of_unique_elements}"
    
    fe = extractor.RadiomicsFeatureExtractor(str(fe_settings_path))
    extracted_features = fe.execute(image, binary_mask)

    features = {}
    for key in extracted_features.keys():
        if not key.startswith("diagnostics_"):
            features[key] = np.float64(extracted_features[key])       
    features = pd.DataFrame(features, index = [0])
    return features



def extract_and_save_features_from_nii(patientid: str, image_loc: Union[Path, str], mask_loc: Union[Path, str], 
                                       output_dir : Union[Path, str], 
                                       fe_settings_path: Union[str, Path], 
                                       tissue_class_dct : Dict[str, int] = {"liver": 1, "spleen": 2},
                                       check_existence=True,
                                       verbose = True) -> None:
    """
    Extracts radiomics features from NIfTI images and saves them to specified output directory.

    Args:
        patientid (str): Identifier for the patient = name of the folder containing the features
        image_loc (Union[Path, str]): Path to the NIfTI image file.
        mask_loc (Union[Path, str]): Path to the NIfTI mask file. Must have a unique integer for each tissue type matching `tissue_class_dct`
        output_dir (Union[Path, str]): Directory where the extracted features will be saved.
        fe_settings_path (Union[str, Path]): Path to the radiomics feature extraction settings file.
        tissue_class_dct (Dict[str, int], optional): Dictionary mapping tissue types to their corresponding mask values. Default is {"liver": 1, "spleen": 2}.
        check_existence (bool, optional): If True, checks if the features file already exists and skips extraction if it does. Default is True.
        verbose (bool, optional): If True, prints progress messages. Default is True.

    Returns:
        None
        
    Example:
        extract_and_save_features_from_nii(
            patientid='patient0001', 
            image_loc='path/to/image.nii', 
            mask_loc='path/to/mask.nii', 
            output_dir='path/to/output', 
            fe_settings_path='path/to/settings.yaml'
        )
    """
    
    mask = sitk.ReadImage(mask_loc)
    img = sitk.ReadImage(image_loc)
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir / patientid, exist_ok=True)
    
    for tissue_type in tissue_class_dct.keys():
        out_file = output_dir / patientid / f"Features_{tissue_type}.xlsx"
        if check_existence and os.path.isfile(out_file): 
            if verbose:
                print(f"Radiomics features already exists in {out_file}. Skipped!")
        else: 
            features_df = extract_radiomics_features(img, mask == tissue_class_dct[tissue_type], fe_settings_path)
            features_df.to_excel(out_file)
            if verbose:
                print(f"Radiomics features saved to {out_file}.")
                                
    return None



# TODO add to repo or remove
# import conversion_utils_mw_with_reporting_function as cu

# TODO: clean up / maybe remove  this is the old code
def convert_and_extract(dcm, mask):
##define directories
    idx = mask
    print(idx)

    dicom_dir = os.path.join("clean_DICOM", dcm)
    first_conversion_dir = os.path.join("output", idx)
    output_dir = os.path.join("output", idx, "output_png")
    png_dir = os.path.join(output_dir, "png")
    
    output_liver_cond = os.path.isfile(os.path.join(output_dir, "Features_liver.xlsx"))
    output_spleen_cond = os.path.isfile(os.path.join(output_dir, "Features_spleen.xlsx"))
    
    if output_liver_cond and  output_spleen_cond: 
        print("Radiomics features already extracted!")
        return True

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(png_dir):
        os.makedirs(png_dir)

    #read the input image
    files = [pydicom.dcmread(os.path.join(dicom_dir, f), force = True) for f in os.listdir(dicom_dir)]
    files = [x for x in files if "ImagePositionPatient" in dir(x)]
    patient_position = files[0].PatientPosition
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=reverse)

    #loop over files
    for i, file in enumerate(files):
        ext = i
        orig = cu.extract_pixels_for_viewing(file)
        aim = Image.fromarray(orig, mode='L')
        aim.save(os.path.join(png_dir, str(ext)+".png"), format='PNG')


    #read the images
    images = np.array([x for x in os.listdir(png_dir) if x.endswith(".png")])
    indices = np.array([x.strip(".png") for x in images], dtype = int)
    images = images[np.argsort(indices)]
    images = np.array([plt.imread(os.path.join(png_dir, x)) for x in images])
    images = images[::-1,::-1,:]

    #convert to image object
    img = sitk.GetImageFromArray(images)

    #get the filepath for the nifti that was produced from directly converting dicom to nifti via dicom2nifti 
    fp_nifti_image1 = [x for x in os.listdir(first_conversion_dir) if x.endswith("nii.gz") and not x.startswith("mask")][0]
    img.CopyInformation(sitk.ReadImage(os.path.join(first_conversion_dir, fp_nifti_image1)))
    sitk.WriteImage(img, os.path.join(output_dir, "image_from_png.nii.gz"))

    #radiomics feature extraction
    mask_liver = sitk.ReadImage(os.path.join(first_conversion_dir, "mask_liver.nii.gz"))
    mask_spleen = sitk.ReadImage(os.path.join(first_conversion_dir, "mask_spleen.nii.gz"))

    print("Extracting liver features.")
    fe = extractor.RadiomicsFeatureExtractor("C://Users/marti/OneDrive - CeMM Research Center GmbH/stuff/sma/feature_extraction/exampleCT.yaml")
    features_liver = fe.execute(img, mask_liver)

    features_liver_df = {}
    for key in features_liver.keys():
        if not key.startswith("diagnostics_"):
            features_liver_df[key] = np.float64(features_liver[key])

    features_liver_df = pd.DataFrame(features_liver_df, index = [0])
    features_liver_df.to_excel(os.path.join(output_dir, "Features_liver.xlsx"))


    #extract spleen features
    print("Extracting spleen features.")
    features_spleen = fe.execute(img, mask_spleen)
    features_spleen_df = {}
    for key in features_spleen.keys():
        if not key.startswith("diagnostics_"):
            features_spleen_df[key] = np.float64(features_spleen[key])

    features_spleen_df = pd.DataFrame(features_spleen_df, index = [0])
    features_spleen_df.to_excel(os.path.join(output_dir, "Features_spleen.xlsx"))
    print("Feature extraction {} done!".format(output_dir))