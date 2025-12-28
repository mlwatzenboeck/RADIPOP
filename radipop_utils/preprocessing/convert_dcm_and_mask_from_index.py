from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import tqdm
import os
from pathlib import Path
import pydicom
import nibabel as nib
import dicom2nifti
import numpy as np
from matplotlib import pyplot as plt
import importlib
import pickle
import SimpleITK as sitk
from radiomics import featureextractor as extractor
import pandas as pd
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
from PIL import Image


import radipop_utils
import radipop_utils.conversion_utils_mw_with_reporting_function as cu
import radipop_utils.features
path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent



def check_inputs(folder_name: str, i_start: int, i_end: int, idx_shift: int) -> None:
    # some checks: 
    assert i_end > i_start
    dcm_folder = Path(folder_name)
    assert dcm_folder.exists() and dcm_folder.is_dir(), f"Folder does not exist: {dcm_folder}"

    # Check for required .dcm files
    missing_dcm_files = []
    for i in range(i_start, i_end + 1):
        dcm_filename = f"IM-0001-{i:04d}.dcm"
        dcm_path = dcm_folder / dcm_filename
        if not dcm_path.exists():
            missing_dcm_files.append(dcm_filename)

    if missing_dcm_files:
        print("Missing .dcm files:")
        for name in missing_dcm_files:
            print(f"  {name}")
        assert False, f"{len(missing_dcm_files)} .dcm files are missing from {folder_name}"

    # Check for required .p files (mask files)
    missing_p_files = []
    for i in range(i_start, i_end + 1):
        p_filename = f"{i - idx_shift}.p"
        p_path = dcm_folder / p_filename
        if not p_path.exists():
            missing_p_files.append(p_filename)

    if missing_p_files:
        print("Missing .p (mask) files:")
        for name in missing_p_files:
            print(f"  {name}")
        assert False, f"{len(missing_p_files)} .p files are missing from {folder_name}"


# Note: This function is from the original code and must not be changed!!!
# BUT it can serve as inspiration for the functions to be written: 
#  - convert_dcm_from_index(dcm_files: List[str], out_path:str)
#  - convert_masks_from_index(p_files: List[str], out_path:str)
# Or similar
def convert_and_extract2(dcm, mask, cut_indices, flipped_z_axis = False, flip_mask = True, liver_label = 1, spleen_label = 2):
    ##rerun for dicoms with different dicoms in folder

    #get the cut indices
    cut_indices = [int(x) for x in cut_indices.split("_")]

    ##define directories
    idx = mask
    print(idx)

    #input dirs
    dicom_dir = os.path.join("E://radipop/clean_DICOM", dcm)
    mask_dir = os.path.join("E://radipop/MASKS/", mask)

    #output dir for cleaned dcm and masks
    output_dicom = os.path.join("E://radipop/cut_DICOM", mask)
    output_mask = os.path.join("E://radipop/output/", mask)

    #output dirs for png conversion
    output_dir = os.path.join("E://radipop/output/", idx, "output_png")
    png_dir = os.path.join(output_dir, "png_cut")

    output_liver_cond = os.path.isfile(os.path.join(output_dir, "Features_liver.xlsx"))
    output_spleen_cond = os.path.isfile(os.path.join(output_dir, "Features_spleen.xlsx"))

    #disable conditionals to enable rerun
    #if output_liver_cond and  output_spleen_cond: 
    #    print("Radiomics features already extracted!")
    #    return True

    if not os.path.isdir(output_dicom):
        os.makedirs(output_dicom)

    if not os.path.isdir(output_mask):
        os.makedirs(output_mask)    

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(png_dir):
        os.makedirs(png_dir)

    #read the input image
    files = [pydicom.dcmread(os.path.join(dicom_dir, f), force = True) for f in os.listdir(dicom_dir)]
    files = [x for x in files if "ImagePositionPatient" in dir(x)]
    patient_position = files[0].PatientPosition
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.InstanceNumber), reverse=reverse)

    #cut and write the files
    files_cut = files[(cut_indices[0]-1):(cut_indices[1])]

    for i, j in enumerate(files_cut):
        pydicom.dcmwrite(os.path.join(output_dicom, str(i) + ".dcm"), j)

    #reread the dicom image
    files = [pydicom.dcmread(os.path.join(output_dicom, f), force = True) for f in os.listdir(output_dicom)]
    files = [x for x in files if "ImagePositionPatient" in dir(x)]
    patient_position = files[0].PatientPosition
    reverse = patient_position == "FFS"
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=reverse)

    #loop over and convert to png
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

    if flipped_z_axis:
        images = images[::-1,:,:]

    #convert to image object
    img = sitk.GetImageFromArray(images)

    #convert the dicom using dicom2nifti
    dicom2nifti.convert_directory(output_dicom, output_mask)

    #get the filepath for the nifti that was produced from directly converting dicom to nifti via dicom2nifti 
    fp_nifti_image1 = [x for x in os.listdir(output_mask) if x.endswith("nii.gz") and not x.startswith("mask")][0]

    #copy the information to the image
    img.CopyInformation(sitk.ReadImage(os.path.join(output_mask, fp_nifti_image1)))
    sitk.WriteImage(img, os.path.join(output_dir, "image_from_png.nii.gz"))

    #read the mask
    mask_files = [x for x in os.listdir(mask_dir) if x.endswith(".p") and not x.startswith("._")]

    file_position = [int(x.strip(".p")) for x in mask_files]
    #get masks aligned
    mask_files = np.array(mask_files)[np.argsort(file_position)]

    masks = [pickle.load(open(os.path.join(mask_dir, file), "rb")) for file in mask_files]

    if flip_mask:
        masks = masks[::-1]

    mask = np.stack(masks, axis = 0)
    print(mask.dtype)

    assert mask.shape[1:] == (512,512)

    mask_coords = [0, 0, 512, 512]

    mask_new = np.zeros((mask.shape[0], 512, 512))
    for zi in range(mask.shape[0]):
        mask_new[zi,mask_coords[0]:mask_coords[2], mask_coords[1]:mask_coords[3]] = mask[zi,:,:]
        mask_new[zi, :, :] = mask_new[zi, ::-1, :]

    #define orientation in z axis 
    #disable due to jens tool
    #testfile = os.listdir(dcm_dir)[0]
    #testfile = pydicom.dcmread(os.path.join(dcm_dir, testfile))
    #print("Patient orientation", testfile.PatientPosition)
    #if testfile.PatientPosition == "FFS":
    #    mask_new = mask_new[::-1, :, :]

    mask = mask_new
    print(mask.dtype)

    #cut the mask
    mask_cut = mask[(cut_indices[0]-1):(cut_indices[1])]


    #create masks for spleen and liver
    mask_liver = mask_cut.copy() 
    mask_liver[np.where(mask_liver != liver_label)] = 0
    mask_liver[np.where(mask_liver == liver_label)] = 1

    mask_spleen = mask_cut.copy() 
    mask_spleen[np.where(mask_spleen != spleen_label)] = 0
    mask_spleen[np.where(mask_spleen == spleen_label)] = 1

    #get mask for liver as nifti
    mask_liver = sitk.GetImageFromArray(mask_liver)
    mask_liver.CopyInformation(img)
    sitk.WriteImage(mask_liver, os.path.join(output_mask, "mask_liver.nii.gz"))

    #get mask for spleen as nifti
    mask_spleen = sitk.GetImageFromArray(mask_spleen)
    mask_spleen.CopyInformation(img)
    sitk.WriteImage(mask_spleen, os.path.join(output_mask, "mask_spleen.nii.gz"))

    #radiomics feature extraction
    #mask_liver = sitk.ReadImage(os.path.join(first_conversion_dir, "mask_liver.nii.gz"))
    #mask_spleen = sitk.ReadImage(os.path.join(first_conversion_dir, "mask_spleen.nii.gz"))

    print("Extracting liver features.")
    fe = extractor.RadiomicsFeatureExtractor(str(RADIPOP_PACKAGE_ROOT / "yaml" / "exampleCT.yaml"))
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



DEBUGGING = False

if __name__ == "__main__":
    if DEBUGGING:
        folder_name = "/home/clemens/data/RADIPOP_EXTRA/working_env/FINAL.38"
        i_start=1
        i_end=230
        idx_shift=1
        folder_name_out= "__AUTO__"
        out_root=None
    else: 
        parser = argparse.ArgumentParser(description="Separate and link DICOM files by acquisition time")
        parser.add_argument("-f", "--folder_name", type=str, help="Path to the folder containing the DICOM files", required=True)
        parser.add_argument("--i_start", type=int, help="Start index for .dcm files", required=True)
        parser.add_argument("--i_end", type=int, help="Start index for .dcm files", required=True)
        parser.add_argument("--folder_name_out", type=str, default="__AUTO__")
        parser.add_argument("--idx_shift", type=int, help="idx_dcm - idx_shift = idx_mask; [default=1; 0.png -> ...-001.dcm]", default=1)
        parser.add_argument("-o", "--out_root", type=str, help="Path to the output folder; Default: same as folder_name", default=None)
        args = parser.parse_args()
        
        folder_name = args.folder_name
        i_start = args.i_start
        i_end = args.i_end
        folder_name_out = args.folder_name_out
        idx_shift = args.idx_shift
        out_root = args.out_root

    if folder_name_out == "__AUTO__":
        folder_name_out = f"merged_{i_start}-{i_end}"
    if out_root == None: 
        out_root = folder_name
    output_folder = f"{out_root}/{folder_name_out}"


    check_inputs(folder_name, i_start, i_end, idx_shift)

    # TODO: 
    # Combine .dcm files to nifty file. 
    # Maybe need to first create links in temorary directory and then convert with 
    #     radipop_utils.utils.dcm2nii(
    #     dicom_folder=dicom_folder,
    #     output_folder=output_folder,
    #     out_id="nii",
    #     verbose=verbose
    # )
    

    # TODO: 
    # create list for correspondong .p files. And merge in similar way as for convert_and_extract2

    
