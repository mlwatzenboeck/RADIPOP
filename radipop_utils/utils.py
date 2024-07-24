import os
import sys
import csv

from dotenv import load_dotenv, dotenv_values

from glob import glob
from typing import List, Union
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import nibabel as nib
from tqdm import tqdm
import pydicom
import dicom2nifti
from glob import glob
import tempfile
import shutil
import re 
from collections import defaultdict
from radiomics import featureextractor

import SimpleITK as sitk
import SimpleITK  # for type hinting


def dcm2nii(dicom_folder: Path, output_folder: Path, verbose: bool = True, out_id: Optional[str] = None) -> None:
    """
    Convert DICOM files in a folder to NIfTI format and save the result to the output folder.

    Args:
        dicom_folder (Path): Path to the folder containing DICOM files.
        output_folder (Path): Path to the output folder where the converted NIfTI files will be saved.
        verbose (bool, optional): If True, print conversion details. Defaults to True.
        out_id (str, optional): Identifier for the output folder. If None, the patient ID extracted from DICOM files will be used. Defaults to None.

    Returns:
        None

    Raises:
        AssertionError: If the patient ID extracted from DICOM files is not in the expected format.

    Notes:
        This function uses pydicom and dicom2nifti libraries to perform the conversion.

    Example:
        dcm_folder = Path('/path/to/dicom/folder')
        output_folder = Path('/path/to/output')
        dcm2nii(dcm_folder, output_folder, verbose=True, out_id=None)
        # Converted id: [patient_id] from [dicom_folder] to [output_folder/patient_id]
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(str(tmp))

        # convert dicom directory to nifti
        dicom2nifti.convert_directory(dicom_folder, str(tmp),
                                      compression=True, reorient=True)

        #looks for the first NIfTI file (*nii.gz) in temp
        nii = next(tmp.glob('*nii.gz'))

        if out_id == None:
            # get patient_id from dicom data:
            single_slices = glob(f"{dicom_folder}/*")
            ds = pydicom.filereader.dcmread(single_slices[0])
            assert str(ds.PatientName)[:3] == "ID_", (f"PatientName {ds.PatientName} does not start with 'ID_'. Did you make sure to pysdonomized data? \n" + 
                                                      "Hint: You can continue nontheless by providing the 'out_id' as a keyword.")
            id = int(str(ds.PatientName)[3:]) # check that it can be converted to int
            # patient_id = f"patient{str(id).zfill(3)}"

            output_folder_new = output_folder / str(ds.PatientName)[3:]
        else: 
            output_folder_new = output_folder / str(out_id)
            id = out_id

        os.makedirs(output_folder_new, exist_ok=True)
        
        # copy nifti file to the specified output path and named it 'base.nii.gz'
        shutil.copy(nii, output_folder_new / 'base.nii.gz')

        if verbose:
            print(f"Converted id: {id}  from {dicom_folder}  to {output_folder_new}")


def get_files_dict_by_regex_pattern(base_path, regex_pattern, strict=True):
    """
    Retrieve filenames matching a regex pattern within patient subdirectories.

    Parameters
    ----------
    base_path : Path
        The base directory containing patient subdirectories.
    regex_pattern : str
        The regular expression pattern to match the filenames.
    strict : bool, optional
        If True, assert that each patient subdirectory contains exactly one file matching the pattern (default is True).

    Returns
    -------
    dict
        A dictionary where keys are patient subdirectory names and values are lists of matched filenames.
    """
    # Compile the regular expression pattern for matching files
    pattern = re.compile(regex_pattern)
    
    # Initialize a dictionary to hold the results
    results = defaultdict(list)
    
    # Iterate over each subdirectory in the base path
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Get the name of the subdirectory (e.g., patient0001)
            patient_name = subdir.name
            
            # Find all files in the subdirectory that match the regex pattern
            matched_files = [f for f in subdir.iterdir() if f.is_file() and pattern.match(f.name)]
            
            if strict:
                assert len(matched_files) == 1, (
                    f"Expected exactly one file matching the pattern in {subdir}, but found {len(matched_files)}." + 
                    "\n Files in dir:" +  "\n".join([file.name for file in subdir.iterdir() if file.is_file()])
                    )
                
            if strict:
                # Add the matched files to the results dictionary
                results[patient_name] = matched_files[0]
            else:
                results[patient_name] = matched_files
    
    return dict(results)



def suggest_radiomics_binwidth(img: SimpleITK.SimpleITK.Image,
                               mask: SimpleITK.SimpleITK.Image, 
                               settings: dict = {
                                   "binWidth": 1, 
                                   "resampledPixelSpacing": [1, 1, 1],  # use None for no resampling
                                   "interpolator": sitk.sitkBSpline, 
                                   "padDistance": 10,
                               }):
    """Run extraction with only the first-order feature: Range to suggest a decent binwidth
    
    Make sure to provide the other settings according consistent with your yaml file
    """


    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Disable all classes except firstorder
    extractor.disableAllFeatures()

    # Only enable Range in firstorder
    extractor.enableFeaturesByName(firstorder=['Range'])

    print("Calculating features")
    extracted_features = extractor.execute(img, mask)

    features = {}
    for key in extracted_features.keys():
        if not key.startswith("diagnostics_"):
            features[key] = np.float64(extracted_features[key])       
    features = pd.DataFrame(features, index = [0])
    # display(features)

    r = features.loc[0, "original_firstorder_Range"]
    ratio = r / settings['binWidth']   # should be ~16-128 bins

    print(f"Range / binWidth =  {ratio}  (should be 16-128)")
    
    print(f"Suggestion:  Use a binwith of ~ ", r / ((128 + 16)*0.5))
    return r / ((128 + 16)*0.5)