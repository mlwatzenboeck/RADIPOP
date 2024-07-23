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


