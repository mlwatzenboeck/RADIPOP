import os
import importlib
import pickle
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import nibabel as nib
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
import SimpleITK as sitk
import SimpleITK  # for type hinting
import skopt
import numba
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from radiomics import featureextractor as extractor
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score, roc_curve,    r2_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Disable validation of slice increment for dicom2nifti
# settings.disable_validate_slice_increment()


import radipop_utils
import radipop_utils.conversion_utils_mw_with_reporting_function as cu
# extract path of repo. Needed for to find ./yaml/exampleCT.yaml with settings for the radiomics feature extractor 
path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent




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
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir / patientid, exist_ok=True)
    
    # skip before loading the image
    if check_existence: 
        can_be_skipped = True
        for tissue_type in tissue_class_dct.keys():
            out_file = output_dir / patientid / f"Features_{tissue_type}.xlsx"
            if not os.path.isfile(out_file): 
                can_be_skipped = False
                break
        if can_be_skipped:
            if verbose:
                print(f"All Radiomics features already exists in {output_dir / patientid}. Skipped!")
            return None
    
    mask = sitk.ReadImage(mask_loc)
    img = sitk.ReadImage(image_loc)
    
    
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





def get_most_corr(f, data_train, Y_train):
    "Get the most associated variable from a cluster"
    if len(f) == 0:
        return None
    corrcoefs = [spearmanr(Y_train, data_train[:,x])[0] for x in f]
    return f[np.argmax(np.abs(corrcoefs))]


class SpearmanReducerCont(BaseEstimator, TransformerMixin):
    "custom feature reduction method based on spearman correlations"
        
    def __init__(self, split_param = 1):
        self.split_param = split_param
    
    def fit(self, X, y=None):
        
        if self.split_param is None:
            
            self.selected_features = list(np.arange(X.shape[1]))
            return self
        
        else:
            #calculate correlation matrix
            corr = spearmanr(X).correlation

            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)

            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(corr)
            dist_linkage = hierarchy.ward(squareform(distance_matrix))
            cluster_ids = hierarchy.fcluster(dist_linkage, self.split_param, criterion="distance")
            cluster_id_to_feature_ids = defaultdict(list)
            for idx, cluster_id in enumerate(cluster_ids):
                cluster_id_to_feature_ids[cluster_id].append(idx)

            self.selected_features = [get_most_corr(v, X, y) for v in cluster_id_to_feature_ids.values()]

            return self

    def transform(self, X, y=None):
        #print(self.selected_features)
        # Perform transformation
        return X[:, self.selected_features]

