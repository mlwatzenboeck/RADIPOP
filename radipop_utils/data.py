""" load_data.py
Some utility functions to reload the radiomics data and combine it with the HVPG 
value in one dataframe.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal, Union, Dict
from glob import glob


import radipop_utils 
import radipop_utils.utils
from radipop_utils.utils import get_files_dict_by_regex_pattern

import sklearn
from sklearn.preprocessing  import StandardScaler


def get_HVPG_values_and_radiomics_paths(hvpg_data_file: Union[str, Path], radiomics_dir: Union[str, Path]):

    df = pd.read_excel(hvpg_data_file)

    strict = False
    dct_paths = get_files_dict_by_regex_pattern(radiomics_dir, regex_pattern="^Features_liver", strict=strict)
    df_dirs_features_liver = pd.DataFrame.from_records({ 'id': dct_paths.keys(), 'radiomics-features: liver': dct_paths.values() })

    dct_paths = get_files_dict_by_regex_pattern(radiomics_dir, regex_pattern="^Features_spleen", strict=strict)
    df_dirs_features_spleen = pd.DataFrame.from_records({ 'id': dct_paths.keys(), 'radiomics-features: spleen': dct_paths.values() })

    # Merge the DataFrames on the 'id' column
    df = df.merge(df_dirs_features_liver, on='id', how='inner').merge(df_dirs_features_spleen, on='id', how='inner')
    
    # drop unnamed columns (index)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # TODO rm after strict 
    df['radiomics-features: liver'] = df['radiomics-features: liver'].apply(lambda x: x[0] if len(x)==1 else pd.NA)
    df['radiomics-features: spleen'] = df['radiomics-features: spleen'].apply(lambda x: x[0] if len(x)==1 else pd.NA)
    
    return df



def combined_radiomics_features(radiomics_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_r1 = radiomics_dataframes["liver"]
    df_r2 = radiomics_dataframes["spleen"]
    assert len(df_r1) == 1
    assert len(df_r2) == 1

    df_r1 = df_r1.loc[:, ~df_r1.columns.str.contains('^Unnamed')]
    df_r2 = df_r2.loc[:, ~df_r2.columns.str.contains('^Unnamed')]

    # Add prefixes to the columns
    df_r1 = df_r1.add_prefix('liver: ')
    df_r2 = df_r2.add_prefix('spleen: ')

    combined_df = pd.concat([df_r1, df_r2], axis=1)
    return combined_df


def load_and_combined_radiomics_features(df_paths: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    df_paths = df_paths.reset_index(drop=True)
    for i in range(len(df_paths)):
        patientid = df_paths.loc[i, 'id']
        file_r1 = df_paths.loc[i, 'radiomics-features: liver']
        file_r2 = df_paths.loc[i, 'radiomics-features: spleen']
        df_r1 = pd.read_excel(file_r1)
        df_r2 = pd.read_excel(file_r2)  
        combined_df = combined_radiomics_features({"liver": df_r1, "spleen": df_r2})
        combined_df['id'] = patientid
        dfs.append(combined_df)
        
    df_radiomics = pd.concat(dfs, axis=0)

    # Move column "patient_id" to be the first column
    cols = list(df_radiomics.columns)
    cols.insert(0, cols.pop(cols.index('id')))
    df_radiomics = df_radiomics[cols].reset_index(drop=True)

    return df_radiomics

def load_HVPG_values_and_radiomics(paths_and_hvpg_data_file, radiomics_dir):
    
    df = get_HVPG_values_and_radiomics_paths(hvpg_data_file=paths_and_hvpg_data_file, radiomics_dir=radiomics_dir)
    
    mandatory_cols = ["id", "y", "radiomics-features: liver", "radiomics-features: spleen"]
    for col in mandatory_cols:
        assert col in df.columns, f"Column {col} is missing in the data."

    # Check if the data is complete
    m = df["radiomics-features: liver"].isna() | df["radiomics-features: spleen"].isna()
    assert len(df[m])==0, f"some radiomics data is missing: Check {list(df[m]['id'])}"

    # drop not completed radiomics for now ... this does no harm if the data is complete
    df_  = df.dropna(subset=["radiomics-features: liver", "radiomics-features: spleen"])

    # load radiomics data for completed calcs
    df_radiomics = load_and_combined_radiomics_features(df_)
    df_merged = df.merge(df_radiomics, on='id', how='inner')

    return df_merged


def split_radiomics_data(df_merged):
    # final filtered dataframe 
    dff = df_merged.filter(regex="^id|^y|^set type|^Tr split|^liver|^spleen")

    # splitting the data was already done before hand. Otherwise you might want to do this now
    m_Tr = dff["set type"] == "Tr"
    m_iTs = dff["set type"] == "internal Ts"
    m_eTs = dff["set type"] == "external Ts"
    
    df_Tr  = dff[m_Tr]
    df_iTs = dff[m_iTs]
    df_eTs = dff[m_eTs]
        
    m_other = ~m_Tr & ~m_iTs & ~m_eTs
    df_other = dff[m_other]
    if len(df_other) > 0:
        print("Warning: Data other than Tr, internal Ts, external Ts found. This data will be ignored.")

    return df_Tr, df_iTs, df_eTs


def save_loaded_radiomics_for_quickload(df_Tr, df_eTs, df_iTs, save_path, verbose=True):
    df_Tr.to_csv(save_path / "df_Tr.csv", index=False)
    df_eTs.to_csv(save_path / "df_eTs.csv", index=False)
    df_iTs.to_csv(save_path / "df_iTs.csv", index=False)
    if verbose: 
        print(f"Data df_Tr.csv saved to: {save_path / 'df_Tr.csv'}")
        print(f"Data df_eTs.csv saved to: {save_path / 'df_eTs.csv'}")
        print(f"Data df_iTs.csv saved to: {save_path / 'df_iTs.csv'}")


def load_radiomics_for_quickload(save_path, verbose=True):
    df_Tr = pd.read_csv(save_path / "df_Tr.csv")
    df_eTs = pd.read_csv(save_path / "df_eTs.csv")
    df_iTs = pd.read_csv(save_path / "df_iTs.csv")
    if verbose: 
        print(f"Data df_Tr.csv, df_eTs.csv, df_iTs.csv loaded from: {save_path}")
    return df_Tr, df_eTs, df_iTs


def quickload_or_combine_radiomics_data(paths_and_hvpg_data_file, radiomics_dir, verbose=True):
    if os.path.exists(radiomics_dir / "df_Tr.csv") and os.path.exists(radiomics_dir / "df_eTs.csv") and os.path.exists(radiomics_dir / "df_iTs.csv"):
        df_Tr, df_eTs, df_iTs = load_radiomics_for_quickload(radiomics_dir, verbose=False)
        if verbose: 
            print(f"(Quick) loaded data from: {radiomics_dir}")
    else:
        if verbose: 
            print(f"csv files for quickloading not found. Recombining it from patient subfoders.")
        df_Tr, df_iTs, df_eTs = radipop_utils.data.load_HVPG_values_and_radiomics(paths_and_hvpg_data_file, radiomics_dir)
        save_loaded_radiomics_for_quickload(df_Tr, df_eTs, df_iTs, radiomics_dir, verbose=verbose)
    return df_Tr, df_eTs, df_iTs



def extract_CV_indices(df_Tr):
    df_Tr = df_Tr.reset_index(drop=True)
    split_indices_CV5_Tr = []
    for i in range(5):
        m = df_Tr["Tr split"] == i
        idx_split_tr = df_Tr[~m].index.to_numpy()
        idx_split_ts = df_Tr[m].index.to_numpy()
        split_indices_CV5_Tr.append([idx_split_tr, idx_split_ts])
    return split_indices_CV5_Tr


def preprocess_data(df_Tr, df_iTs, df_eTs, normalize_X=True):
    # extract np arrays
    X_Tr,  Y_Tr  = df_Tr.filter(regex="^liver|^spleen").values, df_Tr["y"].values
    X_iTs, Y_iTs = df_iTs.filter(regex="^liver|^spleen").values, df_iTs["y"].values
    X_eTs, Y_eTs = df_eTs.filter(regex="^liver|^spleen").values, df_eTs["y"].values
    assert df_Tr.columns.equals(df_iTs.columns)
    assert df_Tr.columns.equals(df_eTs.columns)

    if normalize_X:
        # Normalize mostly for numerical stability. Also important for ElasticNet and just good practice in general
        transformer = StandardScaler().fit(X_Tr)  # fit on training data only
        X_Tr = transformer.transform(X_Tr)
        X_iTs = transformer.transform(X_iTs)
        X_eTs = transformer.transform(X_eTs)

    return X_Tr, Y_Tr, X_iTs, Y_iTs, X_eTs, Y_eTs


# Note better used the version independent of data below 
def preprocess_data_single(df_Tr_for_fit, df, normalize_X=True, return_cols = False):
    # extract np arrays
    dff1 = df_Tr_for_fit.filter(regex="^liver|^spleen")
    dff2 = df.filter(regex="^liver|^spleen")
    assert dff1.columns.equals(dff2.columns)
    X_Tr= dff1.values
    X = dff2.values

    if normalize_X:
        transformer = StandardScaler().fit(X_Tr)  # fit on training data only
        X = transformer.transform(X)
    if return_cols:
        return X, dff2.columns
    else:
        return X
    
def make_normalization_df(df):
    c1 = list(df.columns)
    c2 = list(df.mean())
    c3 = list(df.std())
    normalization_df = pd.DataFrame({"name": c1, "mean": c2, "std": c3}).set_index("name").transpose()
    return normalization_df

def make_and_save_normalization_df(df, model_dir, verbose=True):
    normalization_df = make_normalization_df(df)
    normalization_df.to_csv(model_dir / "normalization.csv")
    if verbose:
        print(f"Normalization dataframe saved to: {model_dir / 'normalization.csv'}")
    return normalization_df

def make_scaler_from_normalization_df(normalization_df):
    mean = normalization_df.loc["mean"].values
    std = normalization_df.loc["std"].values
    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = std**2
    scaler.n_samples_seen_ = len(mean)
    scaler.feature_names_in_ = normalization_df.columns
    return scaler

def load_normalization_df(model_dir):
    normalization_df = pd.read_csv(model_dir / "normalization.csv", index_col=0)
    return normalization_df
