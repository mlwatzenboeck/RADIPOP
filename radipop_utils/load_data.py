""" load_data.py
Some utility functions to reload the radiomics data and combine it with the HVPG 
value in one dataframe.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Literal, Union
from glob import glob


import radipop_utils 
import radipop_utils.utils
from radipop_utils.utils import get_files_dict_by_regex_pattern





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


def read_and_combined_radiomics_features(df_paths: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    df_paths = df_paths.reset_index(drop=True)
    for i in range(len(df_paths)):

        patientid = df_paths.loc[i, 'id']
        file_r1 = df_paths.loc[i, 'radiomics-features: liver']
        file_r2 = df_paths.loc[i, 'radiomics-features: spleen']

        df_r1 = pd.read_excel(file_r1)  # these all have just a single row of data
        df_r2 = pd.read_excel(file_r2)  
        assert len(df_r1) == 1
        assert len(df_r2) == 1

        df_r1 = df_r1.loc[:, ~df_r1.columns.str.contains('^Unnamed')]
        df_r2 = df_r2.loc[:, ~df_r2.columns.str.contains('^Unnamed')]

        # Add prefixes to the columns
        df_r1 = df_r1.add_prefix('liver: ')
        df_r2 = df_r2.add_prefix('spleen: ')

        combined_df = pd.concat([df_r1, df_r2], axis=1)
        combined_df['id'] = patientid
        
        dfs.append(combined_df)
        
    df_radiomics = pd.concat(dfs, axis=0)

    # Move column "patient_id" to be the first column
    cols = list(df_radiomics.columns)
    cols.insert(0, cols.pop(cols.index('id')))
    df_radiomics = df_radiomics[cols].reset_index(drop=True)

    return df_radiomics
