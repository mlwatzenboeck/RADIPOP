import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pickle
import joblib
from pprint import pprint
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

import radipop_utils
import radipop_utils.features
from radipop_utils.features import SpearmanReducerCont
import radipop_utils.utils
import radipop_utils.data

import datetime


def load_hyperparams_from_yaml(model_dir: Path, model_type: str) -> dict:
    """Load hyperparameters from YAML file."""
    filename = model_dir / f"SpearmanRed1_{model_type}_opt_params.yml"
    if not filename.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {filename}")
    with open(filename, 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_hyperparams_from_excel(model_dir: Path, num_searches: int) -> tuple:
    """Load optimal hyperparameters from Excel file (Bayesian search results)."""
    # Find the Excel file
    excel_files = list(model_dir.glob("Bayesian_results_*_iterations_RFvsEN.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No Bayesian_results Excel file found in {model_dir}")
    
    cv_res = pd.read_excel(excel_files[0])
    
    # Find best models
    idx_best_EN_model = cv_res["mean_test_score"][:num_searches].argmax()
    idx_best_RF_model = cv_res["mean_test_score"][num_searches:].argmax() + num_searches
    
    # Extract parameters
    params_RF = dict(cv_res.iloc[idx_best_RF_model, :].params)
    params_EN = dict(cv_res.iloc[idx_best_EN_model, :].params)
    
    # Remove 'regression' key if present (it's the estimator object, not a hyperparameter)
    if "regression" in params_RF:
        params_RF.pop("regression")
    if "regression" in params_EN:
        params_EN.pop("regression")
    
    return params_RF, params_EN


def main_function():
    parser = argparse.ArgumentParser(
        description='Train models with optimal hyperparameters and save in multiple formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_Tr', type=Path,
                        help='Path to the CSV file containing the training data.')
    
    parser.add_argument('--model_dir', type=Path,
                        help='Directory containing optimal hyperparameters (YAML files) OR hyperparameter search results Excel file.')
    
    parser.add_argument('--outdir', type=Path, 
                        help='Output directory for trained models.')
    
    parser.add_argument('--hyperparams_source', type=str, default="auto",
                        choices=["auto", "yaml", "excel"],
                        help='Source for hyperparameters: "auto" (detect automatically), "yaml" (from YAML files), or "excel" (from Bayesian search results). (default: %(default)s)')
    
    parser.add_argument('--num_searches', type=int, default=10,
                        help='Number of hyperparameter searches (only needed if loading from Excel). (default: %(default)s)')
    
    args = parser.parse_args()
    args_dict = vars(args)
    print(f"Running: '{Path(__file__).name}' with the following arguments:")
    print("---------------")
    pprint(args_dict)
    print()
    
    # make out dir if not exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    dst = args.outdir / f"args_settings__train_with_optimal_hp_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)

    training_data_csv_file = Path(args.data_Tr)
    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    
    organs = ["liver", "spleen"]
    
    # load data, filter to relevant features only
    df_Tr = pd.read_csv(training_data_csv_file)
    print(f"Loaded data from {training_data_csv_file} with {len(df_Tr)=}") 
    re_pattern = "^(" + "|".join(organs) + ")"
    df_Tr_X, df_Tr_y = df_Tr.filter(regex=re_pattern), df_Tr["y"]

    # normalize the data and save normalization df
    normalization_df = radipop_utils.data.make_and_save_normalization_df(df_Tr_X, outdir, verbose=True)
    
    # Load optimal hyperparameters
    hyperparams_source = args.hyperparams_source
    if hyperparams_source == "auto":
        # Try YAML first, then Excel
        yaml_rf = model_dir / "SpearmanRed1_RF_opt_params.yml"
        yaml_en = model_dir / "SpearmanRed1_EN_opt_params.yml"
        if yaml_rf.exists() and yaml_en.exists():
            hyperparams_source = "yaml"
            print("Auto-detected: Using YAML files for hyperparameters")
        else:
            hyperparams_source = "excel"
            print("Auto-detected: Using Excel file for hyperparameters")
    
    if hyperparams_source == "yaml":
        params_RF = load_hyperparams_from_yaml(model_dir, "RF")
        params_EN = load_hyperparams_from_yaml(model_dir, "EN")
        print("Loaded hyperparameters from YAML files")
    else:  # excel
        params_RF, params_EN = load_hyperparams_from_excel(model_dir, args.num_searches)
        print("Loaded hyperparameters from Excel file")
    
    print("\nOptimal hyperparameters for Random Forest:")
    pprint(params_RF)
    print("\nOptimal hyperparameters for ElasticNet:")
    pprint(params_EN)
    print()

    # Create and fit Random Forest model
    print("Fitting Random Forest model...")
    reg_RF = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SpearmanReducerCont()),
        ('regression', RandomForestRegressor())
    ])
    reg_RF.set_output(transform="pandas")
    
    # Set params (regression step is already defined in pipeline)
    np.random.seed(2023)
    reg_RF.set_params(**params_RF)
    reg_RF.fit(df_Tr_X, df_Tr_y.values)
    
    # Save RF model in both formats
    dst_pickle = outdir / "SpearmanRed1_RF_opt.p"
    with open(dst_pickle, "wb") as fp:
        pickle.dump(reg_RF, fp)
        print(f"Saved RF model (pickle) to: {dst_pickle}")
    
    dst_joblib = outdir / "SpearmanRed1_RF_opt.joblib"
    joblib.dump(reg_RF, dst_joblib)
    print(f"Saved RF model (joblib) to: {dst_joblib}")

    # Create and fit ElasticNet model
    print("\nFitting ElasticNet model...")
    reg_EN = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SpearmanReducerCont()),
        ('regression', ElasticNet())
    ])
    reg_EN.set_output(transform="pandas")
    
    # Set params (regression step is already defined in pipeline)
    reg_EN.set_params(**params_EN)
    reg_EN.fit(df_Tr_X, df_Tr_y.values)
    
    # Save EN model in both formats
    dst_pickle = outdir / "SpearmanRed1_EN_opt.p"
    with open(dst_pickle, "wb") as fp:
        pickle.dump(reg_EN, fp)
        print(f"Saved EN model (pickle) to: {dst_pickle}")
    
    dst_joblib = outdir / "SpearmanRed1_EN_opt.joblib"
    joblib.dump(reg_EN, dst_joblib)
    print(f"Saved EN model (joblib) to: {dst_joblib}")
    
    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main_function()
    print("Done!")

