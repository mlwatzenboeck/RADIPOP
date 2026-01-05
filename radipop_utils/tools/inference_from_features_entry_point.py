import os
from pathlib import Path
import pandas as pd
from pprint import pprint
import argparse
import json

import radipop_utils
import radipop_utils.data
import radipop_utils.inference
import radipop_utils.utils

import datetime


def main_function():

    parser = argparse.ArgumentParser(
        description='Run inference on pre-extracted radiomics features from Features_liver.xlsx and Features_spleen.xlsx files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--features_folder', type=Path, required=True,
                        help='Path to the folder containing Features_liver.xlsx and Features_spleen.xlsx files.')
    
    parser.add_argument('--model_dir', type=Path, required=True,
                        help='Path to the folder containing the trained models.')
    
    parser.add_argument('--out_name', type=str, default='predictions',
                        help='Base name for the output JSON file (without .json extension). Default: predictions')

    args = parser.parse_args()
    args_dict = vars(args)
    print(f"Running: '{Path(__file__).name}' with the following arguments:")
    print("---------------")
    pprint(args_dict)
    print()
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    dst = args.features_folder / f"args_settings__inference_from_features_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)

    features_folder = Path(args.features_folder)
    model_dir = Path(args.model_dir)
    
    # Check that features folder exists
    if not features_folder.exists():
        raise FileNotFoundError(f"Features folder does not exist: {features_folder}")
    
    # Check that model directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    # Check that required feature files exist
    features_liver_path = features_folder / "Features_liver.xlsx"
    features_spleen_path = features_folder / "Features_spleen.xlsx"
    
    if not features_liver_path.exists():
        raise FileNotFoundError(f"Features_liver.xlsx not found in: {features_folder}")
    
    if not features_spleen_path.exists():
        raise FileNotFoundError(f"Features_spleen.xlsx not found in: {features_folder}")
    
    print(f"Loading features from: {features_folder}")
    
    # Load features
    df_liver = pd.read_excel(features_liver_path)
    df_spleen = pd.read_excel(features_spleen_path)
    
    print(f"Loaded liver features: {len(df_liver.columns)} columns, {len(df_liver)} rows")
    print(f"Loaded spleen features: {len(df_spleen.columns)} columns, {len(df_spleen)} rows")
    
    # Combine features
    radiomics_dataframes = {"liver": df_liver, "spleen": df_spleen}
    df_combined = radipop_utils.data.combined_radiomics_features(radiomics_dataframes)
    
    print(f"Combined features: {len(df_combined.columns)} columns, {len(df_combined)} rows")
    
    # Filter features using regex pattern to match model expectations
    organs = ["liver", "spleen"]
    re_pattern = "^(" + "|".join(organs) + ")"
    df_X = df_combined.filter(regex=re_pattern)
    
    print(f"Filtered features for model: {len(df_X.columns)} columns")
    
    # Load models
    print(f"Loading models from: {model_dir}")
    loaded_models, loaded_params, models_bare = radipop_utils.inference.load_models_and_params(model_dir=model_dir)
    
    modelRF = loaded_models["RF"]
    modelEN = loaded_models["EN"]
    
    print("Models loaded successfully")
    
    # Run inference
    print("Running inference...")
    rf_pred = modelRF.predict(df_X)
    en_pred = modelEN.predict(df_X)
    
    # Extract single prediction value (should be a single value for one patient)
    # sklearn predict() returns numpy array, so we take the first (and only) element
    rf_value = float(rf_pred[0])
    en_value = float(en_pred[0])
    
    print(f"RF prediction: {rf_value:.4f}")
    print(f"EN prediction: {en_value:.4f}")
    
    # Save results as JSON
    predictions = {
        "RF_HVPG": rf_value,
        "EN_HVPG": en_value
    }
    
    output_json_path = features_folder / f"{args.out_name}.json"
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    print(f"Predictions saved to: {output_json_path}")
    
    return None


if __name__=="__main__":
    main_function()

