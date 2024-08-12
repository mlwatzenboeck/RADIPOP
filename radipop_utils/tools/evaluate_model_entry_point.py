import os
from pathlib import Path
import pandas as pd
from pprint import pprint
import argparse

import radipop_utils
import radipop_utils.visualization
import radipop_utils.features
import radipop_utils.utils
import radipop_utils.data
import radipop_utils.inference

import datetime

# load user/ system specific env variables:
from dotenv import dotenv_values, find_dotenv
# load environment variables as dictionary
config = dotenv_values(find_dotenv())
DATA_ROOT_DIRECTORY_DEFAULT = Path(config["DATA_ROOT_DIRECTORY"])

path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent


def main_function():

    parser = argparse.ArgumentParser(
        description='Evaluate model on training and test sets. For training set, rotating CV is used (and the model is retrained).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--model_dir', type=Path,
                        help='Path to the folder containing the models, ...')

    parser.add_argument('--data_Tr', type=Path,#, default=None,   # needed for oos_R2 -> mandatory
                        help='Path to the CSV file containing the training data.')
    
    parser.add_argument('--data_iTs', type=Path, default=None, 
                        help='Path to the CSV file containing the internal test data.')
    
    parser.add_argument('--data_eTs', type=Path, default=None, 
                        help='Path to the CSV file containing the external test data.')

    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory. (default: %(default)s -> results are saved to model_dir)')
    

    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    if args.outdir is None:
        args.outdir = args.model_dir
        print("  ->  Output directory is set to model_dir: ", args.outdir)
    print()
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst = args.outdir / f"args_settings__evaluate_model_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)

    model_dir = args.model_dir
    outdir = Path(args.outdir)
    data_Tr = args.data_Tr
    data_eTs = args.data_eTs
    data_iTs = args.data_iTs

    organs = ["liver", "spleen"]
    re_pattern = "^(" + "|".join(organs) + ")"
    

    # load_models_and_params
    loaded_models, loaded_params, models_bare = radipop_utils.inference.load_models_and_params(model_dir = model_dir)
    normalization_df = radipop_utils.data.load_normalization_df(model_dir = model_dir)
    scaler  = radipop_utils.data.make_scaler_from_normalization_df(normalization_df)
    

    df_Tr = pd.read_csv(data_Tr)
    print(f"{len(df_Tr)=}")
    X_Tr = scaler.transform(df_Tr.filter(regex=re_pattern)) 
    Y_Tr = df_Tr["y"].values

    ### Evaluate the models on training set with rotating CV
    # Load the data
    if data_Tr != None:
        # df_Tr = pd.read_csv(data_Tr)
        # print(f"{len(df_Tr)=}")
        
        # X_Tr = scaler.transform(df_Tr.filter(regex=re_pattern)) 
        # Y_Tr = df_Tr["y"].values
        
        split_indices_CV5_Tr = radipop_utils.data.extract_CV_indices(df_Tr)
        
        modelRF = models_bare["RF"].set_params(**loaded_params["RF"])
        modelEN = models_bare["EN"].set_params(**loaded_params["EN"])
        res_training = radipop_utils.inference.fit_and_prediction_CV5_training(modelRF, modelEN, X_Tr, Y_Tr, split_indices_CV5_Tr)

        dst = outdir / "raw_results_training_CV5.xlsx"
        res_training.to_excel(dst)
        print(f"Results saved to {dst}")
    
    
        # metrics:
        y_true = res_training["True_HVPG"]
        y_pred_RF = res_training["RF_HVPG"]
        y_pred_EN = res_training["EN_HVPG"]
        metrics_training_CV5 = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN) 
        dst = outdir / "metrics_training_CV5.xlsx"
        metrics_training_CV5.to_excel(dst)
        print(f"Results saved to {dst}")

        # load the model trained on the whole training set
        modelRF = loaded_models["RF"]
        modelEN = loaded_models["EN"]  

        # evaluate on training set (no good performance measure, but useful checking against overfitting)
        rf_res = modelRF.predict(X_Tr)
        en_res = modelEN.predict(X_Tr)
        res_Tr = pd.DataFrame({"True_HVPG" : Y_Tr, 
                                "RF_HVPG" : rf_res,
                                "EN_HVPG" : en_res})
        y_true = res_Tr["True_HVPG"]
        y_pred_RF = res_Tr["RF_HVPG"]
        y_pred_EN = res_Tr["EN_HVPG"]
        metrics_Tr = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN, y_train_mean=Y_Tr.mean())     
    
        #### export feature importances
        # <CW:> There might still be a bug in there. It crashed for one radiomics option. 
        # (Different lenght of feature names and feature importances, or similar)
        #
        # dst = OUTDIR / "regression" / RADIOMICS_OPTION / "Feature_importances_RF_regressor.xlsx"
        # feature_impRF = radipop_utils.inference.get_feature_importancesRF(df_Tr, X_Tr, Y_Tr, modelRF, loaded_params)
        # feature_impRF.to_excel(dst)
        # print(f"Feature importances saved to {dst}")
    
        dst_Tr = outdir / "raw_results_training_set.xlsx"
        res_Tr.to_excel(dst_Tr)
        print(f"Results saved to {dst_Tr}")

        dst_metrics_Tr = outdir / "metrics_training_set.xlsx"
        metrics_Tr.to_excel(dst_metrics_Tr)
        print(f"Metrics saved to {dst_metrics_Tr}")
        print()
    
    
    # load the model trained on the whole training set
    modelRF = loaded_models["RF"]
    modelEN = loaded_models["EN"]  
    
    # evaluate on the internal test set
    if data_iTs != None:
        df_iTs = pd.read_csv(data_iTs)
        print(f"{len(df_iTs)=}")
        
        X_iTs = scaler.transform(df_iTs.filter(regex=re_pattern)) 
        Y_iTs = df_iTs["y"].values

        rf_res = modelRF.predict(X_iTs)
        en_res = modelEN.predict(X_iTs)
        res_iTs = pd.DataFrame({"True_HVPG" : Y_iTs, 
                                "RF_HVPG" : rf_res,
                                "EN_HVPG" : en_res})
        y_true = res_iTs["True_HVPG"]
        y_pred_RF = res_iTs["RF_HVPG"]
        y_pred_EN = res_iTs["EN_HVPG"]
        metrics_iTs = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN, y_train_mean=Y_Tr.mean()) 

        
        dst_iTs = outdir / "raw_results_internal_test_set.xlsx"
        res_iTs.to_excel(dst_iTs)
        print(f"Results saved to {dst_iTs}")

        dst_metrics_iTs = outdir / "metrics_internal_test_set.xlsx"
        metrics_iTs.to_excel(dst_metrics_iTs)
        print(f"Metrics saved to {dst_metrics_iTs}")
        print()

    # evaluate on the external test set
    if data_eTs != None:
        df_eTs = pd.read_csv(data_eTs)
        print(f"{len(df_eTs)=}")
        
        X_eTs = scaler.transform(df_eTs.filter(regex=re_pattern)) 
        Y_eTs = df_eTs["y"].values

        rf_res = modelRF.predict(X_eTs)
        en_res = modelEN.predict(X_eTs)
        res_eTs = pd.DataFrame({"True_HVPG" : Y_eTs, 
                                "RF_HVPG" : rf_res,
                                "EN_HVPG" : en_res})
        y_true = res_eTs["True_HVPG"]
        y_pred_RF = res_eTs["RF_HVPG"]
        y_pred_EN = res_eTs["EN_HVPG"]
        metrics_eTs = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN, y_train_mean=Y_Tr.mean()) 
        
        dst_eTs = outdir / "raw_results_external_test_set.xlsx"
        res_eTs.to_excel(dst_eTs)
        print(f"Results saved to {dst_eTs}")

        dst_metrics_eTs = outdir / "metrics_external_test_set.xlsx"
        metrics_eTs.to_excel(dst_metrics_eTs)
        print(f"Metrics saved to {dst_metrics_eTs}")
        print()
    
    
    if data_Tr != None:
        # print some results to the console
        print("metrics_training:")
        print("="*20)
        pprint(metrics_Tr)
        print()
    
        print("metrics_training_CV5:")
        print("="*20)
        pprint(metrics_training_CV5)
        print()
    
    if data_iTs != None:
        # print some results to the console
        print("metrics_iTs:")
        print("="*20)
        pprint(metrics_iTs)
        print()

    if data_eTs != None:
        # print some results to the console
        print("metrics_eTs:")
        print("="*20)
        pprint(metrics_eTs)
        print()
        
    return None


if __name__=="__main__":
    main_function()
    