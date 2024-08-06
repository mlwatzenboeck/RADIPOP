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

    parser.add_argument('--data_root_directory', type=str, default=str(DATA_ROOT_DIRECTORY_DEFAULT),
                        help='Root directory of the data. (default: %(default)s)')
    parser.add_argument('--outdir', type=str, default=str(DATA_ROOT_DIRECTORY_DEFAULT / "radiomics" / "Dataset125_LSS"),
                        help='Output directory. (default: %(default)s)')
    parser.add_argument('--dataset', type=str, default="Dataset125_LSS",
                        help='Dataset name. (default: %(default)s)')
    parser.add_argument('--radiomics_option', type=str, default="radipop_111",
                        help='Radiomics option. (default: %(default)s)')

    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    print()

    DATA_ROOT_DIRECTORY = Path(args.data_root_directory)
    DATASET = args.dataset
    OUTDIR = Path(args.outdir)
    RADIOMICS_OPTION = args.radiomics_option


    # load the data
    df_Tr, df_iTs, df_eTs = radipop_utils.data.quickload_or_combine_radiomics_data(DATASET=DATASET, RADIOMICS_OPTION=RADIOMICS_OPTION, DATA_ROOT_DIRECTORY=DATA_ROOT_DIRECTORY, verbose=True)
    print(f"{len(df_Tr)=}, {len(df_eTs)=}, {len(df_iTs)=}")
    split_indices_CV5_Tr = radipop_utils.data.extract_CV_indices(df_Tr)
    X_Tr, Y_Tr, X_iTs, Y_iTs, X_eTs, Y_eTs = radipop_utils.data.preprocess_data(df_Tr, df_iTs, df_eTs, normalize_X=True)


    # load_models_and_params
    model_dir = DATA_ROOT_DIRECTORY / "radiomics" / DATASET / "regression" / RADIOMICS_OPTION
    loaded_models, loaded_params, models_bare = radipop_utils.inference.load_models_and_params(model_dir = model_dir)
    
    ### Evaluate the models on training set with rotating CV
    modelRF = models_bare["RF"].set_params(**loaded_params["RF"])
    modelEN = models_bare["EN"].set_params(**loaded_params["EN"])
    res_training = radipop_utils.inference.fit_and_prediction_CV5_training(modelRF, modelEN, X_Tr, Y_Tr, split_indices_CV5_Tr)
    dst = OUTDIR / "regression" / RADIOMICS_OPTION / "raw_results_training_CV5.xlsx"
    res_training.to_excel(dst)
    print(f"Results saved to {dst}")
    
    
    # metrics:
    y_true = res_training["True_HVPG"]
    y_pred_RF = res_training["RF_HVPG"]
    y_pred_EN = res_training["EN_HVPG"]
    metrics_training_CV5 = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN) 
    dst = OUTDIR / "regression" / RADIOMICS_OPTION / "metrics_training_CV5.xlsx"
    metrics_training_CV5.to_excel(dst)
    print(f"Results saved to {dst}")
    
    
    #### export feature importances
    # <CW:> There might still be a bug in there. It crashed for one radiomics option. 
    # (Different lenght of feature names and feature importances, or similar)
    #
    # dst = OUTDIR / "regression" / RADIOMICS_OPTION / "Feature_importances_RF_regressor.xlsx"
    # feature_impRF = radipop_utils.inference.get_feature_importancesRF(df_Tr, X_Tr, Y_Tr, modelRF, loaded_params)
    # feature_impRF.to_excel(dst)
    # print(f"Feature importances saved to {dst}")
    
    #### Evaluate on testsets
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
    metrics_Tr = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN) 

    # evaluate on the internal test set
    rf_res = modelRF.predict(X_iTs)
    en_res = modelEN.predict(X_iTs)
    res_iTs = pd.DataFrame({"True_HVPG" : Y_iTs, 
                            "RF_HVPG" : rf_res,
                            "EN_HVPG" : en_res})
    y_true = res_iTs["True_HVPG"]
    y_pred_RF = res_iTs["RF_HVPG"]
    y_pred_EN = res_iTs["EN_HVPG"]
    metrics_iTs = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN) 


    # evaluate on the external test set
    rf_res = modelRF.predict(X_eTs)
    en_res = modelEN.predict(X_eTs)
    res_eTs = pd.DataFrame({"True_HVPG" : Y_eTs, 
                            "RF_HVPG" : rf_res,
                            "EN_HVPG" : en_res})
    y_true = res_eTs["True_HVPG"]
    y_pred_RF = res_eTs["RF_HVPG"]
    y_pred_EN = res_eTs["EN_HVPG"]
    metrics_eTs = radipop_utils.inference.quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN) 


    # save results
    dst_Tr = OUTDIR / "regression" / RADIOMICS_OPTION / "raw_results_training_set.xlsx"
    res_Tr.to_excel(dst_Tr)
    print(f"Results saved to {dst_Tr}")
    
    dst_iTs = OUTDIR / "regression" / RADIOMICS_OPTION / "raw_results_internal_test_set.xlsx"
    res_iTs.to_excel(dst_iTs)
    print(f"Results saved to {dst_iTs}")

    dst_eTs = OUTDIR / "regression" / RADIOMICS_OPTION / "raw_results_external_test_set.xlsx"
    res_eTs.to_excel(dst_eTs)
    print(f"Results saved to {dst_eTs}")


    dst_metrics_Tr = OUTDIR / "regression" / RADIOMICS_OPTION / "metrics_training_set.xlsx"
    metrics_Tr.to_excel(dst_metrics_Tr)
    print(f"Metrics saved to {dst_metrics_Tr}")

    dst_metrics_iTs = OUTDIR / "regression" / RADIOMICS_OPTION / "metrics_internal_test_set.xlsx"
    metrics_iTs.to_excel(dst_metrics_iTs)
    print(f"Metrics saved to {dst_metrics_iTs}")

    dst_metrics_eTs = OUTDIR / "regression" / RADIOMICS_OPTION / "metrics_external_test_set.xlsx"
    metrics_eTs.to_excel(dst_metrics_eTs)
    print(f"Metrics saved to {dst_metrics_eTs}")
    
    
    
    # print some results to the console
    print("metrics_training:")
    print("="*20)
    pprint(metrics_Tr)
    print()
    
    print("metrics_training_CV5:")
    print("="*20)
    pprint(metrics_training_CV5)
    print()
    
    print("metrics_iTs:")
    print("="*20)
    pprint(metrics_iTs)
    print()

    print("metrics_eTs:")
    print("="*20)
    pprint(metrics_eTs)
    print()
    
    return None


if __name__=="__main__":
    main_function()
    