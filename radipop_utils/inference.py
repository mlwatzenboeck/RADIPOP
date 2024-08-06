from pathlib import Path
import yaml
import pickle
from typing import Union, Tuple, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, r2_score, mean_squared_error, mean_absolute_error)
from scipy.stats import pearsonr
from radipop_utils.features import SpearmanReducerCont


import radipop_utils.features
import radipop_utils.visualization
import radipop_utils.inference
import radipop_utils.data



def load_models_and_params(model_dir: Union[str, Path]) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
    """
    Load the models (RF, EN) and the hyperparameters from the specified directory.
    
    Returns:
    - loaded_models: dict of loaded models
    - loaded_params: dict of loaded hyperparameters
    - models_bare: dict of bare models (without hyperparameters set)
    """  
  
    model_dir = Path(model_dir)
    # RF model
    reg_RF = Pipeline([
        ('feature_selection', SpearmanReducerCont()),
        ('regression', RandomForestRegressor())
    ])

    # EN model
    reg_EN = Pipeline([
        ('feature_selection', SpearmanReducerCont()),
        ('regression', ElasticNet())
    ])

    models_bare = dict(RF=reg_RF, EN=reg_EN)

    loaded_params = dict()
    loaded_models = dict()
    for model in models_bare.keys():
        # Load hyperparameters
        filename = model_dir / f"SpearmanRed1_{model}_opt_params.yml"
        with open(filename) as f:
            loaded_hyperparams = yaml.safe_load(f)
        loaded_params[model] = loaded_hyperparams

        # Load the model
        filename = model_dir / f"SpearmanRed1_{model}_opt.p"
        loaded_model = pickle.load(open(filename, 'rb'))
        loaded_models[model] = loaded_model

    return loaded_models, loaded_params, models_bare


def fit_and_prediction_CV5_training(modelRF, modelEN, X_Tr, Y_Tr, split_indices_CV5_Tr) -> pd.DataFrame:
    #run 5 fold cv
    rf_train_res = np.array([])
    en_train_res = np.array([])
    obs = np.array([])

    for train, test in split_indices_CV5_Tr:
        modelRF.fit(X_Tr[train], Y_Tr[train])
        rf_train_res = np.append(rf_train_res, modelRF.predict(X_Tr[test]))
        
        modelEN.fit(X_Tr[train], Y_Tr[train])
        en_train_res = np.append(en_train_res, modelEN.predict(X_Tr[test]))
        
        obs = np.append(obs, Y_Tr[test])
        

    res_training = pd.DataFrame({"True_HVPG" : obs, 
                                "RF_HVPG" : rf_train_res,
                                "EN_HVPG" : en_train_res})
    return res_training


def quantitation_metrics_RF_and_EN(y_true, y_pred_RF, y_pred_EN):
    y_true_cat = np.array([0 if x < 10 else 1 for x in y_true])

    y_pred = y_pred_RF
    results = dict(r2_score = r2_score(y_true, y_pred), 
                mean_absolute_error = mean_absolute_error(y_true, y_pred),
                roc_auc_score = roc_auc_score(y_true_cat, y_pred),
                pearsonr = pearsonr(y_true, y_pred).correlation,
    )
    df_RF = pd.DataFrame(results, index=["RF"])


    y_pred = y_pred_EN
    results = dict(r2_score = r2_score(y_true, y_pred), 
                mean_absolute_error = mean_absolute_error(y_true, y_pred),
                roc_auc_score = roc_auc_score(y_true_cat, y_pred),
                pearsonr = pearsonr(y_true, y_pred).correlation,
    )
    df_EN = pd.DataFrame(results, index=["EN"])

    return pd.concat([df_RF, df_EN])


def get_feature_importancesRF(df_Tr, X_Tr, Y_Tr, modelRF, loaded_params):
    raise NotImplementedError("This function is not implemented yet.")
    # dim(features[selector.selected_features])  != dim(modelRF.named_steps["regression"].feature_importances_)
    # use permuation based feature importance instead  ... but is not super important for now
    selector = SpearmanReducerCont(loaded_params["RF"]['feature_selection__split_param'])
    selector.fit(X_Tr, Y_Tr)

    features = np.array(df_Tr.filter(regex="^liver|^spleen").columns)
    feat_imp = pd.DataFrame({
        "feature": features[selector.selected_features],
        "importance": modelRF.named_steps["regression"].feature_importances_
    })
    
    return feat_imp.sort_values("importance", ascending=False)


def process_metric_files(experiments, metric_files):
    metric_dataframes = {}
    for exp_name, exp in experiments.items():
        for metric_file in metric_files:
            path = exp["path"] / metric_file
            if path.exists():
                df = pd.read_excel(path)
                df.rename(columns={"Unnamed: 0": "model"}, inplace=True)  # Rename the unnamed column to 'model'
                df["experiment"] = exp_name  # Add a new column with the exp_name
                metric_dataframes.setdefault(metric_file, []).append(df)
            else:
                print(f"File {path} does not exist")
        print("\n\n")

    # Access the dataframes for each metric type
    r = {}
    for metric_file, dfs in metric_dataframes.items():
        df_concat = pd.concat(dfs).reset_index(drop=True)
        df_concat = df_concat[['experiment'] + [col for col in df_concat.columns if col != 'experiment']]
        r[metric_file.split(".")[0]] = df_concat
    
    return r