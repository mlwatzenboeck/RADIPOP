import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
import skopt

import radipop_utils
import radipop_utils.visualization
import radipop_utils.features
from radipop_utils.features import SpearmanReducerCont
import radipop_utils.utils
import radipop_utils.data

import datetime


def main_function():
    parser = argparse.ArgumentParser(
        description='Training and Hyperparameter Search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_Tr', type=Path,
                        help='Path to the CSV file containing the training data.')
    
    parser.add_argument('--outdir', type=Path, 
                        help='Output directory for model, hyper_params serach results, normalization_df, ... .')
    
    parser.add_argument('--num_searches', type=int, default=10,
                        help='Number of hyperparameter searches (default: %(default)s)')
    
    parser.add_argument('--search_scoring_metric', type=str, default="r2",
                        help='Scoring metric for hyperparameter search. (e.g. "r2", "neg_root_mean_squared_error", ... ). (default: %(default)s)')
    
    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    print()
    
    # make out dir if not exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # save settings
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst = args.outdir / f"args_settings__training_and_hyperparams_search_{ts}.yaml"    
    radipop_utils.utils.save_args_settings(args_dict, dst)

    training_data_csv_file = Path(args.data_Tr)
    outdir = Path(args.outdir)
    NUM_SEARCHES = args.num_searches
    SEARCH_SCORING_METRIC = args.search_scoring_metric
    
    organs = ["liver", "spleen"]
    
    # load data, filter to relevant features only get split indices
    df_Tr = pd.read_csv(training_data_csv_file)
    print(f"Loaded data from {training_data_csv_file} with {len(df_Tr)=}") 
    split_indices_CV5_Tr = radipop_utils.data.extract_CV_indices(df_Tr)
    re_pattern = "^(" + "|".join(organs) + ")"
    df_Tr_X, df_Tr_y = df_Tr.filter(regex=re_pattern), df_Tr["y"]

    # normalize the data and save normalization df
    normalization_df = radipop_utils.data.make_and_save_normalization_df(df_Tr_X, outdir, verbose=True)
    scaler  = radipop_utils.data.make_scaler_from_normalization_df(normalization_df)
    X_Tr = scaler.transform(df_Tr_X)
    Y_Tr = df_Tr_y.values

    # decide on a rought range for the cut parameters for dendrogram
    split_params = [0.5, 0.75, 1, 2.75,  5, 7.5, 10]
    for split_param in split_params:
        selector = SpearmanReducerCont(split_param=split_param)
        print(f"Selected features at height {split_param}:", len(
            selector.fit(X_Tr, Y_Tr).selected_features))

    # #### Fit on `Tr` data with CV and estimate best model + hyper parameters
    # Bounds for hyperparameters
    param_bounds_rf = {
        'feature_selection__split_param': skopt.space.Real(1, 5, prior="uniform"),
        'regression': [RandomForestRegressor(random_state=2023)],
        'regression__n_estimators': skopt.space.Integer(100, 2000),
        'regression__max_depth': skopt.space.Integer(1, 50),
        'regression__min_samples_split': skopt.space.Integer(2, 25)  # ,
    }

    param_bounds_en = {
        'feature_selection__split_param': skopt.space.Real(1, 5, prior="uniform"),
        'regression': [ElasticNet(random_state=2023)],
        'regression__alpha': skopt.space.Real(0.0001, 1.0, 'uniform'),
        'regression__l1_ratio': skopt.space.Real(0, 1.0, 'uniform')
    }

    print("Bounds for Random Forest hyperparameters:")
    pprint(param_bounds_rf)
    print()
    print("Bounds for ElasticNet hyperparameters:")
    pprint(param_bounds_en)
    print()

    # create a pipeline
    reg_RF = Pipeline([
        # ('scaler', StandardScaler()),
        ('feature_selection', SpearmanReducerCont()),
        ('regression', RandomForestRegressor())
    ])

    # try out models
    opt0 = skopt.BayesSearchCV(
        reg_RF,
        [(param_bounds_en, NUM_SEARCHES), (param_bounds_rf, NUM_SEARCHES)],
        cv=split_indices_CV5_Tr,
        scoring=SEARCH_SCORING_METRIC,
        verbose=True,
        random_state=2023,
        n_jobs=6
    )
    opt0.fit(X_Tr, Y_Tr)
    cv_res = pd.DataFrame(opt0.cv_results_)

    # save results

    
    dst = outdir / f"Bayesian_results_{NUM_SEARCHES}_iterations_RFvsEN.xlsx"
    cv_res.to_excel(dst)
    print("Saved hyperparams search to : ", dst)

    # #### save model trained on the whole training data set and optimal paramters
    idx_best_EN_model = cv_res["mean_test_score"][:NUM_SEARCHES].argmax()
    idx_best_RF_model = cv_res["mean_test_score"][NUM_SEARCHES:].argmax() + NUM_SEARCHES

    # save optimal parameters as yaml:
    dst = outdir / "SpearmanRed1_RF_opt_params.yml"
    data = {**cv_res.iloc[idx_best_RF_model, :].params}
    if "regression" in data:
        data.pop("regression")
    with open(dst, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        print("saved params to ", dst)

    # save optimal parameters as yaml:
    dst = outdir / "SpearmanRed1_EN_opt_params.yml"
    # make copy to keep original dict unchanged
    data = {**cv_res.iloc[idx_best_EN_model, :].params}
    if "regression" in data:
        data.pop("regression")
    with open(dst, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        print("saved params to ", dst)

    # ---- set best performing en/rf models
    # create a pipeline
    reg_RF = Pipeline([
        # ('scaler', StandardScaler()),
        ('feature_selection', SpearmanReducerCont()),
        ('regression', RandomForestRegressor())
    ])
    # Set params
    np.random.seed(2023)
    reg_RF.set_params(**cv_res.iloc[idx_best_RF_model, :].params)
    reg_RF.fit(X_Tr, Y_Tr)
    dst = outdir / f"SpearmanRed1_RF_opt.p"
    with open(dst, "wb") as fp:
        pickle.dump(reg_RF, fp)
        print("Saved model to ", dst)

    # create a pipeline
    reg_EN = Pipeline([
        # ('scaler', StandardScaler()),
        ('feature_selection', SpearmanReducerCont()),
        ('regression', ElasticNet())
    ])
    reg_EN.set_params(**cv_res.iloc[idx_best_EN_model, :].params)
    reg_EN.fit(X_Tr, Y_Tr)
    dst = outdir / f"SpearmanRed1_EN_opt.p"
    with open(dst, "wb") as fp:
        pickle.dump(reg_EN, fp)
        print("Saved model to ", dst)


if __name__ == "__main__":
    main_function()
    print("Done!")
