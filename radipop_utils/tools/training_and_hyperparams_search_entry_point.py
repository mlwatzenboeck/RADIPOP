import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import pickle
from pprint import pprint
import argparse

from sklearn.ensemble import RandomForestRegressor
# see https://stackoverflow.com/questions/60321389/sklearn-importerror-cannot-import-name-plot-roc-curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
import skopt

import radipop_utils
import radipop_utils.visualization
import radipop_utils.features
from radipop_utils.features import SpearmanReducerCont
import radipop_utils.utils
import radipop_utils.data


# load user/ system specific env variables:
from dotenv import dotenv_values, find_dotenv
# load environment variables as dictionary
config = dotenv_values(find_dotenv())
DATA_ROOT_DIRECTORY_DEFAULT = Path(config["DATA_ROOT_DIRECTORY"])

path = Path(os.path.abspath(radipop_utils.__file__))
RADIPOP_PACKAGE_ROOT = path.parent.parent


def main_function():

    parser = argparse.ArgumentParser(
        description='Training and Hyperparameter Search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_root_directory', type=str, default=str(DATA_ROOT_DIRECTORY_DEFAULT),
                        help='Root directory of the data. (default: %(default)s)')
    parser.add_argument('--outdir', type=str, default=str(DATA_ROOT_DIRECTORY_DEFAULT / "radiomics" / "Dataset125_LSS"),
                        help='Output directory. (default: %(default)s)')
    parser.add_argument('--dataset', type=str, default="Dataset125_LSS",
                        help='Dataset name. (default: %(default)s)')
    parser.add_argument('--radiomics_option', type=str, default="radipop_111",
                        help='Radiomics option. (default: %(default)s)')
    parser.add_argument('--num_searches', type=int, default=10,
                        help='Number of hyperparameter searches. (default: %(default)s)')
    parser.add_argument('--search_scoring_metric', type=str, default="r2",
                        help='Scoring metric for hyperparameter search. (default: %(default)s)')

    args = parser.parse_args()
    args_dict = vars(args)
    print("Used arguments: ")
    print("---------------")
    pprint(args_dict)
    print()

    DATA_ROOT_DIRECTORY = Path(args.data_root_directory)
    OUTDIR = Path(args.outdir)
    DATASET = args.dataset
    RADIOMICS_OPTION = args.radiomics_option
    NUM_SEARCHES = args.num_searches
    SEARCH_SCORING_METRIC = args.search_scoring_metric

    # Load the data
    df_Tr, df_iTs, df_eTs = radipop_utils.data.load_HVPG_values_and_radiomics(
        DATASET=DATASET, RADIOMICS_OPTION=RADIOMICS_OPTION, DATA_ROOT_DIRECTORY=DATA_ROOT_DIRECTORY)
    print(f"{len(df_Tr)=}, {len(df_eTs)=}, {len(df_iTs)=}")

    split_indices_CV5_Tr = radipop_utils.data.extract_CV_indices(df_Tr)

    X_Tr, Y_Tr, X_iTs, Y_iTs, X_eTs, Y_eTs = radipop_utils.data.preprocess_data(
        df_Tr, df_iTs, df_eTs, normalize_X=True)

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
    os.makedirs(OUTDIR / "regression" / RADIOMICS_OPTION, exist_ok=True)
    dst = OUTDIR / "regression" / RADIOMICS_OPTION / \
        f"Bayesian_results_{NUM_SEARCHES}_iterations_RFvsEN.xlsx"
    cv_res.to_excel(dst)
    print("Saved hyperparams search to : ", dst)

    # #### save model trained on the whole training data set and optimal paramters
    idx_best_EN_model = cv_res["mean_test_score"][:NUM_SEARCHES].argmax()
    idx_best_RF_model = cv_res["mean_test_score"][NUM_SEARCHES:].argmax(
    ) + NUM_SEARCHES

    # save optimal parameters as yaml:
    save_dst = OUTDIR / "regression" / RADIOMICS_OPTION
    dst = save_dst / "SpearmanRed1_RF_opt_params.yml"
    data = {**cv_res.iloc[idx_best_RF_model, :].params}
    if "regression" in data:
        data.pop("regression")
    with open(dst, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
        print("saved params to ", dst)

    # save optimal parameters as yaml:
    dst = save_dst / "SpearmanRed1_EN_opt_params.yml"
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
    dst = save_dst / f"SpearmanRed1_RF_opt.p"
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
    dst = save_dst / f"SpearmanRed1_EN_opt.p"
    with open(dst, "wb") as fp:
        pickle.dump(reg_EN, fp)
        print("Saved model to ", dst)


if __name__ == "__main__":
    main_function()
    print("Done!")
