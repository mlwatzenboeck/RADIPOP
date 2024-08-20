# RADIPOP   CT Radiomics for Portal Hypertension

 

## Install instructions: 


Most of the code is in our python package `radipop_utils` which comes with an appropriate `pyproject.toml` file, which specifies the dependencies. 
This allows you to install it simply with 
```
pip install -e .
```

However, if you want to use automatic segmentation (currently only implemented with the `TotalSegmentor`) it is advised to install `pytorch` first on your system by following the official install instructions from their website. E.g. when you use `conda` as a package mananger you might want to use something like this

```bash
# make new env: 
conda create --name pyt3-10mc -c conda-forge python=3.10
conda activate pyt3-10mc

# Some parts require pytorch
# This must first be installed according to the offical website (matching your GPU, ...)
# https://pytorch.org/get-started/locally/

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# check python: torch.cuda.is_available()

# After that the other requirements should be handeled by the `pyproject.toml`
pip install -e . 
```

<!-- 
## Example `.env`

The path to the dataset in the provided sample notebooks read from a `.env` file. 
This will be different for your dataset. Please create a `.env` file with the following entires, or change the notebooks accordingly
```bash
local_user=cwatzenboeck
DATA_ROOT_DIRECTORY=/home/${local_user}/data/cirdata
```
or store them as environment variables.


```python
# OPTION 1:  
# Load environment variables from .env file if it exists 
from dotenv import load_dotenv
load_dotenv()

# OPTION 2:
from dotenv import dotenv_values

config = dotenv_values(".env"),  # load environment variables as dictionary
``` -->

_________________


## How to use on a single CT-scan: 
Given a CT scan in the form of a DICOM folder or a NII.GZ file we provide a script which can be used to: 
- create the segmentation masks (e.g. for liver and spleen)
- extract the corresponding radiomics features
- predict the (radio-)HVPG using an already trained ML model

All this can be done by simply running: 

```bash
radipop_segment_and_predict --image_path IMAGE_PATH --output_folder OUTPUT_FOLDER
```


__________________

## How to retrain use on your own data: 

Let us assume that you have your *own* set CT scans and segmentation masks for some organs, as well some scalar value (e.g. HVPG value measured in close proximity to the CT-scan data) and that you want to 
- extract the radiomic from the CT scans/masks
- train some ML models like RandomForrest or ElasticNetRegression for predicting the scalar value (e.g. HVPG) and also utilize recursive feature elimination, as we did in our manuscript. 

We will give a brief description of the suggested workflow to achive this. After installing the `radipop_utils` utils package the following commands are available: 


-  `radipop_suggest_binwidth`-- Estimate a reasonable BinWidth for the radiomics settings of your data
<!-- - `radipop_extract_features`  -- Can be used to extract the radiomics feature from single CT images/ masks.  -->
- `radipop_extract_features_many_times` -- Can be used to extract the radiomics feature from  CT images.
- `radipop_combine_radiomics_with_scalar_target_and_split`
- `radipop_training_and_hyperparams_search` -- Get reasonable hyperparameters by running Bayesian optimization (on the training set!)
- `radipop_evaluate_model` -- Run inference on internal testset (`iTs`) and external testset (`eTs`) and save metrics as well as raw inference.

In essence, you just need to run these command line functions in order, if you want to retrain the model with your own data. 



### Preliminaries: 

The file with the path-names for the images and masks for `radipop_extract_features_many_times --images_and_mask_paths_file <this file>` as well as the training should look like this: 


<img src="../fig/file_paths_and_hvpg_values.png" style="width: 55vw; min-width: 330px;">

Note that this assumes that the training test-split, was well as the rotation (non-overlapping!) cross-validation sets were already defined beforehand. It might for instance make sense to perform a stratified-training test split, with a stratification on scanner-type, sex, health status (as we did) or something similar. 

*Yes, I know that there is an error in paths of the screenshot. This is just for illustration purposes. You need the correct paths for your images/masks anyhow.*

### Workflow


```bash
#!/usr/bin/bash 

health_status=ORD
images_and_mask_paths_file=/home/cwatzenboeck/data/cirdata/tabular_data/paths_and_hvpg_values/file_paths_and_hvpg_data_Dataset125_LSS_REGRESSION_${health_status}_no_autoseg.xlsx
radiomics_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/radiomics_no_autoseg_spacing_111/
fe_settings=/home/cwatzenboeck/code/RADIPOP/yaml/radiomics_fe_setttings_CT_no_preprocessing_spacing_111.yaml
model_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/regression/radipop_no_autoseg_spacing_111_${health_status}



# optional: 
#   # get a reasonable binwidth and update the fe_settings_file accordingly
#   radipop_suggest_binwidth \
#     --images_and_mask_paths_file  $images_and_mask_paths_file \
#     --frac 0.1  \
#     --fe_settings $fe_settings


# mandatory:
radipop_extract_features_many_times  \
   --images_and_mask_paths_file  $images_and_mask_paths_file  \
   --output_dir $radiomics_dir \
   --fe_settings $fe_settings


radipop_combine_radiomics_with_scalar_target_and_split  \
    --images_and_mask_paths_file  $images_and_mask_paths_file  \
    --radiomics_dir $radiomics_dir  \
    --output_prefix  df_$health_status


radipop_training_and_hyperparams_search --data_Tr $radiomics_dir/df_${health_status}_Tr.csv  \
                                        --outdir $model_dir  \
                                        --num_searches 100 \
                                        --search_scoring_metric neg_root_mean_squared_error  

radipop_evaluate_model \
    --model_dir $model_dir  \
    --data_iTs $radiomics_dir/df_${health_status}_iTs.csv \
    --data_eTs $radiomics_dir/df_${health_status}_eTs.csv \
    --data_Tr $radiomics_dir/df_${health_status}_Tr.csv  \


```

_________________

## TODOS
Just some notes so that I don't forget. 

### Ongoing
- [ ] Freeze environment and add as [requirements.txt](requirements.txt)
- [ ] Add a different segmentation model (trained on many patientes with severe and diverse liver problems)

### Completed Column ✓
- [x] Refactor and change to package
- [x] Add `.yaml` files for the `pyradiomics` settigs and describe in more detail how to set `binWidth` ect. 
- [x] Create pipeline
  - [x] Feature extraction as a tool. 
  - [x] Hyperparameter search as a tool. 
  - [x] Evaluation as a tool. 
  - [x] Prediction as a tool (maybe publish trained model to huggingface, or similar)

