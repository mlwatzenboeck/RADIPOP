# RADIPOP   CT Radiomics for Portal Hypertension

 

## Install instructions: 


Most of the code is in our python package `radipop_utils` which comes with an appropriate `pyproject.toml` file, which specifies the dependencies. 
This allows you to install it simply with 
```
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

## Usage instructions: 

After installing the `radipop_utils` utils package the following commands are available: 

-  `radipop_suggest_binwidth`-- Estimate a reasonable BinWidth for the radiomics settings of your data
- `radipop_extract_features`  -- Can be used to extract the radiomics feature from single CT images/ masks. 
- `radipop_extract_features_many_times` -- Can be used to extract the radiomics feature from  CT images.
- `radipop_predict` -- TODO!!

__________________

## How to use on your own data: 

Let us assume that you have your *own* set CT scans and segmentation masks for some organs, as well some scalar value (e.g. HVPG value measured in close proximity to the CT-scan data) and that you want to 
- extract the radiomic from the CT scans/masks
- train some ML models like RandomForrest or ElasticNetRegression for predicting the scalar value (e.g. HVPG) and also utilize recursive feature elimination, as we did in our manuscript. 

We will give a brief description of the suggested workflow to achive this. 



## Preliminaries: 



The file with the path-names for the images and masks for `radipop_extract_features_many_times --images_and_mask_paths_file <this file>` as well as the training should look like this: 


<img src="../fig/file_paths_and_hvpg_values.png" style="width: 55vw; min-width: 330px;">

Note that this assumes that the training test-split, was well as the rotation (non-overlapping!) cross-validation sets were already defined beforehand. It might for instance make sense to perform a stratified-training test split, with a stratification on scanner-type, sex, health status (as we did) or something similar.



### TODO Other prelimiary steps 


## Workflow

### 1) Adjust the settings for radiomics extraction
First you need to adjust the settings, in particular the `binWidth` of the [`exampleCT.yaml`](../yaml/exampleCT.yaml) to your data. 
It is suggested (in the radiomics package) that the number of bins should be 16-128. This means that you need to adjust the ratio of the value range (of the ROI) to the binWidth should be 16-128. We preform a conveneint script for estimating the binWidth: `radipop_suggest_binwidth`

```bash
$ radipop_suggest_binwidth -h

usage: radipop_suggest_binwidth [-h] [--images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE]
                                [--fe_settings FE_SETTINGS] [--frac FRAC]
                                [--window_location_middle WINDOW_LOCATION_MIDDLE] [--window_width WINDOW_WIDTH]

Read the radiomics file you intend to use and suggest an estimate for the binwidth

optional arguments:
  -h, --help            show this help message and exit
  --images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE
                        Path to the Excel file containing image and mask paths and patient IDs.
  --fe_settings FE_SETTINGS
                        Path to the radiomics feature extraction settings file.
  --frac FRAC           Use only a random fraction of the images for estimating the binwidth. 
                        (Speeds up the estimation, at the cost of accuracy.)
  --window_location_middle WINDOW_LOCATION_MIDDLE
                        Position (middpoint) of the intesity window. (Default = 50 HU -> soft tissue CT window.)
  --window_width WINDOW_WIDTH
                        Width of the intesity window. (Default = 500 HU -> soft tissue CT window.)
```
This scripts performs windowing to a certain intensity window, ensures that the output range between `[0, 1]` and suggest binWidth to have 16-128 bins, for the radiomics. Note that the other settings (in particular the resampling_size) is taken from the file `fe_settings` file. 
We suggest to use the *median spacing* of your dataset for resizing. This way the data is changed as little as possible.

After adjusting the fe_settings file (see [`exampleCT.yaml`](../yaml/exampleCT.yaml)) accordingly you can proceed to extract the radiomics from your dataset. 


### 2) radiomics extraction

For extracting the radiomics of your whole dataset we suggest you run the following command: 


```bash
$ radipop_extract_features_many_times -h
usage: radipop_extract_features_many_times [-h] [--images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE]
                                           [--output_dir OUTPUT_DIR] [--fe_settings FE_SETTINGS]
                                           [--window_location_middle WINDOW_LOCATION_MIDDLE]
                                           [--window_width WINDOW_WIDTH]

Extract and save radiomics features from NIfTI images (paths provided as an xlsx file.)

optional arguments:
  -h, --help            show this help message and exit
  --images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE
                        Path to the Excel file containing image and mask paths and patient IDs.
  --output_dir OUTPUT_DIR
                        Directory where the extracted features will be saved.
  --fe_settings FE_SETTINGS
                        Path to the radiomics feature extraction settings file.
  --window_location_middle WINDOW_LOCATION_MIDDLE
                        Position (middpoint) of the intesity window. (Default = 50 HU -> soft tissue CT window.)
  --window_width WINDOW_WIDTH
                        Width of the intesity window. (Default = 500 HU -> soft tissue CT window.)

```

If you only want to extract the radiomics for a single patient you might want to take a look at `radipop_extract_features -h` instead. 


### 3) Feature reduction, Training, Validation and Testing

TODO Update and clean up. Basic version is in  [`CW_model_training.ipynb`](../notebooks/CW_model_training.ipynb)).


_________________

## TODOS
Just some notes so that I don't forget. 

### Ongoing
- [ ] Freeze environment and add as [requirements.txt](requirements.txt)
- [ ] Create pipeline
  - [x] Feature extraction as a tool. 
  - [ ] Prediction as a tool (maybe publish trained model to huggingface, or similar)

### Completed Column ✓
- [x] Refactor and change to package
- [x] Add `.yaml` files for the `pyradiomics` settigs and describe in more detail

