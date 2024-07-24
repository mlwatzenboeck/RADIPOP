# RADIPOP  -- CT Radiomics for Portal Hypertension

This repository accompanies the manuscript 

> Radiomics-based assessment of portal hypertension severity and risk stratification of cirrhotic patients using routine CT scans 
>
> Celine Sin1,2,3, Martin L. Watzenboeck4,5, Eugenia Iofinova1,6, Lorenz Balcar5,7,8, Georg Semmler5,7, Bernhard Scheiner5,7, Katharina Lampichler4,5,7, Mattias Mandorfer5,7, Lucile Moga9,10, Pierre-Emmanuel Rautou9,10, Maxime Ronot10,11, Jörg Menche1,2,3, Thomas Reiberger1,5,7,8‡, Martina Scharitzer1,4,5 ‡
> 
>  CS and MLW share first authorship position
>‡ TR and MS share last authorship position 

>1 CeMM Research Center for Molecular Medicine of the Austrian Academy of Sciences, Vienna, Austria
>2 Max Perutz Labs, Vienna Biocenter Campus (VBC), Vienna, Austria
>3 University of Vienna, Center for Molecular Biology, Department of Structural and Computational Biology, Vienna, Austria
>4 Department of Biomedical Imaging and Image-Guided Therapy, Medical University of Vienna, Vienna, Austria
>5 Clinical Research Group MOTION, Medical University of Vienna, Vienna, Austria
>6 Institute of Science and Technology Austria, Vienna, Austria 
>7 Vienna Hepatic Hemodynamic Laboratory, Division of Gastroenterology and Hepatology, Department of Medicine III, Medical University of Vienna, Vienna, Austria
>8 Christian Doppler Laboratory for Portal Hypertension and Liver Fibrosis, Medical University of Vienna, Vienna, Austria
>9 AP-HP, Hôpital Beaujon, Service d'Hépatologie, DMU DIGEST, Centre de Référence des Maladies Vasculaires du Foie, FILFOIE, ERN RARE-LIVER, Clichy, France
>10 Université Paris-Cité, Inserm, Centre de recherche sur l'inflammation, UMR 1149, Paris, France
>11  Service de radiologie, Hôpital Beaujon, APHP Nord, Clichy & Université Paris Cité, CRI UMR 1149, Paris, France


It contains scripts and tools used in the study for radiomics analysis, including image processing, feature extraction, feature selection, and model training.

For additional scripts related to this study can be found [https://github.com/menchelab/radipop_scripts](https://github.com/menchelab/radipop_scripts). 

Our custom segmentation software can be provided on request. For more information, visit Radipop Scripts and RADIPOP.


# RADIPOP -- CT Radiomics for Portal Hypertension

This repository supports the manuscript:

**Radiomics-based assessment of portal hypertension severity and risk stratification of cirrhotic patients using routine CT scans**

**Authors:**
Celine Sin¹²³, Martin L. Watzenboeck⁴⁵, Eugenia Iofinova¹⁶, Lorenz Balcar⁵⁷⁸, Georg Semmler⁵⁷, Bernhard Scheiner⁵⁷, Katharina Lampichler⁴⁵⁷, Mattias Mandorfer⁵⁷, Lucile Moga⁹¹⁰, Pierre-Emmanuel Rautou⁹¹⁰, Maxime Ronot¹⁰¹¹, Jörg Menche¹²³, Thomas Reiberger¹⁵⁷⁸, Martina Scharitzer¹⁴⁵

¹ CeMM Research Center for Molecular Medicine, Vienna, Austria  
² Max Perutz Labs, Vienna Biocenter Campus, Vienna, Austria  
³ University of Vienna, Center for Molecular Biology, Vienna, Austria  
⁴ Department of Biomedical Imaging and Image-Guided Therapy, Medical University of Vienna, Vienna, Austria  
⁵ Clinical Research Group MOTION, Medical University of Vienna, Vienna, Austria  
⁶ Institute of Science and Technology Austria, Vienna, Austria  
⁷ Vienna Hepatic Hemodynamic Laboratory, Medical University of Vienna, Vienna, Austria  
⁸ Christian Doppler Laboratory for Portal Hypertension and Liver Fibrosis, Medical University of Vienna, Vienna, Austria  
⁹ AP-HP, Hôpital Beaujon, Clichy, France  
¹⁰ Université Paris-Cité, Inserm, Paris, France  
¹¹ Service de radiologie, Hôpital Beaujon, Paris, France

This repository contains scripts and tools for radiomics analysis used in the study, including image processing, feature extraction, feature selection, and model training.

Additional related scripts are available at [Radipop Scripts](https://github.com/menchelab/radipop_scripts).

Our custom segmentation software can be provided upon request. For more information, visit Radipop Scripts and RADIPOP.

## Install instructions: 


The python packages come with an appropriate `pyproject.toml` file, which specifies the dependencies. 
This allows you to install them simply with

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

## TODOS
Just some notes so that I don't forget. 

### Ongoing
- [ ] Add `.yaml` files for the `pyradiomics` settigs and describe in more detail
  - [x] Added [exampleCT.yaml](yaml/exampleCT.yaml), but needs to be checked
- [ ] Freeze environment and add as [requirements.txt](requirements.txt)
- [ ] Create pipeline
  - [x] Feature extraction as a tool:
  - [ ] Prediction as a tool 

### Completed Column ✓
- [x] Refactor and change to package


_________________

## Usage instructions: 

After installing the `radipop_utils` utils package the following commands are available: 

- `radipop_extract_features`  -- Can be used to extract the radiomics feature from single CT images/ masks. 
- `radipop_extract_features_many_times` -- Can be used to extract the radiomics feature from  CT images.
- `radipop_predict` -- TODO!!



```bash
$ radipop_extract_features --help 

usage: radipop_extract_features [-h] [--patientid PATIENTID] [--image IMAGE] [--masks MASKS] [--output_dir OUTPUT_DIR] [--fe_settings FE_SETTINGS]

Extract and save radiomics features from NIfTI images.

optional arguments:
  -h, --help            show this help message and exit
  --patientid PATIENTID
                        Name of the newly created output folder
  --image IMAGE         Path to image in .nii format
  --masks MASKS         Path to masks in .nii format; in mask 1=liver; 2=spleen; 0=other
  --output_dir OUTPUT_DIR
                        Directory where the extracted features will be saved.
  --fe_settings FE_SETTINGS
                        Path to the radiomics feature extraction settings file.
```



```bash
$ radipop_extract_features_many_times --help 

usage: radipop_extract_features_many_times [-h] [--images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE] [--output_dir OUTPUT_DIR] [--fe_settings FE_SETTINGS]

Extract and save radiomics features from NIfTI images (paths provided as an xlsx file.)

optional arguments:
  -h, --help            show this help message and exit
  --images_and_mask_paths_file IMAGES_AND_MASK_PATHS_FILE
                        Path to the Excel file containing image and mask paths and patient IDs.
  --output_dir OUTPUT_DIR
                        Directory where the extracted features will be saved.
  --fe_settings FE_SETTINGS
                        Path to the radiomics feature extraction settings file.
```


Example:
--------

```bash
radipop_extract_features --patientid patient0001 \
 --image /home/cwatzenboeck/data/cirdata/nnUNet_raw/Dataset125_LSS/imagesAll/patient0001_0000.nii.gz \
 --masks /home/cwatzenboeck/data/cirdata/nnUNet_raw/Dataset125_LSS/labelsAll/patient0001.nii.gz \
 --output_dir /home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/radipop 
```


```bash
radipop_extract_features_many_times  \
 --images_and_mask_paths_file /home/cwatzenboeck/code/RADIPOP/data/file_paths_and_hvpg_data.xlsx \
 --output_dir /home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/radipop \
 --fe_settings /home/cwatzenboeck/code/RADIPOP/yaml/exampleCT.yaml 
```
