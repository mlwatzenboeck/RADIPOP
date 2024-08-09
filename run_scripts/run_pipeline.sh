#!/usr/bin/bash 

# images_and_mask_paths_file=/home/cwatzenboeck/data/cirdata/tabular_data/paths_and_hvpg_values/file_paths_and_hvpg_data_Dataset125_LSS_REGRESSION_only_real_data_no_autoseg.xlsx

health_status=ORD
images_and_mask_paths_file=/home/cwatzenboeck/data/cirdata/tabular_data/paths_and_hvpg_values/file_paths_and_hvpg_data_Dataset125_LSS_REGRESSION_${health_status}_nnUNet_seg.xlsx
radiomics_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/radiomics_nnUNet_spacing_222/
fe_settings=/home/cwatzenboeck/code/RADIPOP/yaml/radiomics_fe_setttings_CT_no_preprocessing_222.yaml
model_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/regression/radipop_nnUNet_spacing_222_${health_status}



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
                                        --search_scoring_metric r2  

radipop_evaluate_model \
    --model_dir $model_dir  \
    --data_iTs $radiomics_dir/df_${health_status}_iTs.csv \
    --data_eTs $radiomics_dir/df_${health_status}_eTs.csv \
    --data_Tr $radiomics_dir/df_${health_status}_Tr.csv  \


###############################################

health_status=WFD
images_and_mask_paths_file=/home/cwatzenboeck/data/cirdata/tabular_data/paths_and_hvpg_values/file_paths_and_hvpg_data_Dataset125_LSS_REGRESSION_${health_status}_nnUNet_seg.xlsx
radiomics_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/radiomics_nnUNet_spacing_222/
fe_settings=/home/cwatzenboeck/code/RADIPOP/yaml/radiomics_fe_setttings_CT_no_preprocessing_222.yaml
model_dir=/home/cwatzenboeck/data/cirdata/radiomics/Dataset125_LSS/regression/radipop_nnUNet_spacing_222_${health_status}



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
                                        --search_scoring_metric r2  

radipop_evaluate_model \
    --model_dir $model_dir  \
    --data_iTs $radiomics_dir/df_${health_status}_iTs.csv \
    --data_eTs $radiomics_dir/df_${health_status}_eTs.csv \
    --data_Tr $radiomics_dir/df_${health_status}_Tr.csv  \




