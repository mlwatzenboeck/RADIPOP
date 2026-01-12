set -x 
set -e 



#   radipop_pipeline_dcm_separation_dcm2nii_totalseg \
#     --base_folder "/home/clemens/data/RADIPOP_EXTRA/FINAL SEGM DATEN/FINAL.456"   \
#     --roi_subset liver spleen \
#     --device cpu \
#     --skip_separation \
#     --skip_dcm2nii


#   radipop_pipeline_dcm_separation_dcm2nii_totalseg \
#     --base_folder ~/data/RADIPOP_EXTRA/working_env/FINAL.38/ \
#     --roi_subset liver spleen \
#     --device cpu



radipop_combine_masks_and_extract_features \
     --base_folder "/home/clemens/data/RADIPOP_EXTRA/FINAL SEGM DATEN/FINAL.456"   \
  --fe_settings /home/clemens/data/RADIOMICS_LOCAL/radiomics_no_autoseg_spacing_111/radiomics_fe_setttings_CT_no_preprocessing_spacing_111.yaml 


#   python /home/clemens/code/RADIPOP/radipop_utils/tools/full_pipeline_entry_point.py \
#     --base_folder "/home/clemens/data/RADIPOP_EXTRA/FINAL SEGM DATEN/" \
#     --fe_settings /home/clemens/data/RADIOMICS_LOCAL/radiomics_no_autoseg_spacing_111/radiomics_fe_setttings_CT_no_preprocessing_spacing_111.yaml



