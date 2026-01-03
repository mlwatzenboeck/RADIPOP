# python ~/code/RADIPOP/radipop_utils/preprocessing/dcm_seperation.py --folder_name ~/data/RADIPOP_EXTRA/working_env/FINAL.11/

set -e 
set -x

base="~/data/RADIPOP_EXTRA/working_env/"
ID="11"
dcm_folder_all="${base}/FINAL.${ID}"

#python ~/code/RADIPOP/radipop_utils/preprocessing/mask_and_png_seperation.py --folder_name ~/data/RADIPOP_EXTRA/working_env/FINAL.11/

series_name="090300.497405"
dcm_folder="${base}/FINAL.${ID}/by_acqtime/${series_name}"

## Convert dicoms in folder to NII
#python /home/clemens/code/RADIPOP/radipop_utils/tools/my_dcm2nii.py  --dicom_folder "${dcm_folder}"   --out_id "V${ID}"

python /home/clemens/code/RADIPOP/radipop_utils/preprocessing/combine_pickeled_masks.py  \
  --folder_name "${dcm_folder_all}"   --series "${series_name}"  


#python ~/code/RADIPOP/radipop_utils/preprocessing/mask_and_png_seperation.py --folder_name ~/data/RADIPOP_EXTRA/working_env/FINAL.38/
