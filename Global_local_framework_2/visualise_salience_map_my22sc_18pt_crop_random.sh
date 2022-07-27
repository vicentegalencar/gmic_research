#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model_rnd22.py \
--model_path="checkpoints/cbis_rn22sc_rn18pt_ep150_crop_random_v2/last_model.pth" \
--data_path="/media/hdd/filipe/datasets/preprocessed_gmic_random/" \
--mask_dir="checkpoints/cbis_rn22sc_rn18pt_ep150_crop_random_v2/segmentation/masks" \
--output_path="checkpoints/cbis_rn22sc_rn18pt_ep150_crop_random_v2/segmentation" \
--bs=4 \
--pretrained_model=True \
--percent_t=0.02 \
--gpuid 0 \
--num_chan=1