#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model_rnd.py \
--model_path="checkpoints/model_1-cbis_rn18_pretrain_ep150_crop_random/last_model.pth" \
--data_path="/media/hdd/filipe/datasets/preprocessed_gmic_random/" \
--mask_dir="checkpoints/model_1-cbis_rn18_pretrain_ep150_crop_random/segmentation/masks" \
--output_path="checkpoints/model_1-cbis_rn18_pretrain_ep150_crop_random/segmentation" \
--bs=6 \
--percent_t=0.02 \
--gpuid 0 \
--v1_global=True \
--pretrained_model=True \
--num_chan=3