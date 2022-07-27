#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model.py \
--model_path="checkpoints/model_1-cbis_scratch_ep50/val_best_model.pth" \
--data_path="/media/hdd/filipe/datasets/preprocessed/" \
--mask_dir="checkpoints/model_1-cbis_scratch_ep50/segmentation/masks" \
--output_path="checkpoints/model_1-cbis_scratch_ep50/segmentation" \
--bs=6 \
--percent_t=0.02 \
--gpuid 1