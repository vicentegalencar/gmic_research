#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model_VINDR.py \
--model_path="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt/last_model.pth" \
--data_path="/media/hdd/filipe/datasets/vindr-mammo/1.0.0/" \
--mask_dir="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt/segmentation/masks" \
--output_path="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt/segmentation" \
--bs=4 \
--pretrained_model=True \
--percent_t=0.02 \
--gpuid 0 \
--num_chan=1