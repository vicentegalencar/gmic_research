#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model_VINDR.py \
--model_path="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt_k3/" \
--data_path="/media/hdd/filipe/datasets/vindr-mammo/1.0.0/" \
--mask_dir="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt_k3/segmentation/masks" \
--output_path="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt_k3/" \
--bs=4 \
--percent_t=0.02 \
--gpuid 1 \
--num_chan=1 \
--kvalue=3