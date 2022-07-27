#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model.py \
--model_path="models/sample_model_1.p" \
--data_path="/media/hdd/filipe/datasets/preprocessed_gmic/" \
--mask_dir="checkpoints/model_1-cbis_rn22_ep50_cropped/segmentation/masks" \
--output_path="checkpoints/model_1-cbis_rn22_ep50_cropped/segmentation" \
--bs=4 \
--pretrained_model=True \
--percent_t=0.02 \
--gpuid 0 \
--num_chan=1