#!/bin/bash


cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/myvisualise_model.py \
--model_path="checkpoints/cbis_rn22sc_rn18pt_ep150_crop/val_best_model.pth" \
--data_path="/media/hdd/filipe/datasets/preprocessed_gmic/" \
--mask_dir="checkpoints/cbis_rn22sc_rn18pt_ep150_crop/masks" \
--output_path="checkpoints/cbis_rn22sc_rn18pt_ep150_crop/segmentation" \
--bs=2 \
--pretrained_model=True \
--percent_t=0.02 \
--gpuid 0 \
--num_chan=1