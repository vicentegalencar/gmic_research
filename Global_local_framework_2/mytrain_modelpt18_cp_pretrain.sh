#!/bin/bash

#cd "Global_local_framework_2/"
cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

#export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/mytrain.py \
--save_model_path="checkpoints/model_1-cbis_rn18_pretrain_ep50_crop" \
--data_path="/media/hdd/filipe/datasets/preprocessed_gmic/" \
--epochs=50 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True \
--gpuid=2 \
--v1_global=True \
--pretrain=True

