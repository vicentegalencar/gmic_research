#!/bin/bash

#cd "Global_local_framework_2/"
cd "/mnt/c/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"

#export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/mnt/c/filipe/codes/gmic_mdpp-master/"

python src/scripts/mytrain_superpt.py \
--save_model_path="checkpoints/cbis_superpt_rn22_rn18_ep150_crop" \
--data_path="/mnt/c/filipe/datasets/preprocessed_gmic/" \
--epochs=150 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True \
--num_chan=1 \
--gpuid=0


