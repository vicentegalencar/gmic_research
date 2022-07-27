#!/bin/bash

#cd "Global_local_framework_2/"
#cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"
cd "/mnt/c/filipe/codes/gmic_mdpp-master/Global_local_framework_2"

#export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
#export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"
export PYTHONPATH=$PYTHONPATH:"/mnt/c/filipe/codes/gmic_mdpp-master/"

python src/scripts/mytrain.py \
--save_model_path="checkpoints/cbis_rn22sc_rn18pt_ep150_crop_random" \
--data_path="/mnt/c/filipe/datasets/preprocessed_gmic_random/" \
--epochs=150 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True \
--pretrain_local=True \
--num_chan=1 \
--gpuid=0


