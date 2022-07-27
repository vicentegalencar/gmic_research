#!/bin/bash

#cd "Global_local_framework_2/"
#cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"
cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2"

#export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
#export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"

python src/scripts/mytrain_vindr_superpt_frozen.py \
--save_model_path="checkpoints/vindr_rn22pt_rn18pt_ep50_superpt_frozen2" \
--data_path="/media/hdd/filipe/datasets/vindr-mammo/1.0.0/" \
--epochs=50 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True \
--num_chan=1 \
--gpuid=0 \
--frozen_ep=1



