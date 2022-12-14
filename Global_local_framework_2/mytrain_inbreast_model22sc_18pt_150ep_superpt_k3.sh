#!/bin/bash

#cd "Global_local_framework_2/"
#cd "/media/hdd/filipe/codes/gmic_mdpp-master/Global_local_framework_2/"
cd "/media/hdd/filipe/codes/gmic_research/Global_local_framework_2"

#export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
#export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_mdpp-master/"
export PYTHONPATH=$PYTHONPATH:"/media/hdd/filipe/codes/gmic_research/"

python src/scripts/mytrain_inbreast.py \
--save_model_path="checkpoints/inbreast_rn22pt_rn18pt_ep50_superpt_k5_cam" \
--data_path="/media/hdd/filipe/datasets/PNG_MASKS/" \
--epochs=50 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True \
--num_chan=1 \
--gpuid=0  \
--K=5 \
--cam_method="cam"


