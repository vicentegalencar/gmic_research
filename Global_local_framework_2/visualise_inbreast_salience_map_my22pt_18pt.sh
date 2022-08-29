#!/bin/bash


# cd "/content/gmic_research/Global_local_framework_2/"

# export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"
export PYTHONPATH=$PYTHONPATH:"/content/gmic_research/Global_local_framework_2/"

python src/scripts/myvisualise_model_INBREAST.py \
--model_path="models/cam_k3_vindr_val_best_model.pth" \
--data_path="/content/drive/MyDrive/PNG_MASKS" \
--mask_dir="checkpoints/inbreast/segmentation/masks" \
--output_path="checkpoints/inbreast/segmentation" \
--bs=4 \
--pretrained_model=True \
--percent_t=0.02 \
--gpuid 0 \
--num_chan=1