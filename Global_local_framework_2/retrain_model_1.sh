#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"

export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"

python src/scripts/train.py \
--model_path="pretrain_models/sample_model_1.p" \
--save_model_path="checkpoints/model_1-16" \
--train_data_csv="data/cropped_mammo-16/train.csv" \
--val_data_csv="data/cropped_mammo-16/validate.csv" \
--test_data_csv="data/cropped_mammo-16/test.csv" \
--data_path="data/cropped_mammo-16/" \
--epochs=50 \
--lr=4.134478662168656e-05 \
--lr_step=10 \
--bs=6 \
--beta=3.259162430057801e-06 \
--percent_t=0.02 \
--augmentation=True