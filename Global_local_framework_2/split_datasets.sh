#!/bin/bash

cd "Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"

export PYTHONPATH=$PYTHONPATH:"Validate_and_Improve_Breast_Cancer_AI_Approach/Global_local_framework_2/"

python src/utilities/split_datasets.py \
--train_cancer_path="data/cropped_mammo-16/train/cancer" \
--train_normal_path="data/cropped_mammo-16/train/normal" \
--test_cancer_path="data/cropped_mammo-16/test/cancer" \
--test_normal_path="data/cropped_mammo-16/test/normal" \
--val_size=0.1
--train_csv_path="data/cropped_mammo-16/train.csv" \
--val_csv_path="data/cropped_mammo-16/validate.csv" \
--test_csv_path="data/cropped_mammo-16/test.csv"
