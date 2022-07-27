# Introduction

This folder contains the code of global+local framework_2 which implements the method proposed in [An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization](https://arxiv.org/abs/1906.02846). Most of the code is cloned from the public Github repo [https://github.com/nyukat/GMIC](https://github.com/nyukat/GMIC) with a minor modification.

Two sub-folders(```sample_data``` and ```src```) are included in this folder. Sub-folder ```sample_data``` contains 16 example images from 4 exams (each exam has images taken at 4 standard views) and their benign/malignant masks (if exist) as well as the information of these exams. These images are used to visualise the effectiveness of the developed GMIC model. Detailed description of these images and exams can be found in the "Data" Section of [https://github.com/nyukat/GMIC](https://github.com/nyukat/GMIC). Sub-folder ```sample_data``` contains the source code of training and evaluation of GMIC models.


The following main files are included in the ```src``` sub-folder

**src/utilities/split_datasets.py**
* Split training data into training set and validation set
* Generate the csv files which save the filenames of the training, validation and test sets

**src/utilities/metric.py**
* Calculate the performance metrics including classification accuracy, AUC, TP, FP, TN and FN   

**src/data_loading/datasets.py**
* Read the csv files containing training, validation and test image filenames
* Load a batch of image data from the csv file
* Flip the right-view images such that the breasts in all images are facing right

**src/data_loading/loading.py**
* Read an image and performance image-wise standardisation such that each image has zero mean and unit standard deviation

**src/data_loading/transforms.py**
* Perform image augmentation during training stage

**src/modeling/gmic.py**
* Implement the whole framework of the proposed Globally-aware Multiple Instance Classifier (GMIC)

**src/modeling/modules.py**
* Implement the global module, local module and fusion module in GMIC
  
**src/scripts/train.py**
* Re-train GMIC on SV training data
* Evaluate the performance of re-trained GMIC on SV test data 
  
**src/scripts/one_epoch.py**
* Calculate the loss function at the end of each training epoch and update network weights according to the loss function
* Calculate performance metrics on training and validation sets at the end of each training epoch; save them to a csv file
* Save the best re-trained model which achieves the highest auc on validation set so far 

**src/scripts/eval_model.py**
* Evaluate the performance of the best re-trained GMIC on SV test data 

**src/scripts/visualise_model.py**
* Visualise the ground-truth annotation of malignant lesions and the saliency map of malignant lesions predicted from re-trained/pre-trained classification model 

**split_datasets.sh**
* Example script to split the training data into training and validation sets, and generate the csv files which save the filenames of the training, validation and test sets

**retrain_model_1.sh**
* Example script to re-train pre-trained GMIC model with setting 1 on SV data (You can re-train other pre-trained GMIC models by modifying the relevant hyper-parameters in this script)

**model_evaluation.sh**
* Example script to evaluate the performance of classification models on specific data set

**visualise_salience_map.sh**
* Example script to visualise the ground-truth annotation of malignant lesions and the saliency map of malignant lesions predicted from classification models on specified images (Note: in order to visualise the ground-truth annotation of malignant lesions, you should provide the pixel-level annotation of malignant lesions)