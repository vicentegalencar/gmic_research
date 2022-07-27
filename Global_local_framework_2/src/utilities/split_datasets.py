import os, argparse
import pandas as pd
from sklearn.utils import shuffle


ap = argparse.ArgumentParser()
ap.add_argument("--train_cancer_path", type=str, default='data/cropped_mammo-16/train/cancer')
ap.add_argument("--train_normal_path", type=str, default='data/cropped_mammo-16/train/normal')
ap.add_argument("--test_cancer_path", type=str, default='data/cropped_mammo-16/test/cancer')
ap.add_argument("--test_normal_path", type=str, default='data/cropped_mammo-16/test/normal')
ap.add_argument("--val_size", type=float, default=0.1)
ap.add_argument("--train_csv_path", type=str, default='data/cropped_mammo-16/train.csv')
ap.add_argument("--val_csv_path", type=str, default='data/cropped_mammo-16/validate.csv')
ap.add_argument("--test_csv_path", type=str, default='data/cropped_mammo-16/test.csv')
args = vars(ap.parse_args())


datasets_train_cancer_path = args['train_cancer_path']
datasets_train_normal_path = args['train_normal_path']

train_cancer_filelist = os.listdir(datasets_train_cancer_path) # label 1
train_normal_filelist = os.listdir(datasets_train_normal_path) # label 0

file_sets = {
    'name' : [],
    'label' : []
}

for filename in train_cancer_filelist:
    file_sets['name'].append('cancer/' + filename)
    file_sets['label'].append(1)

for filename in train_normal_filelist:
    file_sets['name'].append('normal/' + filename)
    file_sets['label'].append(0)
    
val_size = args['val_size']

frame = pd.DataFrame(file_sets)
frame = shuffle(frame)
frame = frame.reset_index()
total = len(frame)
ratio = [1-val_size, val_size]
a, b = ratio[0]*total, ratio[1]*total
train = frame.loc[0:a]
validate = frame.loc[a:]
print('training data:{}, training set:{}, validation set:{}', total, len(train), len(validate))
train.to_csv(args['train_csv_path'])
validate.to_csv(args['val_csv_path'])

datasets_test_cancer_path = args['test_cancer_path']
datasets_test_normal_path = args['test_normal_path']

test_cancer_filelist = os.listdir(datasets_test_cancer_path) # label 1
test_normal_filelist = os.listdir(datasets_test_normal_path) # label 0

file_sets = {
    'name' : [],
    'label' : []
}

for filename in test_cancer_filelist:
    file_sets['name'].append('cancer/' + filename)
    file_sets['label'].append(1)

for filename in test_normal_filelist:
    file_sets['name'].append('normal/' + filename)
    file_sets['label'].append(0)

test = pd.DataFrame(file_sets)
test.to_csv(args['test_csv_path'])
