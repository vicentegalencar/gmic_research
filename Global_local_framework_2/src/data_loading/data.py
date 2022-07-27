import os
import torch
import imageio
import numpy as np
from .transforms import transform1
from torch.utils.data import Dataset,DataLoader

class Mammo(Dataset):
    def __init__(self, root, csv_path, image_size, max_value):
        self.root = root
        self.data_type = csv_path.split(os.sep)[-1].split('.')[0]
        self.imgsize = image_size
        self.max_value = max_value
        with open(csv_path) as file:
            self.lines = file.readlines()
            self.lines.pop(0)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            _, _, filename, label = self.lines[idx].split(',')
        elif self.data_type == 'test':
            _, filename, label = self.lines[idx].split(',')

        filepath = os.path.join(self.root, filename)
        label = torch.tensor([int(label)], dtype=torch.float32)
        img = np.array(imageio.imread(filepath), dtype=np.float32)
        # In training, flip right breast images such that all breasts are facing right
        # view = filename.rsplit('-', 2)[1].split('_')[2]
        # if view == 'R':
        #     img = np.fliplr(img)
        
        img = transform1(img, self.imgsize, type=self.data_type)
        # img = transform1(img, self.imgsize, type=self.data_type)
        return img, label

def get_dataloader(datapth, csv_path, image_size, batch_size, shuffle, max_value):
    return DataLoader(
        Mammo(datapth, csv_path, image_size, max_value),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )