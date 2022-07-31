import os
import torch
import imageio
import numpy as np
from .transforms import transform, transform1
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from PIL import Image

class Mammo(Dataset):
    def __init__(self, root, csv_path, image_size, max_value, aug):
        self.root = root
        self.data_type = csv_path.split(os.sep)[-1].split('.')[0]
        self.imgsize = image_size
        self.max_value = max_value
        self.aug = aug
        with open(csv_path) as file:
            self.lines = file.readlines()
            self.lines.pop(0)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if self.data_type == 'train' or self.data_type == 'validate':
            _, _, filename, label = self.lines[idx].split(',')
        elif self.data_type == 'test':
            _, filename, label = self.lines[idx].split(',')

        filepath = os.path.join(self.root, filename)
        label = torch.tensor([int(label)], dtype=torch.float32)
        img = np.array(imageio.imread(filepath), dtype=np.float32)
        # In training, flip right breast images such that all breasts are facing right
        view = filename.rsplit('-', 2)[1].split('_')[2]
        if view == 'R':
            img = np.fliplr(img)
        if self.aug:
            img = transform(img, self.imgsize, self.max_value, type=self.data_type)
        else:
            img = transform1(img, self.imgsize, type=self.data_type)
        return img, label

class CBIS(Dataset):
    def __init__(self, root, data_type, image_size, max_value, aug, num_chan):
        self.root = root
        #self.data_type = csv_path.split(os.sep)[-1].split('.')[0]
        self.data_type = data_type
        self.imgsize = image_size
        self.max_value = max_value
        self.aug = aug
        self.num_chan = num_chan
        # self.csv_path = csv_path
        # with open(csv_path) as file:
        #     self.lines = file.readlines()
        #     self.lines.pop(0)
        if data_type == 'train' or data_type == 'validate':
            df = pd.read_csv(os.path.join(root,"annotations_train.csv"))
            # df = pd.read_csv(os.path.join(data_path,"annotations_train.csv"))
            # import pdb; pdb.set_trace()
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(root,"train","full",x) for x in img_paths]
            self.mask_path = [os.path.join(root,"train","merged_masks",x) for x in mask_paths]
            self.labels = targets

        elif data_type == 'train_rnd':
            df = pd.read_csv(os.path.join(root,"annotation_random_train1.csv"))
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(root,"all_full",x) for x in img_paths]
            self.mask_path = [os.path.join(root,"all_merged_masks",x) for x in mask_paths]
            self.labels = targets

        elif data_type == 'test':
            df = pd.read_csv(os.path.join(root,"annotations_test.csv"))
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(root,"test","full",x) for x in img_paths]
            self.mask_path = [os.path.join(root,"test","merged_masks",x) for x in mask_paths]
            self.labels = targets

        elif data_type == 'test_rnd':
            df = pd.read_csv(os.path.join(root,"annotation_random_test1.csv"))
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(root,"all_full",x) for x in img_paths]
            self.mask_path = [os.path.join(root,"all_merged_masks",x) for x in mask_paths]
            self.labels = targets


    def __len__(self):
        #return len(self.lines)
        return len( self.img_path)

    def __getitem__(self, idx):
        # if self.data_type == 'train' or self.data_type == 'validate':
        #     # _, _, filename, label = self.lines[idx].split(',')
        #     df = pd.read_csv(os.path.join(self.csv_path,"annotations_train.csv"))
        # elif self.data_type == 'test':
        #     _, filename, label = self.lines[idx].split(',')


    
        # filepath = os.path.join(self.root, filename)
        filepath = self.img_path[idx]
        label = self.labels[idx]
        label = torch.tensor([int(label)], dtype=torch.float32)
        #img = np.array(imageio.imread(filepath), dtype=np.float32)
        #img = np.array(imageio.imread(filepath, pilmode='RGB'), dtype=np.float32)
        #img = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32)
        if self.num_chan ==1:
            img = np.array(Image.open(filepath), dtype=np.float32)
        else:
            img = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32)
        
        # In training, flip right breast images such that all breasts are facing right
        # # view = filename.rsplit('-', 2)[1].split('_')[2]
        # import pdb; pdb.set_trace()

        #if right view
        if "RIGHT" in filepath.split("/")[-1]:  
            img = np.fliplr(img)
        # # if view == 'R':
        #     img = np.fliplr(img)
        if self.aug:
            img = transform(img, self.imgsize, self.max_value, type=self.data_type)
        else:
            img = transform1(img, self.imgsize, type=self.data_type)
        return img, label, filepath

class VINDR(Dataset):
    def __init__(self, root, data_type, image_size, max_value, aug, num_chan):
        self.root = root
        #self.data_type = csv_path.split(os.sep)[-1].split('.')[0]
        self.data_type = data_type
        self.imgsize = image_size
        self.max_value = max_value
        self.aug = aug
        self.num_chan = num_chan
        # self.csv_path = csv_path
        # with open(csv_path) as file:
        #     self.lines = file.readlines()
        #     self.lines.pop(0)
        if data_type == 'train' or data_type == 'validate':
            #df = pd.read_csv(os.path.join(root,"annotations_train.csv"))
            df = pd.read_csv(os.path.join(root,"balanced_mass_nof.csv"))

            df = df[df.fold == "training"]

            # df = pd.read_csv(os.path.join(data_path,"annotations_train.csv"))
            # import pdb; pdb.set_trace()

            img_paths = []
            targets = []
            locations = []
            for _, row in df.iterrows(): 
                study_id = row.study_id
                image_id = row.image_id

                xmin = row.xmin
                ymin = row.ymin
                xmax = row.xmax
                ymax = row.ymax

                img_path = f"images/{study_id}/{image_id}_pre.png"
                img_paths.append(img_path)
                # import pdb; pdb.set_trace()
                # for clas in row.finding_categories:
                #if 'Mass' in clas:
                if 'Mass' in row.finding_categories:
                    targets.append(1)
                    locations.append({'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
                #elif 'No Finding' in clas:
                elif 'No Finding' in row.finding_categories:
                    targets.append(0)
                    # locations.append(None)
                    locations.append({'xmin':-1, 'ymin':-1, 'xmax':-1, 'ymax':-1})

            # img_paths = df["image_full_png_path"].values
            # mask_paths = df["image_mask_png_path"].values
            # targets = df["binary pathology"].values
            
            # self.img_path = [os.path.join(root,"train","full",x) for x in img_paths]
            
            self.img_path = [os.path.join(root,x) for x in img_paths]
            # self.mask_path = [os.path.join(root,"train","merged_masks",x) for x in mask_paths]
            self.labels = targets
            self.locations = locations
            


        elif data_type == 'test':
            df = pd.read_csv(os.path.join(root,"balanced_mass_nof.csv"))

            df = df[df.fold == "test"]
            img_paths = []
            targets = []
            locations = []
            for _, row in df.iterrows(): 
                study_id = row.study_id
                image_id = row.image_id

                xmin = row.xmin
                ymin = row.ymin
                xmax = row.xmax
                ymax = row.ymax

                img_path = f"images/{study_id}/{image_id}_pre.png"
                img_paths.append(img_path)
                #for clas in row.finding_categories:
                if 'Mass' in row.finding_categories:
                    targets.append(1)
                    locations.append({'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
                #elif 'No Finding' in clas:
                elif 'No Finding' in row.finding_categories:
                    targets.append(0)
                    # locations.append(None)
                    locations.append({'xmin':-1, 'ymin':-1, 'xmax':-1, 'ymax':-1})

            # self.img_path = img_paths
            self.img_path = [os.path.join(root,x) for x in img_paths]
            self.labels = targets
            self.locations = locations
            # import pdb; pdb.set_trace()



    def __len__(self):
        #return len(self.lines)
        return len( self.img_path)

    def __getitem__(self, idx):
       
        filepath = self.img_path[idx]
        label = self.labels[idx]
        label = torch.tensor([int(label)], dtype=torch.float32)
        
        if self.num_chan ==1:
            img = np.array(Image.open(filepath), dtype=np.float32)
        else:
            img = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32)

        location = self.locations[idx]
        
        # In training, flip right breast images such that all breasts are facing right
        # # view = filename.rsplit('-', 2)[1].split('_')[2]
        # import pdb; pdb.set_trace()

        #if right view
        # if "RIGHT" in filepath.split("/")[-1]:  
        #     img = np.fliplr(img)
        # # if view == 'R':
        #     img = np.fliplr(img)
        scale = (img.shape[0]/self.imgsize[0], img.shape[1]/self.imgsize[1])
        if location !=None:
            location['xmin'] /= scale[1]
            location['xmax'] /= scale[1]
            location['ymin'] /= scale[0]
            location['ymax'] /= scale[0]

        if self.aug:
            img = transform(img, self.imgsize, self.max_value, type=self.data_type)
        else:
            img = transform1(img, self.imgsize, type=self.data_type)
        return img, label, filepath, location

class INBREAST(Dataset):
    def __init__(self, root, data_type, image_size, max_value, aug, num_chan):
        self.root = root
        
        self.data_type = data_type
        self.imgsize = image_size
        self.max_value = max_value
        self.aug = aug
        self.num_chan = num_chan
        
        if data_type == 'train' or data_type == 'validate':
            
            # df = pd.read_csv(os.path.join(root,"balanced_mass_nof.csv"))
            # df = df[df.fold == "training"]

            # img_paths = []
            # targets = []
            # locations = []
            # for _, row in df.iterrows(): 
            #     study_id = row.study_id
            #     image_id = row.image_id

            #     xmin = row.xmin
            #     ymin = row.ymin
            #     xmax = row.xmax
            #     ymax = row.ymax

            #     img_path = f"images/{study_id}/{image_id}_pre.png"
            #     img_paths.append(img_path)
                
            #     #if 'Mass' in clas:
            #     if 'Mass' in row.finding_categories:
            #         targets.append(1)
            #         locations.append({'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
            #     #elif 'No Finding' in clas:
            #     elif 'No Finding' in row.finding_categories:
            #         targets.append(0)
                    
            #         locations.append({'xmin':-1, 'ymin':-1, 'xmax':-1, 'ymax':-1})
            
            # self.img_path = [os.path.join(root,x) for x in img_paths]
            # self.labels = targets
            # self.locations = locations

            ##vicente's code
            df = pd.read_csv(os.path.join(root,"TrainDataset.csv"))
            img_paths = df["Img_File_Name"].values   
            mask_paths = df["Mask_File_Name"].values
            targets = df["Mass_Type"].values
            
            self.img_path = [os.path.join(root,"TrainDataset","full",x) for x in img_paths] 
            self.mask_path = [os.path.join(root,"TrainDataset","masks",x) for x in mask_paths]
            self.labels = targets
            self.locations = None
            #end vicentes code
            


        elif data_type == 'test':
            # df = pd.read_csv(os.path.join(root,"balanced_mass_nof.csv"))
            # df = df[df.fold == "test"]
            # img_paths = []
            # targets = []
            # locations = []
            # for _, row in df.iterrows(): 
            #     study_id = row.study_id
            #     image_id = row.image_id

            #     xmin = row.xmin
            #     ymin = row.ymin
            #     xmax = row.xmax
            #     ymax = row.ymax

            #     img_path = f"images/{study_id}/{image_id}_pre.png"
            #     img_paths.append(img_path)
            #     #for clas in row.finding_categories:
            #     if 'Mass' in row.finding_categories:
            #         targets.append(1)
            #         locations.append({'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
            #     #elif 'No Finding' in clas:
            #     elif 'No Finding' in row.finding_categories:
            #         targets.append(0)
            #         # locations.append(None)
            #         locations.append({'xmin':-1, 'ymin':-1, 'xmax':-1, 'ymax':-1})

            # self.img_path = [os.path.join(root,x) for x in img_paths]
            # self.labels = targets
            # self.locations = locations

            #vicente's code
            df = pd.read_csv(os.path.join(root,"TestDataset.csv"))
            img_paths = df["Img_File_Name"].values 
            mask_paths = df["Mask_File_Name"].values
            targets = df["Mass_Type"].values
            
            self.img_path = [os.path.join(root,"TestDataset","full",x) for x in img_paths]  
            self.mask_path = [os.path.join(root,"TestDataset","masks",x) for x in mask_paths]
            self.labels = targets
            self.locations = None
            #end vicente's code
        
    def __len__(self):
        #return len(self.lines)
        return len( self.img_path)

    def __getitem__(self, idx):
       
        filepath = self.img_path[idx]
        label = self.labels[idx]
        label = torch.tensor([int(label)], dtype=torch.float32)
        
        if self.num_chan ==1:
            img = np.array(Image.open(filepath), dtype=np.float32)
        else:
            img = np.array(Image.open(filepath).convert('RGB'), dtype=np.float32)

        #location = self.locations[idx]
        location = None
        
        # scale = (img.shape[0]/self.imgsize[0], img.shape[1]/self.imgsize[1])
        # if location !=None:
        #     location['xmin'] /= scale[1]
        #     location['xmax'] /= scale[1]
        #     location['ymin'] /= scale[0]
        #     location['ymax'] /= scale[0]

        if self.aug:
            img = transform(img, self.imgsize, self.max_value, type=self.data_type)
        else:
            img = transform1(img, self.imgsize, type=self.data_type)
        return img, label, filepath, location

def get_dataloader(datapth, csv_path, image_size, batch_size, shuffle, max_value, aug):
    return DataLoader(
        Mammo(datapth, csv_path, image_size, max_value, aug),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

def get_dataloaderCBIS(datapth, data_type, image_size, batch_size, shuffle, max_value, aug, num_chan):
    return DataLoader(
        CBIS(datapth, data_type, image_size, max_value, aug, num_chan ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

def get_dataloaderVINDR(datapth, data_type, image_size, batch_size, shuffle, max_value, aug, num_chan):
    return DataLoader(
        VINDR(datapth, data_type, image_size, max_value, aug, num_chan ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

def get_dataloaderINBREAST(datapth, data_type, image_size, batch_size, shuffle, max_value, aug, num_chan):
    return DataLoader(
        INBREAST(datapth, data_type, image_size, max_value, aug, num_chan ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

