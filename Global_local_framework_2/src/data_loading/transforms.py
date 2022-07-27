import torch
import albumentations as A
from src.data_loading import loading

def transform(image, img_size, max_value, type='train'):

    # Resize
    affine = A.Compose([
        A.ToFloat(max_value=max_value, p=1),
        A.Resize(img_size[0], img_size[1], p=1)
    ])
    augmented = affine(image=image)
    image = augmented['image']
       
    #if type == 'train' :
    if type in ['train', 'train_rnd', 'test_rnd']:
       
        # data augmentation
        affine = A.Compose([
            A.ElasticTransform(),
            A.GaussNoise(var_limit=20),
            # A.Equalize(),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45)           
            ])
        augmented = affine(image=image)
        image = augmented['image']
        
        

    elif type == 'test' or 'validate':
        pass
        # Not Implement
        '''
        During test phase, we similarly apply data augmentation and
        average predictions over 10 random augmentations to compute
        the prediction for a given image
   
        '''
        
        
    loading.standard_normalize_single_image(image)
    
    #if 2 dimensions (grayscale)
    if len(image.shape) == 2:
        image = torch.Tensor(image).unsqueeze(dim=0)
    else:  #if 3 channels
        image = torch.Tensor(image)
        image = image.permute((2,0,1))
    return image

def transform1(image, img_size, type='train'):
    
    # Resize
    affine = A.Compose([
        A.Resize(img_size[0], img_size[1])
    ], p=1)
    augmented = affine(image=image)
    image = augmented['image']
    
    loading.standard_normalize_single_image(image)

    if type == 'train':
        pass
        # No data augmentation

    elif type == 'test':
        pass
        # Not Implement
        '''
        During test phase, we similarly apply data augmentation and
        average predictions over 10 random augmentations to compute
        the prediction for a given image
        '''

    #if 2 dimensions (grayscale)
    if len(image.shape) == 2:
        image = torch.Tensor(image).unsqueeze(dim=0)
    else:
        image = torch.Tensor(image)
        image = image.permute((2,0,1))
    return image