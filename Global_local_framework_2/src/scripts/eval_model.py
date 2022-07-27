# evaluate performance of model
import os
import torch
import pandas as pd
import numpy as np
from torch import dtype, mode
from src.modeling import gmic
from src.utilities.metric import compute_metric
from src.data_loading.data import get_dataloader
import argparse
import cv2
import imageio

    
    
@torch.no_grad()
def eval_net(model, loader, device, threshold):
    model.eval()

    prediction = np.empty(shape=[0, 2], dtype=np.int)
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        y_global, y_local, y_fusion, saliency_map = model(imgs)
        
        y_fusion = y_fusion[:,1:]
        result = np.concatenate([y_fusion.cpu().data.numpy(), labels.cpu().data.numpy()], axis=1)
        prediction = np.concatenate([prediction, result], axis=0)
       

    print('==> ### test metric ###')
    total = len(loader.dataset)
    TP, FN, TN, FP, acc, roc_auc = compute_metric(prediction, threshold)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    print('Total: %d'%(total))
    print('threshold: %.2f --- TP: %d --- FN: %d --- TN: %d --- FP: %d'%(threshold, TP, FN, TN, FP))
    print('acc: %f --- roc_auc: %f --- sensitivity: %f --- specificity: %f'%(acc, roc_auc, sensitivity, specificity))
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default='models/sample_model_1.p',
        help="pretrained model path")
    ap.add_argument("--data_csv_path", type=str, default='data/cropped_mammo-16/test.csv',
        help="test data csv path")
    ap.add_argument("--data_path", type=str, default='data/cropped_mammo-16/test',
        help="test data path")
    ap.add_argument("--bs", type=int, default=6,
        help="batch size")
    ap.add_argument("--is", type=int, default=(2944, 1920),
        help="image size")
    ap.add_argument("--beta", type=float, default=3.259162430057801e-06,
        help="beta")
    ap.add_argument("--percent_t", type=float, default=0.02,
        help="percent for top-T pooling")
    ap.add_argument("--aug", type=bool, default=False,
        help="whether use data augmentation")
    ap.add_argument("--pretrained_model", type=bool, default=False,
        help="whether use pretrained_model")
    args = vars(ap.parse_args())

   
    Test_data = args['data_csv_path']
    DATA_PATH = args['data_path']
    num_works = 4

    device_type = 'gpu'
    gpu_id = 0
    threshold = 0.5
    model_path = args['model_path']
    beta = args['beta']
    percent_t = args['percent_t']

    img_size = args['is']
    batch_size = args['bs']
    aug = args['aug']
    pretrained_model = args['pretrained_model']
        
    max_value=65535

    parameters = {
        "device_type": device_type,
        "gpu_number": gpu_id,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        # model related hyper-parameters
        "percent_t": percent_t,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
    }

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    model = gmic.GMIC(parameters)
    if device_type == 'gpu' and torch.has_cudnn:
        if pretrained_model:
            model.load_state_dict(torch.load(model_path), strict=False)
        else:
            model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=True)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        if pretrained_model:
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load((model_path)['model_state_dict'],  map_location="cpu"), strict=True)
        device = torch.device("cpu")
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    test_loader = get_dataloader(os.path.join(DATA_PATH, 'test'), Test_data, image_size=img_size, batch_size=batch_size, shuffle=False, \
        max_value=max_value)
    
    print(model_path.split(os.sep)[-1])

    eval_net(model, test_loader, device, threshold)
