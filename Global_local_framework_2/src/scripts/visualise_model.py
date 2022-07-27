# visualize saliency map predicted from the model
import os
import torch
import pandas as pd
import numpy as np
from torch import dtype, mode
from src.modeling import gmic
from src.utilities.metric import compute_metric
from src.data_loading.data import get_dataloader
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utilities import pickling, tools
import cv2
import imageio

def visualize_example(input_img, saliency_maps, true_segs,
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    _, _, H, W = input_img.shape
    # convert tensor to numpy array
    input_img = input_img.data.cpu().numpy()

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas
    alpha_orange = plt.cm.get_cmap('Oranges')
    alpha_orange._init()
    alpha_orange._lut[:, -1] = alphas
    

    # create visualization template
    total_num_subplots = 4 + parameters["K"]
    # total_num_subplots = 3 + parameters["K"]
    figure = plt.figure(figsize=(30, 3))
    # input image + segmentation map
    subfigure = figure.add_subplot(1, total_num_subplots, 1)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    benign_seg, malignant_seg = true_segs
    if benign_seg is not None:
        cm.Greens.set_under('w', alpha=0)
        subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
    if malignant_seg is not None:
        cm.OrRd.set_under('w', alpha=0)
        subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
    subfigure.set_title("input image")
    subfigure.axis('off')

    # patch map
    subfigure = figure.add_subplot(1, total_num_subplots, 2)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    cm.YlGnBu.set_under('w', alpha=0)
    crop_mask = tools.get_crop_mask(
        patch_locations[0, np.arange(parameters["K"]), :],
        parameters["crop_shape"], (H, W),
        "upper_left")
    subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
    if benign_seg is not None:
        cm.Greens.set_under('w', alpha=0)
        subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
    if malignant_seg is not None:
        cm.OrRd.set_under('w', alpha=0)
        subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
    subfigure.set_title("patch map")
    subfigure.axis('off')
    
    subfigure = figure.add_subplot(1, total_num_subplots, 3)
    subfigure.imshow(np.zeros((H,W), dtype=np.float32), cmap='gray')
    # subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(resized_cam_malignant, cmap='gray', clim=[0.0, 1.0])
    subfigure.set_title("saliency maps")
    subfigure.axis('off')
    

    # class activation maps
    subfigure = figure.add_subplot(1, total_num_subplots, 4)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(resized_cam_malignant, cmap=alpha_orange, clim=[0.0, 1.0])
    subfigure.set_title("overlayed malignant")
    subfigure.axis('off')


    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 5 + crop_idx)
        # subfigure = figure.add_subplot(1, total_num_subplots, 4 + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    #================
    plt.show()
    #================
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()
    
def visualize_saliency_patch_maps(model, output_path, loader, device, test_filename, mask_dir, mode):
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        y_global, y_local, y_fusion, saliency_map = model(imgs)
        saliency_maps = model.saliency_map.data.cpu().numpy()
        patch_locations = model.patch_locations
        patch_imgs = model.patches
        # patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
        patch_attentions = model.patch_attns.data.cpu().numpy()
        batch_size = imgs.size()[0]
        for i in range(batch_size):
        # save_dir = os.path.join("visualization", "{0}.png".format(short_file_path))
            filename = test_filename[i]
            filename = filename[filename.find('/')+1:filename.find('.')]
            save_dir = os.path.join(output_path, "{}.png".format(filename))
            print('processing {}'.format(filename))
            #===========================================================================
            # load segmentation if available
            # benign_seg_path = os.path.join(mask_dir, "{0}_{1}".format(filename, "benign.png"))
            malignant_seg_path = os.path.join(mask_dir, "{0}_{1}".format(filename, "malignant.png"))
            benign_seg = None
            malignant_seg = None
            # if os.path.exists(benign_seg_path):
            #     mask_image = np.array(imageio.imread(benign_seg_path))
            #     benign_seg = mask_image.astype(np.float32)
            if os.path.exists(malignant_seg_path):
                mask_image = np.array(imageio.imread(malignant_seg_path))
                malignant_seg = mask_image.astype(np.float32)
            #=========================================================================
                    
            visualize_example(imgs[i:i+1,:,:,:], saliency_maps[i:i+1,:,:,:], [benign_seg, malignant_seg],
                    patch_locations[i:i+1,:,:], patch_imgs[i:i+1,:,:,:], patch_attentions[i],
                    save_dir, parameters, mode)
       
    

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
    ap.add_argument("--percent_t", type=float, default=0.02,
        help="percent for top-T pooling")
    ap.add_argument("--mask_dir", type=str, default='data/mammo-test-samples/segmentation', 
        help='mask directory')
    ap.add_argument("--output_path", type=str, default='retrained_visualization', 
        help='visualization directory')
    ap.add_argument("--mammo_sample_data", type=bool, default=False,
        help="whether visualise the provided sample data")
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
    
    
    test_filename = test_loader.dataset.lines
    
    # visuaize segmentation maps
    if not args['mammo_sample_data']:
        mask_dir = "data/sample_data/segmentation"
        output_path = 'visualization'
    else:
        mask_dir = args('mask_dir')
        output_path = args('output_path')
    if pretrained_model:
        visualize_saliency_patch_maps(model, output_path, test_loader, device, test_filename, mask_dir)
    else:
        visualize_saliency_patch_maps(model, output_path, test_loader, device, test_filename, mask_dir)