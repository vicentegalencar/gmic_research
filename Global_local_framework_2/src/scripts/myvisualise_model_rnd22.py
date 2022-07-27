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
from os.path import exists
from PIL import Image
import pdb
import torch.nn.functional as F
from src.data_loading.datasets import get_dataloaderCBIS

from torchcam.methods import SmoothGradCAMpp, CAM, ScoreCAM, GradCAM, GradCAMpp, XGradCAM, LayerCAM, SSCAM, ISCAM

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
    total_num_subplots = 5 + parameters["K"]
    #total_num_subplots = 4 + parameters["K"]
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
    # subfigure.imshow(np.zeros((H,W), dtype=np.float32), cmap='gray')
    # subfigure.imshow(np.zeros((H,W), dtype=np.float32), cmap='gray')
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='Greys_r')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap='gray', clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_malignant, cmap='Greys_r', clim=[0.0, 1.0])
    subfigure.set_title("saliency maps")
    subfigure.axis('off')
    

    # class activation maps
    subfigure = figure.add_subplot(1, total_num_subplots, 4)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(resized_cam_malignant, cmap=alpha_orange, clim=[0.0, 1.0])
    subfigure.set_title("overlayed malignantx")
    subfigure.axis('off')

    # class activation maps benign
    subfigure = figure.add_subplot(1, total_num_subplots, 5)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(1-resized_cam_malignant, cmap=alpha_orange, clim=[0.0, 1.0])
    subfigure.set_title("overlayed benignx")
    subfigure.axis('off')


    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 6 + crop_idx)
        #subfigure = figure.add_subplot(1, total_num_subplots, 5 + crop_idx)
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

def visualize_example_git(input_img, saliency_maps, true_segs,
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    #_, _, H, W = input_img.shape
    H, W = input_img.shape
    # convert tensor to numpy array
    # input_img = input_img.data.cpu().numpy()

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas

    # create visualization template
    total_num_subplots = 4 + parameters["K"]
    figure = plt.figure(figsize=(30, 3))
    # input image + segmentation map
    subfigure = figure.add_subplot(1, total_num_subplots, 1)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
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
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
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

    # class activation maps
    subfigure = figure.add_subplot(1, total_num_subplots, 4)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("SM: malignant")
    subfigure.axis('off')

    subfigure = figure.add_subplot(1, total_num_subplots, 3)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    resized_cam_benign = cv2.resize(saliency_maps[0,0,:,:], (W, H))
    #subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.5, 1.0])
    subfigure.set_title("SM: benign")
    subfigure.axis('off')


    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 5 + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()

def alter_visualize_example_git(input_img, gt_mask, saliency_maps, true_segs,
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters,alter_cam_ben, alter_cam_malig, gradcam_ben,gradcam_malig, gradcampp_ben,gradcampp_malig,
                    xgradcam_ben, xgradcam_malig, layercam_ben, layercam_malig , info_cam_ben, info_cam_malig, y_fusion, label_gt):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    #_, _, H, W = input_img.shape
    H, W = input_img.shape
    # convert tensor to numpy array
    # input_img = input_img.data.cpu().numpy()

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas

    # create visualization template
    #total_num_subplots = 11 + parameters["K"]
    
    #figure = plt.figure(figsize=(30, 3))
    figure = plt.figure(figsize=(30, 20))
    # input image + segmentation map
    #subfigure = figure.add_subplot(1, total_num_subplots, 1)
    subfigure = figure.add_subplot(3, 8, 1)
    
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    benign_seg, malignant_seg = true_segs
    if benign_seg is not None:
        cm.Greens.set_under('w', alpha=0)
        subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.0, 1])
    if malignant_seg is not None:
        cm.OrRd.set_under('w', alpha=0)
        subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.0, 1])
    #  y_fusion
    # import pdb; pdb.set_trace()
    if y_fusion[1]>=0.5:
        model_pred = "M"
        prob = y_fusion[1]
    else:
        model_pred = "B"
        prob = y_fusion[0]
    subfigure.set_title("({}-{:.2f}/{:.2f}".format(model_pred,y_fusion[0],y_fusion[1]))
    subfigure.axis('off')

    #gt_mask
    #subfigure = figure.add_subplot(1, total_num_subplots, 2)
    subfigure = figure.add_subplot(3, 8, 2)
    subfigure.imshow(gt_mask, aspect='equal', cmap='gray')
    
    if label_gt==0:
        gt_name = "Malig"
    elif label_gt==1:
        gt_name = "Benig"
    subfigure.set_title("{}".format(gt_name))

    # patch map
    # subfigure = figure.add_subplot(1, total_num_subplots,3)
    subfigure = figure.add_subplot(3, 8, 3)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
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

    # class activation maps
    # subfigure = figure.add_subplot(1, total_num_subplots, 5)
    subfigure = figure.add_subplot(3, 8, 9)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    resized_cam_malignant_norm = (resized_cam_malignant- resized_cam_malignant.min())/resized_cam_malignant.max()
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_malignant_norm, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_cam_malignant_norm, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("SM: malignant")
    subfigure.axis('off')

    ## ALT SM MALIGN
    # subfigure = figure.add_subplot(1, total_num_subplots, 6)
    subfigure = figure.add_subplot(3, 8, 10)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # import pdb;pdb.set_trace()
    resized_cam_malignant2 = cv2.resize(alter_cam_malig[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_malignant2, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_cam_malignant2, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("altSM: malignant")
    subfigure.axis('off')

    ## GRAD CAM MALIGN
    subfigure = figure.add_subplot(3, 8, 11)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_malig = cv2.resize(gradcam_malig[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("gradcam: malign")
    subfigure.axis('off')

    ## GRAD CAM++ MALIGN
    subfigure = figure.add_subplot(3, 8, 12)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_malig = cv2.resize(gradcampp_malig[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.set_title("gradcam++: malign")
    subfigure.axis('off')

    ## XGRAD CAM MALIGN
    subfigure = figure.add_subplot(3, 8, 13)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_malig = cv2.resize(xgradcam_malig[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("XGradCam: malign")
    subfigure.axis('off')

    ## LAYER CAM MALIGN
    subfigure = figure.add_subplot(3, 8, 14)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_malig = cv2.resize(layercam_malig[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("Layercam: malign")
    subfigure.axis('off')

    ## INFOCAM MALIG
    #subfigure = figure.add_subplot(1, total_num_subplots, 11)
    subfigure = figure.add_subplot(3, 8, 15)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_infocam_malig = cv2.resize(info_cam_malig[0,:,:], (W, H))
    resized_infocam_malig_norm = (resized_infocam_malig- resized_infocam_malig.min())/resized_infocam_malig.max()
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_infocam_malig_norm, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_infocam_malig_norm, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("infoCAM: malign")
    subfigure.axis('off')

    ## class activation map - benig
    #subfigure = figure.add_subplot(1, total_num_subplots, 4)
    subfigure = figure.add_subplot(3, 8, 17)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    resized_cam_benign = cv2.resize(saliency_maps[0,0,:,:], (W, H))
    resized_cam_benign_norm = (resized_cam_benign- resized_cam_benign.min())/resized_cam_benign.max()
    #subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_benign_norm, cmap=alpha_green, clim=[0.0, 1.0])
    # subfigure.imshow(resized_cam_benign_norm, cmap=alpha_green, clim=[0.0, 1.0])
    subfigure.set_title("SM: benign")
    subfigure.axis('off')

     ## ALT SM BENIG
    # subfigure = figure.add_subplot(1, total_num_subplots, 6)
    subfigure = figure.add_subplot(3, 8, 18)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # import pdb;pdb.set_trace()
    resized_cam_benig = cv2.resize(alter_cam_ben[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_cam_benig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_cam_benig, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("altSM: benig")
    subfigure.axis('off')

    ## GRAD CAM BENIG
    subfigure = figure.add_subplot(3, 8, 19)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_ben = cv2.resize(gradcam_ben[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    #subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("gradcam: benig")
    subfigure.axis('off')

    ## GRAD CAM++ BENIG
    subfigure = figure.add_subplot(3, 8, 20)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_ben = cv2.resize(gradcampp_ben[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("gradcam++: benig")
    subfigure.axis('off')

    ## XGRAD CAM BENIG
    subfigure = figure.add_subplot(3, 8, 21)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_benig = cv2.resize(xgradcam_ben[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_benig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_benig, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("XGradCam: benig")
    subfigure.axis('off')

    ## LAYER CAM BENIG
    subfigure = figure.add_subplot(3, 8, 22)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_gradcam_ben = cv2.resize(layercam_ben[0,:,:], (W, H))
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_ben, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("Layercam: benig")
    subfigure.axis('off')

    ## INFOCAM BENIG
    #subfigure = figure.add_subplot(1, total_num_subplots, 11)
    subfigure = figure.add_subplot(3, 8, 23)
    #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(input_img, aspect='equal', cmap='gray')
    #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    resized_infocam_ben = cv2.resize(info_cam_ben[0,:,:], (W, H))
    resized_infocam_benig_norm = (resized_infocam_ben- resized_infocam_ben.min())/resized_infocam_ben.max()
    #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.imshow(resized_infocam_benig_norm, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_infocam_benig_norm, cmap=alpha_red, clim=[0.5, 1.0])
    subfigure.set_title("infoCAM: benig")
    subfigure.axis('off')

     # class activation maps
    # subfigure = figure.add_subplot(1, total_num_subplots, 7)
    # #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    # subfigure.imshow(input_img, aspect='equal', cmap='gray')
    # #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # resized_cam_benig2 = cv2.resize(alter_cam_ben[0,:,:], (W, H))
    # #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_cam_benig2, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.set_title("altSM: benign")
    # subfigure.axis('off')

     # class activation maps
    # subfigure = figure.add_subplot(1, total_num_subplots, 8)
    # #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    # subfigure.imshow(input_img, aspect='equal', cmap='gray')
    # #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # resized_gradcam_benig2 = cv2.resize(alter_gradcam_ben[0,:,:], (W, H))
    # #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_benig2, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.set_title("altGradSM: benign")
    # subfigure.axis('off')

    #  # class activation maps
    # subfigure = figure.add_subplot(1, total_num_subplots, 9)
    # #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    # subfigure.imshow(input_img, aspect='equal', cmap='gray')
    # #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # resized_gradcam_malig = cv2.resize(alter_gradcam_malig[0,:,:], (W, H))
    # #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_gradcam_malig, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.set_title("altGradSM: malign")
    # subfigure.axis('off')

    # #infocam - ben
    # # pdb.set_trace()
    # subfigure = figure.add_subplot(1, total_num_subplots, 10)
    # #subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    # subfigure.imshow(input_img, aspect='equal', cmap='gray')
    # #resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    # resized_infocam_ben = cv2.resize(info_cam_ben[0,:,:], (W, H))
    # resized_infocam_ben_norm = (resized_infocam_ben- resized_infocam_ben.min())/resized_infocam_ben.max()
    # #subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.imshow(resized_infocam_ben_norm, cmap=alpha_red, clim=[0.0, 1.0])
    # subfigure.set_title("infoCAM: benig")
    # subfigure.axis('off')

    

    


    # crops
    for crop_idx in range(parameters["K"]):
        # subfigure = figure.add_subplot(1, total_num_subplots, 12 + crop_idx)
        subfigure = figure.add_subplot(3, 8, 3+crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()
    
#def visualize_saliency_patch_maps(model, output_path, loader, device, test_filename, mask_dir, mode):
#def visualize_saliency_patch_maps(model, output_path, loader, device, mask_dir, mode):
def visualize_saliency_patch_maps(model, output_path, loader, device, mask_dir):
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for step, (imgs, labels, test_filename) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        # import pdb; pdb.set_trace()
        # model.eval()
        model.train()
        # gradcam_extractor = ScoreCAM(model, target_layer='ds_net.relu') #GradCAM
        # gradcam_extractor = ScoreCAM(model, target_layer='ds_net.layer_list.4') #GradCAM
        #gradcam_extractor = GradCAM(model, target_layer='ds_net.relu')

        #GradCam++
        gradcampp_extractor = GradCAMpp(model, target_layer='ds_net.relu') # [works fine]

        # GradCam
        gradcam__extractor = GradCAM(model, target_layer='ds_net.relu') # [works fine]


        #SmoothGradCAMpp
        # gradcampp_extractor = SmoothGradCAMpp(model, target_layer='ds_net.relu')  # [Failed]

        #XgradCam
        xgradcam_extractor = XGradCAM(model, target_layer='ds_net.relu')  # [works fine]

        #LayerCam
        #extractor = LayerCAM(model, target_layer='ds_net.relu')
        layercam_extractor = LayerCAM(model, target_layer=['ds_net.layer_list.4','ds_net.layer_list.3']) # [works fine, but need to change extractor]

        # gradcam_extractor = GradCAMpp(model, target_layer='ds_net.layer4',input_shape= (3, 2944, 1920))

        #GradCAMpp
        # cam_extractor = CAM(model, target_layer='ds_net.layer_list.4')
        #cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last',target_layer='ds_net.layer_list.4')
        #cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last')

        #CAM
        cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last',target_layer='ds_net.relu')  # [works fine]

        #ScoreCAM
        #cam_extractor = ScoreCAM(model, target_layer='ds_net.layer_list.4')  # [failed]

        # SSCAM
        #cam_extractor = SSCAM(model, target_layer='ds_net.relu')  # [failed]

        #ISCAM      
        # cam_extractor = ISCAM(model, target_layer='ds_net.relu')  # [failed]

        # cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last',target_layer='ds_net.layer4',input_shape= (3, 2944, 1920))
        #ds_net.relu
        # cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last',target_layer='ds_net.layer4',input_shape= (3, 2944, 1920))
        # cam_extractor = CAM(model, input_shape= (3, 2944, 1920))

        y_global, y_local, y_fusion, saliency_map, y_score, last_feature_map = model(imgs)

        # import pdb; pdb.set_trace()
        # with torch.no_grad(): y_global, y_local, y_fusion, saliency_map = model(imgs)
        #cam_extractor = SmoothGradCAMpp(model)
        #ds_net.layer_list.4/  ds_net.final_bn/ ds_net.relu
       
        # cam_extractor = CAM(model, fc_layer='left_postprocess_net.gn_conv_last')
        saliency_maps = model.saliency_map.data.cpu().numpy()

        #InfoCAM
        # pdb.set_trace()
        batch, channel, _, _ = last_feature_map.size()
        _, target = y_score.topk(1, 1, True, True)
        target = target.squeeze()
        #_, top_2_target = y_score.topk(200, 1, True, True)
        _, top_2_target = y_score.topk(2, 1, True, True)

        fc_weight = model.global_network.postprocess_module.gn_conv_last.weight
        # lfc_weight - eft_postprocess_net.gn_conv_last
        target_2 = top_2_target[:, -1]
        cam_weight = fc_weight[target]

        cam_weight_2 = fc_weight[target_2]
        cam_weight -= cam_weight_2

        cam_weight = cam_weight.view(batch, channel, 1, 1).expand_as(last_feature_map)
        cam_info = (cam_weight * last_feature_map)

        cam_filter = torch.ones(1, channel, 3, 3).to(target.device)
        cam_info = F.conv2d(cam_info, cam_filter, padding=2, stride=1)
        # cam_info = cam_info.mean(1).unsqueeze(1)

        
        patch_locations = model.patch_locations
        patch_imgs = model.patches
        # patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
        patch_attentions = model.patch_attns.data.cpu().numpy()
        batch_size = imgs.size()[0]
        for i in range(batch_size):
        # save_dir = os.path.join("visualization", "{0}.png".format(short_file_path))
            filename = test_filename[i]
            print(step, i)
            print(filename)
            # if "MERGED" in filename:
            #     continue
            #filename = filename[filename.find('/')+1:filename.find('.')]
            filename = filename.split('/')[-1]
            #save_dir = os.path.join(output_path, "{}.png".format(filename))
            save_dir = os.path.join(output_path, filename)
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
                    
            # visualize_example(imgs[i:i+1,:,:,:], saliency_maps[i:i+1,:,:,:], [benign_seg, malignant_seg],
            #         patch_locations[i:i+1,:,:], patch_imgs[i:i+1,:,:,:], patch_attentions[i],
            #         save_dir, parameters, mode)

            
                
            
            #img = np.array(Image.open(test_filename[i]), dtype=np.float32)
            img = Image.open(test_filename[i])
            
            gt_mask_path = test_filename[i].replace('full','merged_masks').replace("FULL","MASK_1")
            print(gt_mask_path)
            if exists(gt_mask_path) ==False:
                continue
                gt_mask_path = gt_mask_path.replace("1","MERGED")
            gt_mask = Image.open(gt_mask_path)
            #img = img.resize((2944, 1920))
            img = img.resize((1920, 2944))
            gt_mask = gt_mask.resize((1920, 2944))
            img = np.array(img, dtype=np.float32)
            gt_mask = np.array(gt_mask, dtype=np.float32)
            view = filename.rsplit('-', 2)[1].split('_')[2]
            # if view == 'R':
            #     img = np.fliplr(img)
            #     gt_mask = np.fliplr(gt_mask)
            # import pdb; pdb.set_trace()

            if "RIGHT" in test_filename[i]:  
                img = np.fliplr(img)
                gt_mask = np.fliplr(gt_mask)
            # visualize_example_git(imgs[i:i+1,:,:,:], saliency_maps[i:i+1,:,:,:], [benign_seg, malignant_seg],
            #         patch_locations[i:i+1,:,:], patch_imgs[i:i+1,:,:,:], patch_attentions[i],
            #         save_dir, parameters)

            
            ## CAM extractor
            alter_cam_malig = cam_extractor(1)[0][i:i+1,:,:]
            # alter_cam_malig = cam_extractor(1, scores=y_score)[0][i:i+1,:,:]  # ScoreCAM
            alter_cam_malig = alter_cam_malig.data.cpu().numpy()

            alter_cam_ben = cam_extractor(0)[0][i:i+1,:,:]
            alter_cam_ben = alter_cam_ben.data.cpu().numpy()

            # alter_cam_malig=None
            # alter_cam_ben=None

            ## INFOCAM ##
            info_cam_malig = cam_info[i].data.cpu().numpy()
            info_cam_ben = cam_info[i].data.cpu().numpy()
            # pdb.set_trace()

            ## GRADCAM EXTRACTOR ##
            gradcam_ben = gradcam__extractor(class_idx=0, scores=y_score)[0][i:i+1,:,:]
            # import pdb; pdb.set_trace()
            gradcam_ben = gradcam_ben.data.cpu().numpy()
            gradcam_malig = gradcam__extractor(class_idx=1, scores=y_score)[0][i:i+1,:,:]
            gradcam_malig = gradcam_malig.data.cpu().numpy()
            
            ## GRADCAMPP EXTRACTOR ##
            gradcampp_ben = gradcampp_extractor(class_idx=0, scores=y_score)[0][i:i+1,:,:]    
            gradcampp_ben = gradcampp_ben.data.cpu().numpy()
            gradcampp_malig = gradcampp_extractor(class_idx=1, scores=y_score)[0][i:i+1,:,:] 
            gradcampp_malig = gradcampp_malig.data.cpu().numpy()

            ## XGRADCAM EXTRACTOR
            xgradcam_ben = xgradcam_extractor(class_idx=0, scores=y_score)[0][i:i+1,:,:]
            xgradcam_ben = xgradcam_ben.data.cpu().numpy()
            xgradcam_malig = xgradcam_extractor(class_idx=1, scores=y_score)[0][i:i+1,:,:]
            xgradcam_malig = xgradcam_malig.data.cpu().numpy()


            # LAYER CAM EXTRACTOR
            # alter_gradcam_ben = gradcampp_extractor(class_idx=0, scores=y_score)[0][i:i+1,:,:]
            cams  = layercam_extractor(class_idx=0, scores=y_score)
            # import pdb; pdb.set_trace()
            layercam_ben = layercam_extractor.fuse_cams(cams)[i:i+1,:,:]
            layercam_ben = layercam_ben.data.cpu().numpy()

            cams  = layercam_extractor(class_idx=1, scores=y_score)
            layercam_malig = layercam_extractor.fuse_cams(cams)[i:i+1,:,:]
            layercam_malig= layercam_malig.data.cpu().numpy()
            

            #alter_gradcam_malig = gradcampp_extractor(class_idx=1, scores=y_global)[0][i:i+1,:,:]
            
            #alter_gradcam_malig = gradcampp_extractor(class_idx=1, scores=y_score)[0][i:i+1,:,:]
            # alter_gradcam_malig = gradcampp_extractor(class_idx=1, scores=y_score)
            # alter_gradcam_malig = extractor.fuse_cams(alter_gradcam_malig)
            # alter_gradcam_malig = alter_gradcam_malig.data.cpu().numpy()
            
            # import pdb; pdb.set_trace()
            # visualize_example_git(img, saliency_maps[i:i+1,:,:,:], [benign_seg, malignant_seg],
            #         patch_locations[i:i+1,:,:], patch_imgs[i:i+1,:,:,:], patch_attentions[i],
            #         save_dir, parameters)
            # import pdb; pdb.set_trace()
            
            alter_visualize_example_git(img, gt_mask, saliency_maps[i:i+1,:,:,:], [benign_seg, malignant_seg],
                    patch_locations[i:i+1,:,:], patch_imgs[i:i+1,:,:,:], patch_attentions[i],
                    save_dir, parameters, alter_cam_ben, alter_cam_malig, gradcam_ben,gradcam_malig, gradcampp_ben,gradcampp_malig,
                    xgradcam_ben, xgradcam_malig, layercam_ben, layercam_malig , info_cam_ben, info_cam_malig, y_fusion[i], labels[i])
       
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default='models/sample_model_1.p',
        help="pretrained model path")
    # ap.add_argument("--data_csv_path", type=str, default='data/cropped_mammo-16/test.csv',
    #     help="test data csv path")
    ap.add_argument("--data_path", type=str, default='data/cropped_mammo-16/test',
        help="test data path")
    ap.add_argument("--bs", type=int, default=6,
        help="batch size")
    ap.add_argument("--num_chan", type=int, default=3,
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
    ap.add_argument("--gpuid", type=int, default=0,
        help="gpu id")
    ap.add_argument("--v1_global", type=bool, default=False,
        help="use RN18 as v1_global")
    
    args = vars(ap.parse_args())

   
    # Test_data = args['data_csv_path']
    DATA_PATH = args['data_path']
    num_works = 4

    # device_type = 'gpu'
    device_type = 'cpu'
    gpu_id = args['gpuid']
    
    threshold = 0.5
    model_path = args['model_path']
    # beta = args['beta']
    beta = 3.259162430057801e-06
    percent_t = args['percent_t']

    img_size = args['is']
    batch_size = args['bs']
    # aug = args['aug']
    aug = False
    pretrained_model = args['pretrained_model']
    use_v1_global = args['v1_global']
        
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
        "use_v1_global":use_v1_global
    }

    if use_v1_global:
        parameters["cam_size"] = (92, 60)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # import pdb; pdb.set_trace()
    model = gmic.GMIC(parameters)
    if device_type == 'gpu' and torch.has_cudnn:
        device = torch.device("cuda:{}".format(gpu_id))
        if pretrained_model:
            model.load_state_dict(torch.load(model_path), strict=False)
        else:
            model.load_state_dict(torch.load(model_path)['model_state_dict'], strict=True)
            
        
    else:
        if pretrained_model:
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
        else:
            model.load_state_dict(torch.load((model_path)['model_state_dict'],  map_location="cpu"), strict=True)
        device = torch.device("cpu")
    model = model.to(device)
    # model = torch.nn.DataParallel(model)

    params = [p for p in model.parameters() if p.requires_grad]

    # test_loader = get_dataloader(os.path.join(DATA_PATH, 'test'), Test_data, image_size=img_size, batch_size=batch_size, shuffle=False, \
    #     max_value=max_value)
    #test_loader = get_dataloaderCBIS(os.path.join(DATA_PATH, 'test'), image_size=img_size, batch_size=batch_size, shuffle=False, max_value=max_value, aug=aug)
    test_loader = get_dataloaderCBIS(DATA_PATH, 'test_rnd', image_size=img_size, batch_size=batch_size, shuffle=False, max_value=max_value, aug=aug, num_chan=args['num_chan'])
    
    #just debugging
    # for step, (imgs, labels, test_filename) in enumerate(test_loader):
    #     debug_img = imgs[0]

    #     # filename = test_filename[i]
    #     #filename = filename[filename.find('/')+1:filename.find('.')]
    #     # filename = filename.split('/')[-1]
    #     #save_dir = os.path.join(output_path, "{}.png".format(filename))
    #     debug_img = debug_img.permute((1,2,0))
    #     # debug_img = debug_img / 2 + 0.5
    #     # import pdb; pdb.set_trace()
    #     # debug_img = debug_img.cpu().numpy()
    #     # # debug_img -= np.mean(debug_img)
    #     # debug_img = (debug_img *np.std(debug_img) ) + np.mean(debug_img)
    #     save_dir_debug = oxs.path.join(args['output_path'] , 'debug_img.png')

    #     plt.imshow(debug_img[:,:,0], aspect='equal', cmap='gray')
    #     # plt.imshow(debug_img)
    #     plt.savefig(save_dir_debug, bbox_inches='tight', format="png", dpi=500)
    #     plt.close()
    #     break

    
    # import pdb; pdb.set_trace()
    # test_filename = test_loader.dataset.lines
    
    # visuaize segmentation maps
    # if not args['mammo_sample_data']:
    #     mask_dir = "data/sample_data/segmentation"
    #     output_path = 'visualization'
    # else:
    #     mask_dir = args('mask_dir')
    #     output_path = args('output_path')

    # output_path = os.path.join(args['data_path'].split('/')[:-1])
   # output_path = args('output_path')
    output_path = args['output_path'] 


    if pretrained_model:
        #visualize_saliency_patch_maps(model, output_path, test_loader, device, test_filename, mask_dir)
        visualize_saliency_patch_maps(model = model, output_path=output_path, loader=test_loader, device=device, mask_dir=args['mask_dir'])
    else:
        #visualize_saliency_patch_maps(model, output_path, test_loader, device, test_filename, mask_dir)
        visualize_saliency_patch_maps(model = model, output_path=output_path, loader=test_loader, device=device, mask_dir=args['mask_dir'])
        