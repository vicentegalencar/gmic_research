# This file detects bright breasts in mammogram using image processing method
# It does not work as effectively as the bright mammo detection method implemented in read_Images.py
# The parameters (threshold) in this method should be tuned according to the input images
#=======================================================
# input: a 'png' or 'jpg' image
# output: whether the input image has bright breast
#=========================================================


import cv2, os
from utlis.crop_breast import suppress_artifacts, crop_max_bg

def predict_bright_breast(image):
    breast_region, breast_mask = suppress_artifacts(image)
    
    pix_num = 0
    pix_intensity_sum = 0
    breast_mean = 0
    bright_line = 0
    bright_pix_thresh = breast_region.mean() #this value should be tuned according to input images
    bright_pix_per_line_thresh = 0.3 #this value should be tuned according to input images
    bright_line_num_thresh = 0.8 #this value should be tuned according to input images
    mean_thresh = 0.3 * breast_region.max() #this value should be tuned according to input images
    
    # calculate mean intensity in breast region
    for i in range(breast_mask.shape[0]): # height
        bright_pix_num = 0
        line_pix_num = 0
        for j in range(breast_mask.shape[1]): # width
            if breast_mask[i,j] > 0:
                pix_intensity_sum+=breast_region[i,j]
                pix_num+=1
                line_pix_num+=1
                if breast_region[i,j] > bright_pix_thresh:
                    bright_pix_num+=1
        if bright_pix_num > int(bright_pix_per_line_thresh*line_pix_num):
            bright_line+=1

    breast_mean = pix_intensity_sum/pix_num
            
    if bright_line > int(bright_line_num_thresh*breast_mask.shape[0]) and breast_mean > mean_thresh:
        bright_breast = True
    else:
        bright_breast = False
        
    return bright_breast

#=================================================================
# test predict_bright_breast
#=================================================================
if __name__ == '__main__':
    image_dir = r'I:\MDPP\data\extra_white_images\train\cancer'
    image_names = os.listdir(image_dir)

    for image_name in image_names:
        image = cv2.imread(os.path.join(image_dir, image_name), cv2.IMREAD_UNCHANGED)
        bright_breast = predict_bright_breast(image)
        if bright_breast:
            print('{} is bright mammo'.format(image_name))
        else:
            print('{} is dark mammo'.format(image_name))