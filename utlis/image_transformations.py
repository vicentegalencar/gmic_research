# This files contains gamma/power transformation and inverse log transformation
# to convert bright breasts into dark breasts
import cv2, math
import numpy as np

def gamma_transform(img, gamma, max_intensity):
    '''
        Perform gamma transformation for contrast enhancement of bright breast
        gamma transformation
        s = cr^gamma, (gamma>1)
        r: original intensity
        s: transformed intensity
    Args:
        img: input image.
        gamma: parameter of gamma transformation.
        max_intensity: max grayscale intensity of input image
    Returns:
        An image with unit8 in which the low intensity part is squeezed while the high intensity part is expanded.
    NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
        which has a max value of 255.
    '''
    rows, cols = img.shape
    gamma_img = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            r = img[i,j]
            gamma_img[i,j] = math.pow(r/max_intensity, gamma)*max_intensity

    return gamma_img


def inverse_log_transform(img, max_intensity):
    rows, cols = img.shape
    inverse_log_img = np.zeros((rows, cols), dtype=np.float32)

    # create lUT for inverse log transform
    max_val = img.max()
    lut = np.zeros(max_intensity+1, dtype=np.uint8)
    for k in range(max_val):
        r = int(np.log(1+k)/np.log(1+max_val)*max_intensity)
        lut[r] = k

    inverse_log_img = cv2.LUT(img, lut)
    # for i in range(rows):
    #     for j in range(cols):
    #         r = img[i,j]
    #         inverse_log_img[i,j] = lut[r]

    cv2.normalize(inverse_log_img, inverse_log_img, 0, max_intensity, cv2.NORM_MINMAX)
    inverse_log_img = cv2.convertScaleAbs(inverse_log_img)
    return inverse_log_img