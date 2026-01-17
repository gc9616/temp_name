import sys
from pathlib import Path
import argparse

import cv2
import numpy as np
from skimage import morphology

# Pipeline works like this:
# 1. Load image as grayscale
# 2. Segment palm area
# 3. Correct illumination & exposure issues
# 4. (Optional) Enhance illumination in certain areas on correct img to make vessels more obvious
# 5. Single-scale vessel response with Gabor filter
# 6. Multi-scale vessel response to check for different size veins
# 7. Enhance feature map, send to post processing unit
# 8. Generate a red overlay to get good visual of what the pipeline sees as "veins"

def load_grayscale(path:str):
    """
    Loads image in grayscale, returns as a **UINT8 array**.
    
    :param path: image path
    :type path: str
    """
    bw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if bw is None:
        raise RuntimeError(f"check path: {path}")
    return bw

def segmentation_hand_mask(bw_img):
    """
    Segment the image of the hand so that we draw on an extract features from only the palm (here is calculated as simply the brightest component. Probably have to make this more sophisticated down the line).

    1) Blur + Otsu threshold
    2) Find biggest continuous shape
    3) Morpho cleanup
    4) Pad region with extra space to avoid background getting considered for features. 
    
    :param bw_img: grayscale image
    """

    smoothed = cv2.GaussianBlur(bw_img, (0, 0), 3.0)

    _, threshold = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentWithStats(threshold, connectivity=8)

    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hand_mask = (labels == largest).astype(np.uint8) * 255
    else:
        hand_mask = threshold.copy()

    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1
    )
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1
    )

    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    safe_mask = (dist > 6).astype(np.uint8) * 255
    
    return hand_mask, safe_mask

def illumination_correction(bw_img, safe_mask):
    """
    Make dark veins bright:

    1) Pass through large Gaussian blur to estimate background illumination
    2) 
    
    :param bw_img: Description
    :param safe_mask: Description
    """

    background = cv2.GuassianBlur(bw_img,(0,0), 35.0)

    high_pass = cv2.subtract(background, bw_img)
    high_pass = cv2.bitwise_and(high_pass, high_pass, mask=safe_mask)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # light CLAHE process:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    high_pass = clahe.apply(high_pass)

    high_pass = cv2.GaussianBlur(high_pass, (0, 0), 1.2)
    
    return high_pass