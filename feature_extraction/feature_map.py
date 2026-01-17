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


def highlight_enhance(img, safe_mask, strength=1.5, threshold_percentile=75):
    """
    Strengthens any bright patches, highlights on raw features. Goal is to make vessels more visible and pop out:

    brighter = brighter + (brighter - threshold) * (strength - 1.0) * highlight_mask
    
    :param img: Input image (uint8)
    :param safe_mask: Safe area to work on for hand
    :param strength: multiplier for highlight regions
    :param threshold_percentile: define what gets highlighted (percentile)
    """

    # plz no type mismatch plz plz plz
    img_float = img.astype(np.float32)

    # these are our values of interest
    vals = img[safe_mask > 0]
    threshold = np.percentile(vals, threshold_percentile)

    highlight_mask = (img >= threshold).astype(np.float32)

    enhanced = img_float.copy() + (img_float.copy() - threshold) * (strength - 1) * highlight_mask

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced = cv2.bitwise_and(enhanced, enhanced, safe_mask)

    return enhanced


def enhance_contrast(img, safe_mask, alpha=1.2, beta=0):
    """
    Contrast filter
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        alpha: Contrast control (1.0 = no change, >1.0 = more contrast)
        beta: Brightness control
    
    Returns:
        Contrast-enhanced image
    """
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=safe_mask)
    return enhanced


def strengthen_shadows(img, safe_mask, strength=1.3, threshold_percentile=25):
    """
    Shadow filter - simulates turning up shadow knob on photo editing software
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        strength: Multiplier for shadow regions (default 1.3)
        threshold_percentile: Percentile to define "shadow" regions (default 25)
    
    Returns:
        Enhanced image with stronger shadows
    """
    img_float = img.astype(np.float32)
    
    # Find shadow threshold
    vals = img[safe_mask > 0]
    threshold = np.percentile(vals, threshold_percentile)
    
    shadow_mask = (img <= threshold).astype(np.float32)
    
    # Strengthen shadows: darken dark regions
    enhanced = img_float.copy()
    enhanced = enhanced - (threshold - enhanced) * (strength - 1.0) * shadow_mask
    
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=safe_mask)
    
    return enhanced


def sharpen_image(img, safe_mask, strength=1.5, sigma=1.0):
    """
    Sharpen image using unsharp masking.
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        strength: Sharpening strength (default 1.5)
        sigma: Gaussian blur sigma for unsharp mask (default 1.0)
    
    Returns:
        Sharpened image
    """
    # get low signals
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # emphasize high signals by adding the original minus low signals. 
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    sharpened = img_float + (img_float - blurred_float) * strength
    
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    sharpened = cv2.bitwise_and(sharpened, sharpened, mask=safe_mask)
    
    return sharpened

def reduce_noise_bilateral(img, safe_mask, d=5, sigma_color=50, sigma_space=50):
    """
    Reduce noise using bilateral filter (presrve edges)
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        d: Diameter of pixel neighborhood (default 5)
        sigma_color: Filter sigma in color space (default 50)
        sigma_space: Filter sigma in coordinate space (default 50)
    
    Returns:
        Denoised image
    """
    denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    denoised = cv2.bitwise_and(denoised, denoised, mask=safe_mask)
    return denoised


def reduce_noise_median(img, safe_mask, ksize=3):
    """
    Reduce flaky noise filter.
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        ksize: Kernel size (must be odd, default 3)
    
    Returns:
        Denoised image
    """
    denoised = cv2.medianBlur(img, ksize)
    denoised = cv2.bitwise_and(denoised, denoised, mask=safe_mask)
    return denoised


def reduce_noise_nlm(img, safe_mask, h=10, template_window_size=7, search_window_size=21):
    """
    Reduce noise using non-local means denoising (stronger but slower).
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        h: Filter strength (default 10, higher = stronger denoising)
        template_window_size: Size of template patch (default 7)
        search_window_size: Size of search area (default 21)
    
    Returns:
        Denoised image
    """
    denoised = cv2.fastNlMeansDenoising(img, None, h=h, 
                                        templateWindowSize=template_window_size,
                                        searchWindowSize=search_window_size)
    denoised = cv2.bitwise_and(denoised, denoised, mask=safe_mask)
    return denoised


def remove_small_noise_morph(img, safe_mask, min_size=5):
    """
    Remove small noise blobs using morpho.
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        min_size: Minimum blob size to keep (default 5)
    
    Returns:
        Cleaned image
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=safe_mask)
    
    # Remove small objects
    binary_clean = morphology.remove_small_objects(binary.astype(bool), max_size=min_size - 1)
    binary_clean = (binary_clean.astype(np.uint8) * 255)
    
    cleaned = cv2.bitwise_and(img, img, mask=binary_clean)
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=safe_mask)
    
    return cleaned

def gabor_bank(img_u8, ksize, sig, lambd, gamma=0.5, psi=0.0, ntheta=12):
    """
    Gabor filter bank for vessel/ridge response. Baisc idea is that veins look like Gaussians, we exploit that an try to match a Guassian template to things that look like vessels at n different angles.
    
    Parameters:
    - ksize: kernel size
    - sig: sigma (standard deviation)
    - lambd: wavelength
    - gamma: spatial aspect ratio
    - psi: phase offset
    - ntheta: number of orientations
    
    Returns max response over all orientations.
    """
    thetas = np.linspace(0, np.pi, ntheta, endpoint=False)
    resp_max = np.zeros_like(img_u8, dtype=np.float32)
    src = img_u8.astype(np.float32)
    
    for th in thetas:
        kern = cv2.getGaborKernel((ksize, ksize), sig, th, lambd, gamma, psi, ktype=cv2.CV_32F)
        kern -= kern.mean()  # IMPORTANT: zero-mean kernel
        r = cv2.filter2D(src, cv2.CV_32F, kern)
        r = np.maximum(r, 0)  # keep the max ridge-like (veins) positive response
        resp_max = np.maximum(resp_max, r)
    
    return resp_max

def vessel_response_single_scale(high_pass, safe_mask, ksize=31, sig=5.0, lambd=10.0, gamma=0.5, ntheta=12):
    """
    Single-scale vessel response (baseline).
    
    Default params:
    - ksize = 31
    - sigma = 5.0
    - lambda = 10.0
    - gamma = 0.5
    - psi = 0
    - num_orientations = 12
    """
    resp_max = gabor_bank(high_pass, ksize, sig, lambd, gamma, psi=0.0, ntheta=ntheta)
    
    # Normalize to uint8
    resp_u8 = cv2.normalize(resp_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=safe_mask)
    
    return resp_u8, resp_max


def vessel_response_multiscale(high_pass, safe_mask):
    """
    B2) Multi-scale vessel response (captures thin + thick veins).
    
    Runs two Gabor banks:
    - Fine scale (thin vessels): ksize=31, sigma=5, lambda=10
    - Coarse scale (thicker vessels): ksize=51, sigma=8, lambda=18
    
    Returns normalized and maxed response.
    """
    # get the tiny vessels
    resp_fine = gabor_bank(high_pass, 31, 5.0, 10.0, gamma=0.5, ntheta=12)
    
    # BIG
    resp_coarse = gabor_bank(high_pass, 51, 8.0, 18.0, gamma=0.7, ntheta=10)
    
    # Normalize each to [0,1] then max
    rf_min, rf_max = resp_fine.min(), resp_fine.max()
    rc_min, rc_max = resp_coarse.min(), resp_coarse.max()
    
    rf = (resp_fine - rf_min) / (rf_max - rf_min + 1e-9)
    rc = (resp_coarse - rc_min) / (rc_max - rc_min + 1e-9)
    
    resp = np.maximum(rf, rc)
    
    # Convert to uint8
    resp_u8 = (resp * 255).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=safe_mask)
    
    return resp_u8, resp


def enhance_vessel_map(resp_u8, safe_mask, gamma=0.5, p_low=1, p_high=99):
    """
    Normalize within hand + apply gamma (makes vessels more obvious).
    
    Parameters:
    - gamma: 0.5 (brighten midrange vessel responses)
    - p_low, p_high: percentile clip (1, 99)
    
    This makes the vessel-enhanced response pop more while preserving structure.
    """
    # Robust normalization inside the hand
    vals = resp_u8[safe_mask > 0].astype(np.float32)
    lo, hi = np.percentile(vals, p_low), np.percentile(vals, p_high)
    x = np.clip(resp_u8.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo + 1e-9)  # [0,1]
    
    # Gamma to emphasize vessels
    x = np.power(x, gamma)
    
    # Back to uint8
    enhanced = (x * 255.0).astype(np.uint8)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=safe_mask)
    
    return enhanced


def create_red_overlay(gray, resp_u8, safe_mask, percentile_thresh=92.5, min_size=140):
    """
    A5) Red overlay tracing (debug/visualization only).
    
    Parameters:
    - percentile_thresh: typical 92-96 (default 92.5)
    - min_size: minimum object size for removal (default 140)
    """
    vals = resp_u8[safe_mask > 0]
    t = np.percentile(vals, percentile_thresh)
    binv = (resp_u8 >= t) & (safe_mask > 0)
    
    # Remove small objects
    # Note: newer scikit-image uses max_size (removes objects <= max_size)
    # So to remove objects < min_size, we use max_size = min_size - 1
    binv = morphology.remove_small_objects(binv.astype(bool), max_size=min_size - 1)
    bin_u8 = (binv.astype(np.uint8) * 255)
    
    # Morphological close
    bin_u8 = cv2.morphologyEx(
        bin_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    
    # Skeletonize
    sk = morphology.skeletonize(bin_u8 > 0).astype(np.uint8) * 255
    sk = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Create red overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[sk > 0] = (0, 0, 255)
    
    return overlay, sk


