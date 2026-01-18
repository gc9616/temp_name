#!/usr/bin/env python3
"""
Palm Vein Feature Map Extraction Pipeline

Implements a complete pipeline for extracting vessel-enhanced feature maps from palm images.
Based on Gabor filter banks, hand segmentation, and illumination correction.

Outputs:
1. Enhanced vessel feature map (final product for feature extraction)
2. Red overlay visualization (debug/visualization)
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects


def read_image_grayscale(path):
    """A1) Read image as grayscale."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {path}")
    return gray


def preprocess_image(gray, highlight_strength=1.3, contrast_alpha=1.4, 
                     shadow_strength=1.2, sharpen_strength=1.5, sharpen_sigma=0.8,
                     denoise=True, denoise_strength='light'):
    """
    Preprocess image BEFORE segmentation to improve hand detection.
    
    This applies enhancement filters to the raw grayscale image to:
    1. Reduce noise that can confuse segmentation
    2. Increase contrast between hand and background
    3. Strengthen edges for better boundary detection
    
    Args:
        gray: Raw grayscale image
        highlight_strength: Tone down highlights (multiply bright areas, default 1.3)
        contrast_alpha: Contrast enhancement (default 1.4)
        shadow_strength: Strengthen shadows for better separation (default 1.2)
        sharpen_strength: Sharpen to enhance edges (default 1.5)
        sharpen_sigma: Sharpening blur sigma (default 0.8)
        denoise: Whether to apply initial denoising (default True)
        denoise_strength: Denoising strength (default 'light')
    
    Returns:
        Preprocessed grayscale image (uint8)
    """
    preprocessed = gray.copy()
    
    # Create a temporary full mask for preprocessing (before we have real mask)
    full_mask = np.ones_like(gray, dtype=np.uint8) * 255
    
    # 1. Initial denoising to reduce noise
    if denoise:
        if denoise_strength == 'light':
            preprocessed = cv2.bilateralFilter(preprocessed, 5, 50, 50)
        elif denoise_strength == 'medium':
            preprocessed = cv2.bilateralFilter(preprocessed, 5, 75, 75)
        else:  # strong
            preprocessed = cv2.bilateralFilter(preprocessed, 7, 100, 100)
    
    # 2. Enhance contrast (helps separate hand from background)
    preprocessed = cv2.convertScaleAbs(preprocessed, alpha=contrast_alpha, beta=0)
    
    # 3. Strengthen highlights (tone down very bright areas)
    img_float = preprocessed.astype(np.float32)
    threshold = np.percentile(preprocessed, 75)
    highlight_mask = (preprocessed >= threshold).astype(np.float32)
    # For preprocessing, we want to REDUCE highlights (divide rather than multiply)
    # to make the hand-background boundary more uniform
    reduction_factor = 1.0 / highlight_strength if highlight_strength > 1.0 else 1.0
    img_float = img_float * (1.0 - highlight_mask * (1.0 - reduction_factor))
    preprocessed = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # 4. Strengthen shadows (darken dark areas for better contrast)
    img_float = preprocessed.astype(np.float32)
    shadow_threshold = np.percentile(preprocessed, 25)
    shadow_mask = (preprocessed <= shadow_threshold).astype(np.float32)
    img_float = img_float - (shadow_threshold - img_float) * (shadow_strength - 1.0) * shadow_mask
    preprocessed = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # 5. Sharpen to enhance edges (helps gradient-based detection)
    blurred = cv2.GaussianBlur(preprocessed, (0, 0), sharpen_sigma)
    img_float = preprocessed.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    sharpened = img_float + (img_float - blurred_float) * sharpen_strength
    preprocessed = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return preprocessed


def create_hand_segmentation_mask_adaptive(gray, block_size=101, c_offset=-15,
                                           morph_close_size=35, morph_open_size=9,
                                           safe_distance=15, otsu_bias=0.75):
    """
    Hand segmentation using adaptive thresholding with border exclusion.
    
    More robust to uneven lighting than global Otsu thresholding.
    Excludes regions connected to image borders (background detection).
    
    Args:
        gray: Grayscale input image
        block_size: Size of neighborhood for adaptive threshold (default 101, must be odd)
        c_offset: Constant subtracted from mean (negative = more lenient, default -15)
        morph_close_size: Morphological closing kernel size (default 35)
        morph_open_size: Morphological opening kernel size (default 9)
        safe_distance: Distance transform threshold for safe mask (default 15, larger to avoid edges)
        otsu_bias: Bias for initial Otsu threshold (default 0.75)
    
    Returns:
        hand_mask: Binary mask of hand region
        safe_mask: Eroded mask to avoid boundary artifacts
    """
    h, w = gray.shape
    
    # 1. Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # 2. Get Otsu mask
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, rough_mask = cv2.threshold(blur, int(otsu_thresh * otsu_bias), 255, cv2.THRESH_BINARY)
    
    # 3. CRITICAL: Exclude border-connected regions (background detection)
    # Mark all pixels connected to image border as background
    border_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    rough_mask_copy = rough_mask.copy()
    
    # Flood fill from all border pixels
    for x in range(w):
        if rough_mask_copy[0, x] == 255:
            cv2.floodFill(rough_mask_copy, border_mask, (x, 0), 128)
        if rough_mask_copy[h-1, x] == 255:
            cv2.floodFill(rough_mask_copy, border_mask, (x, h-1), 128)
    for y in range(h):
        if rough_mask_copy[y, 0] == 255:
            cv2.floodFill(rough_mask_copy, border_mask, (0, y), 128)
        if rough_mask_copy[y, w-1] == 255:
            cv2.floodFill(rough_mask_copy, border_mask, (w-1, y), 128)
    
    # Keep only non-border-connected regions
    hand_mask = (rough_mask_copy == 255).astype(np.uint8) * 255
    
    # 4. If no interior regions found, fall back to largest component
    if cv2.countNonZero(hand_mask) < h * w * 0.05:  # Less than 5% of image
        # Fall back: use largest component from original mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rough_mask, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hand_mask = (labels == largest).astype(np.uint8) * 255
    
    # 5. Find largest connected component (in case multiple interior regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hand_mask, connectivity=8)
    if num_labels > 1:
        # Prefer component closest to center
        center_y, center_x = h // 2, w // 2
        best_label = 1
        best_score = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            # Score: larger area and closer to center is better
            score = area / (dist_to_center + 1)
            if score > best_score:
                best_score = score
                best_label = i
        
        hand_mask = (labels == best_label).astype(np.uint8) * 255
    
    # 6. Morphological closing to fill small gaps (but not too aggressive)
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)),
        iterations=2
    )
    
    # 7. Fill interior holes
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    hand_mask_inv = cv2.bitwise_not(hand_mask)
    
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if hand_mask_inv[seed[1], seed[0]] == 255:
            cv2.floodFill(hand_mask_inv, flood_mask, seed, 128)
    
    interior_holes = (hand_mask_inv == 255).astype(np.uint8) * 255
    hand_mask = cv2.bitwise_or(hand_mask, interior_holes)
    
    # 8. Opening to smooth edges
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)),
        iterations=1
    )
    
    # 9. Safe mask via distance transform (larger distance to avoid edge artifacts)
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    safe_mask = (dist > safe_distance).astype(np.uint8) * 255
    
    # 10. CRITICAL: Force safe mask to never touch image borders (prevents edge artifacts)
    border_margin = 20  # pixels from image edge to exclude
    safe_mask[:border_margin, :] = 0  # top
    safe_mask[-border_margin:, :] = 0  # bottom
    safe_mask[:, :border_margin] = 0  # left
    safe_mask[:, -border_margin:] = 0  # right
    
    return hand_mask, safe_mask


def create_hand_segmentation_mask_hybrid(gray, otsu_bias=0.80, canny_low=30, canny_high=100,
                                         morph_close_size=35, morph_open_size=9,
                                         safe_distance=15):
    """
    Hybrid hand segmentation: Otsu with border exclusion and gap-filling.
    
    Uses Otsu with reduced strictness, excludes border-connected regions,
    then morphological operations to fill gaps. Best for problematic palm positioning.
    
    Args:
        gray: Grayscale input image
        otsu_bias: Bias for Otsu threshold (< 1.0 = less strict, default 0.80)
        canny_low: Canny low threshold (unused, kept for API compatibility)
        canny_high: Canny high threshold (unused, kept for API compatibility)
        morph_close_size: Morphological closing kernel size (default 35)
        morph_open_size: Morphological opening kernel size (default 9)
        safe_distance: Distance transform threshold for safe mask (default 15, larger for edge avoidance)
    
    Returns:
        hand_mask: Binary mask of hand region
        safe_mask: Eroded mask to avoid boundary artifacts
    """
    h, w = gray.shape
    
    # 1. Get Otsu mask with reduced strictness
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = int(otsu_thresh * otsu_bias)
    _, thr = cv2.threshold(blur, adjusted_thresh, 255, cv2.THRESH_BINARY)
    
    # 2. CRITICAL: Exclude border-connected regions (background detection)
    border_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    thr_copy = thr.copy()
    
    for x in range(w):
        if thr_copy[0, x] == 255:
            cv2.floodFill(thr_copy, border_mask, (x, 0), 128)
        if thr_copy[h-1, x] == 255:
            cv2.floodFill(thr_copy, border_mask, (x, h-1), 128)
    for y in range(h):
        if thr_copy[y, 0] == 255:
            cv2.floodFill(thr_copy, border_mask, (0, y), 128)
        if thr_copy[y, w-1] == 255:
            cv2.floodFill(thr_copy, border_mask, (w-1, y), 128)
    
    # Keep only non-border-connected regions
    hand_mask = (thr_copy == 255).astype(np.uint8) * 255
    
    # 3. If no interior regions found, fall back to largest component
    if cv2.countNonZero(hand_mask) < h * w * 0.05:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hand_mask = (labels == largest).astype(np.uint8) * 255
    
    # 4. Find best connected component (center-biased)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hand_mask, connectivity=8)
    if num_labels > 1:
        center_y, center_x = h // 2, w // 2
        best_label = 1
        best_score = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            score = area / (dist_to_center + 1)
            if score > best_score:
                best_score = score
                best_label = i
        
        hand_mask = (labels == best_label).astype(np.uint8) * 255
    
    # 5. Morphological closing to fill gaps
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)),
        iterations=2
    )
    
    # 6. Fill interior holes
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    hand_mask_inv = cv2.bitwise_not(hand_mask)
    
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if hand_mask_inv[seed[1], seed[0]] == 255:
            cv2.floodFill(hand_mask_inv, flood_mask, seed, 128)
    
    interior_holes = (hand_mask_inv == 255).astype(np.uint8) * 255
    hand_mask = cv2.bitwise_or(hand_mask, interior_holes)
    
    # 7. Opening to remove small protrusions/noise
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)),
        iterations=1
    )
    
    # 8. Safe mask via distance transform (larger distance for edge artifact avoidance)
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    safe_mask = (dist > safe_distance).astype(np.uint8) * 255
    
    # 9. CRITICAL: Force safe mask to never touch image borders (prevents edge artifacts)
    border_margin = 20  # pixels from image edge to exclude
    safe_mask[:border_margin, :] = 0  # top
    safe_mask[-border_margin:, :] = 0  # bottom
    safe_mask[:, :border_margin] = 0  # left
    safe_mask[:, -border_margin:] = 0  # right
    
    return hand_mask, safe_mask


def create_hand_segmentation_mask_otsu(gray, blur_sigma=3.0, otsu_bias=0.85,
                                       morph_close_size=35, morph_open_size=9,
                                       safe_distance=15):
    """
    Hand segmentation mask using Otsu thresholding with border exclusion.
    
    Steps:
    1. Blur + Otsu thresholding (with bias to reduce strictness)
    2. Exclude border-connected regions (background detection)
    3. Select best component (center-biased)
    4. Morphological cleanup
    5. Safe mask via distance transform (larger erosion for edge artifact avoidance)
    
    Args:
        gray: Grayscale input image
        blur_sigma: Gaussian blur sigma before thresholding (default 3.0)
        otsu_bias: Multiplier for Otsu threshold (< 1.0 = less strict, default 0.85)
        morph_close_size: Morphological closing kernel size (default 35)
        morph_open_size: Morphological opening kernel size (default 9)
        safe_distance: Distance transform threshold for safe mask (default 15, larger for edge avoidance)
    
    Returns:
        hand_mask: Binary mask of hand region
        safe_mask: Eroded mask to avoid boundary artifacts
    """
    h, w = gray.shape
    
    # Blur
    blur = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    
    # Get Otsu threshold value, then apply bias
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_thresh = int(otsu_thresh * otsu_bias)
    _, thr = cv2.threshold(blur, adjusted_thresh, 255, cv2.THRESH_BINARY)
    
    # CRITICAL: Exclude border-connected regions (background detection)
    border_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    thr_copy = thr.copy()
    
    # Flood fill from all border pixels
    for x in range(w):
        if thr_copy[0, x] == 255:
            cv2.floodFill(thr_copy, border_mask, (x, 0), 128)
        if thr_copy[h-1, x] == 255:
            cv2.floodFill(thr_copy, border_mask, (x, h-1), 128)
    for y in range(h):
        if thr_copy[y, 0] == 255:
            cv2.floodFill(thr_copy, border_mask, (0, y), 128)
        if thr_copy[y, w-1] == 255:
            cv2.floodFill(thr_copy, border_mask, (w-1, y), 128)
    
    # Keep only non-border-connected regions
    hand_mask = (thr_copy == 255).astype(np.uint8) * 255
    
    # If no interior regions found, fall back to largest component
    if cv2.countNonZero(hand_mask) < h * w * 0.05:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
        if num_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hand_mask = (labels == largest).astype(np.uint8) * 255
    
    # Find best connected component (center-biased)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hand_mask, connectivity=8)
    if num_labels > 1:
        center_y, center_x = h // 2, w // 2
        best_label = 1
        best_score = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            score = area / (dist_to_center + 1)
            if score > best_score:
                best_score = score
                best_label = i
        
        hand_mask = (labels == best_label).astype(np.uint8) * 255
    
    # Morph cleanup - close gaps
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_size, morph_close_size)),
        iterations=2
    )
    
    # Fill interior holes
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    hand_mask_inv = cv2.bitwise_not(hand_mask)
    
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if hand_mask_inv[seed[1], seed[0]] == 255:
            cv2.floodFill(hand_mask_inv, flood_mask, seed, 128)
    
    interior_holes = (hand_mask_inv == 255).astype(np.uint8) * 255
    hand_mask = cv2.bitwise_or(hand_mask, interior_holes)
    
    # Remove small protrusions
    hand_mask = cv2.morphologyEx(
        hand_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_size, morph_open_size)),
        iterations=1
    )
    
    # Safe mask via distance transform (larger distance for edge artifact avoidance)
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    safe_mask = (dist > safe_distance).astype(np.uint8) * 255
    
    # CRITICAL: Force safe mask to never touch image borders (prevents edge artifacts)
    border_margin = 20  # pixels from image edge to exclude
    safe_mask[:border_margin, :] = 0  # top
    safe_mask[-border_margin:, :] = 0  # bottom
    safe_mask[:, :border_margin] = 0  # left
    safe_mask[:, -border_margin:] = 0  # right
    
    return hand_mask, safe_mask


def exclude_fingers(hand_mask, finger_width_threshold=60, min_palm_area_ratio=0.20):
    """
    Remove finger regions from hand mask to keep only palm area.
    
    Uses a combination of:
    1. Convex hull to find the bounding shape
    2. Convexity defects to identify finger valleys
    3. Morphological erosion to shrink to palm core
    
    The kernel size automatically scales with image size for consistent results.
    
    Args:
        hand_mask: Binary hand mask (uint8)
        finger_width_threshold: Base width of fingers to remove (pixels, default 60)
                               Will be scaled based on image size.
        min_palm_area_ratio: Min ratio of palm area to original mask (default 0.20)
    
    Returns:
        palm_mask: Hand mask with fingers removed
    """
    h, w = hand_mask.shape
    original_area = cv2.countNonZero(hand_mask)
    
    if original_area == 0:
        return hand_mask
    
    # Scale based on image size (assume 600px as baseline)
    scale_factor = max(w, h) / 600.0
    
    # Method 1: Use large erosion to shrink to palm core, then dilate back
    # Fingers are thin protrusions that get eroded away
    erode_size = int(80 * scale_factor)  # More aggressive erosion
    if erode_size % 2 == 0:
        erode_size += 1
    erode_size = max(31, min(401, erode_size))
    
    print(f"  Finger exclusion: image {w}x{h}, scale={scale_factor:.2f}, erode_size={erode_size}px")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    
    # Heavy erosion to get palm core
    palm_core = cv2.erode(hand_mask, kernel, iterations=1)
    
    # Find the bounding rectangle of the palm core
    contours, _ = cv2.findContours(palm_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback: use original mask
        print(f"  Warning: Erosion removed everything, using original mask")
        return hand_mask
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create convex hull of the palm core - this gives a smooth palm shape
    hull = cv2.convexHull(largest_contour)
    
    # Create palm mask from convex hull
    palm_mask = np.zeros_like(hand_mask)
    cv2.drawContours(palm_mask, [hull], -1, 255, -1)
    
    # Intersect with original mask to keep only actual hand pixels
    palm_mask = cv2.bitwise_and(palm_mask, hand_mask)
    
    # Small opening to smooth edges
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    palm_mask = cv2.morphologyEx(palm_mask, cv2.MORPH_OPEN, small_kernel)
    
    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(palm_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        palm_mask = (labels == largest).astype(np.uint8) * 255
    
    # Verify we didn't remove too much
    palm_area = cv2.countNonZero(palm_mask)
    if palm_area < original_area * min_palm_area_ratio:
        print(f"  Warning: Finger exclusion removed too much ({100*palm_area/original_area:.0f}% remaining), using original mask")
        return hand_mask
    
    print(f"  Finger exclusion: kept {100*palm_area/original_area:.0f}% of original mask")
    return palm_mask


def create_hand_segmentation_mask(gray, method='hybrid', preprocess=True,
                                  preprocess_contrast=1.4, preprocess_sharpen=1.5,
                                  otsu_bias=0.85, canny_low=30, canny_high=100,
                                  exclude_fingers_flag=True, finger_width=60):
    """
    Create hand segmentation mask using specified method.
    
    This is the main entry point for hand segmentation. It optionally preprocesses
    the image first, then applies the chosen segmentation method.
    
    Available methods:
    - 'hybrid': Combines Otsu + edge-based filling (RECOMMENDED - most robust)
    - 'adaptive': Uses adaptive thresholding (good for uneven lighting)
    - 'otsu': Uses Otsu thresholding with bias (faster, less robust)
    
    Args:
        gray: Grayscale input image
        method: 'hybrid' (default), 'adaptive', or 'otsu'
        preprocess: Whether to preprocess image before segmentation (default True)
        preprocess_contrast: Contrast alpha for preprocessing (default 1.4)
        preprocess_sharpen: Sharpening strength for preprocessing (default 1.5)
        otsu_bias: Bias for Otsu threshold (< 1.0 = less strict, default 0.85)
        canny_low: Low threshold for Canny edge detection (default 30)
        canny_high: High threshold for Canny edge detection (default 100)
        exclude_fingers_flag: Whether to remove finger regions from mask (default True)
        finger_width: Max finger width in pixels for exclusion (default 60)
    
    Returns:
        hand_mask: Binary mask of hand region
        safe_mask: Eroded mask to avoid boundary artifacts
    """
    # Optionally preprocess the image to improve segmentation
    if preprocess:
        gray_for_seg = preprocess_image(
            gray, 
            contrast_alpha=preprocess_contrast,
            sharpen_strength=preprocess_sharpen,
            denoise=True,
            denoise_strength='light'
        )
    else:
        gray_for_seg = gray
    
    # Apply chosen segmentation method
    if method == 'hybrid':
        hand_mask, safe_mask = create_hand_segmentation_mask_hybrid(
            gray_for_seg,
            otsu_bias=otsu_bias,
            canny_low=canny_low,
            canny_high=canny_high
        )
    elif method == 'adaptive':
        hand_mask, safe_mask = create_hand_segmentation_mask_adaptive(gray_for_seg)
    else:  # 'otsu'
        hand_mask, safe_mask = create_hand_segmentation_mask_otsu(
            gray_for_seg, 
            otsu_bias=otsu_bias
        )
    
    # Optionally exclude fingers from the mask
    if exclude_fingers_flag:
        hand_mask = exclude_fingers(hand_mask, finger_width_threshold=finger_width)
        
        # Recalculate safe mask after finger exclusion
        h, w = gray.shape
        dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
        safe_mask = (dist > 15).astype(np.uint8) * 255
        
        # Apply border margin
        border_margin = 20
        safe_mask[:border_margin, :] = 0
        safe_mask[-border_margin:, :] = 0
        safe_mask[:, :border_margin] = 0
        safe_mask[:, -border_margin:] = 0
    
    return hand_mask, safe_mask


def illumination_correction(gray, safe_mask, clahe_clip=3.5, clahe_grid=6, bg_sigma=30.0):
    """
    A3) Illumination correction (veins dark → become bright).
    
    Parameters:
    - background blur sigma: controls how much background is smoothed (default 30.0)
    - CLAHE: Contrast Limited Adaptive Histogram Equalization
      - clipLimit: higher = more contrast, more sensitive to features (default 3.5)
      - tileGridSize: smaller = more local contrast (default 6)
    - post blur sigma = 1.0 (slightly sharper)
    
    Args:
        gray: Input grayscale image
        safe_mask: Safe mask for hand region
        clahe_clip: CLAHE clip limit (higher = more sensitive, default 3.5)
        clahe_grid: CLAHE tile grid size (smaller = more local, default 6)
        bg_sigma: Background blur sigma (default 30.0)
    
    Returns:
        hp: Illumination-corrected image with enhanced veins
    """
    # Background blur (slightly less blur to preserve more detail)
    bg = cv2.GaussianBlur(gray, (0, 0), bg_sigma)
    
    # High-pass: dark veins -> brighter
    hp = cv2.subtract(bg, gray)
    hp = cv2.bitwise_and(hp, hp, mask=safe_mask)
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE with higher clip limit for more sensitivity
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    hp = clahe.apply(hp)
    
    # Apply CLAHE again for even more enhancement
    hp = clahe.apply(hp)
    
    # Slightly less blur to preserve sharp features
    hp = cv2.GaussianBlur(hp, (0, 0), 1.0)
    
    return hp


def robust_normalize_u8(img, safe_mask, p_low=2, p_high=98):
    """
    Robust percentile-based normalization to uint8.
    
    Args:
        img: Input image (uint8)
        safe_mask: Mask for valid pixels
        p_low, p_high: Percentile bounds (default 2, 98)
    
    Returns:
        Normalized uint8 image
    """
    vals = img[safe_mask > 0].astype(np.float32)
    lo, hi = np.percentile(vals, p_low), np.percentile(vals, p_high)
    x = np.clip(img.astype(np.float32), lo, hi)
    x = (x - lo) / (hi - lo + 1e-9)  # [0,1]
    return (x * 255.0).astype(np.uint8)


def illumination_correction_for_features(gray, safe_mask, bg_sigma=30.0, p_low=2, p_high=98):
    """
    Simplified illumination correction for feature extraction (no CLAHE).
    
    High-pass + robust normalize only. Avoids CLAHE sensitivity for better
    photometric invariance in descriptors.
    
    Args:
        gray: Input grayscale image
        safe_mask: Safe mask for hand region
        bg_sigma: Background blur sigma (default 30.0)
        p_low, p_high: Percentile bounds for normalization (default 2, 98)
    
    Returns:
        hp: Illumination-corrected image (uint8)
    """
    bg = cv2.GaussianBlur(gray, (0, 0), bg_sigma)
    hp = cv2.subtract(bg, gray)
    hp = cv2.bitwise_and(hp, hp, mask=safe_mask)
    return robust_normalize_u8(hp, safe_mask, p_low=p_low, p_high=p_high)


def strengthen_highlights(img, safe_mask, strength=1.5, threshold_percentile=75):
    """
    Strengthen highlights (bright regions) to make vessels pop more.
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        strength: Multiplier for highlight regions (default 1.5)
        threshold_percentile: Percentile to define "highlight" regions (default 75)
    
    Returns:
        Enhanced image with stronger highlights
    """
    img_float = img.astype(np.float32)
    
    # Find highlight threshold
    vals = img[safe_mask > 0]
    threshold = np.percentile(vals, threshold_percentile)
    
    # Create highlight mask (bright regions)
    highlight_mask = (img >= threshold).astype(np.float32)
    
    # Strengthen highlights: boost bright regions
    enhanced = img_float.copy()
    enhanced = enhanced + (enhanced - threshold) * (strength - 1.0) * highlight_mask
    
    # Clip and convert back
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=safe_mask)
    
    return enhanced


def enhance_contrast(img, safe_mask, alpha=1.2, beta=0):
    """
    Enhance contrast using linear transformation.
    
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
    Strengthen shadows (dark regions) to enhance contrast.
    
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
    
    # Create shadow mask (dark regions)
    shadow_mask = (img <= threshold).astype(np.float32)
    
    # Strengthen shadows: darken dark regions
    enhanced = img_float.copy()
    enhanced = enhanced - (threshold - enhanced) * (strength - 1.0) * shadow_mask
    
    # Clip and convert back
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
    # Create unsharp mask
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    
    # Unsharp masking: original + (original - blurred) * strength
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    sharpened = img_float + (img_float - blurred_float) * strength
    
    # Clip and convert back
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    sharpened = cv2.bitwise_and(sharpened, sharpened, mask=safe_mask)
    
    return sharpened


def reduce_noise_bilateral(img, safe_mask, d=5, sigma_color=50, sigma_space=50):
    """
    Reduce noise while preserving edges using bilateral filter.
    
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
    Reduce salt-and-pepper noise using median filter.
    
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
    Remove small noise blobs using morphological operations.
    
    Args:
        img: Input image (uint8)
        safe_mask: Safe mask for hand region
        min_size: Minimum blob size to keep (default 5)
    
    Returns:
        Cleaned image
    """
    # Threshold to get binary (keep bright regions)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=safe_mask)
    
    # Remove small objects
    binary_clean = remove_small_objects(binary.astype(bool), max_size=min_size - 1)
    binary_clean = (binary_clean.astype(np.uint8) * 255)
    
    # Apply cleaned mask to original
    cleaned = cv2.bitwise_and(img, img, mask=binary_clean)
    cleaned = cv2.bitwise_and(cleaned, cleaned, mask=safe_mask)
    
    return cleaned


def enhance_illumination_corrected(hp, safe_mask, 
                                   highlight_strength=1.5,
                                   contrast_alpha=1.2,
                                   shadow_strength=1.2,
                                   sharpen_strength=1.3,
                                   sharpen_sigma=1.0,
                                   denoise=True,
                                   denoise_method='bilateral',
                                   denoise_strength='medium',
                                   remove_small_noise=True):
    """
    Comprehensive enhancement of illumination-corrected image.
    
    Applies a pipeline of enhancements to make vessels pop:
    1. Initial denoising (if enabled)
    2. Highlight strengthening (makes bright vessel areas stronger)
    3. Contrast enhancement
    4. Shadow strengthening (enhances contrast)
    5. Sharpening (makes edges crisper)
    6. Final denoising (if enabled)
    7. Small noise removal (morphological cleanup)
    
    Args:
        hp: Illumination-corrected image (from illumination_correction)
        safe_mask: Safe mask for hand region
        highlight_strength: Strength of highlight enhancement (default 1.5)
        contrast_alpha: Contrast multiplier (default 1.2)
        shadow_strength: Shadow strengthening factor (default 1.2)
        sharpen_strength: Sharpening strength (default 1.3)
        sharpen_sigma: Sharpening blur sigma (default 1.0)
        denoise: Whether to apply denoising (default True)
        denoise_method: 'bilateral', 'median', 'nlm', or 'both' (default 'bilateral')
        denoise_strength: 'light', 'medium', or 'strong' (default 'medium')
        remove_small_noise: Whether to remove small noise blobs (default True)
    
    Returns:
        Enhanced illumination-corrected image
    """
    enhanced = hp.copy()
    
    # 0. Initial denoising (if enabled) - reduces noise before enhancement
    if denoise:
        if denoise_method == 'bilateral':
            if denoise_strength == 'light':
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=5, sigma_color=50, sigma_space=50)
            elif denoise_strength == 'medium':
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=5, sigma_color=75, sigma_space=75)
            else:  # strong
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=7, sigma_color=100, sigma_space=100)
        elif denoise_method == 'median':
            ksize = 3 if denoise_strength == 'light' else (5 if denoise_strength == 'medium' else 7)
            enhanced = reduce_noise_median(enhanced, safe_mask, ksize=ksize)
        elif denoise_method == 'nlm':
            h = 7 if denoise_strength == 'light' else (10 if denoise_strength == 'medium' else 15)
            enhanced = reduce_noise_nlm(enhanced, safe_mask, h=h)
        elif denoise_method == 'both':
            # Apply both median (salt-pepper) and bilateral (general noise)
            enhanced = reduce_noise_median(enhanced, safe_mask, ksize=3)
            if denoise_strength == 'light':
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=5, sigma_color=50, sigma_space=50)
            elif denoise_strength == 'medium':
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=5, sigma_color=75, sigma_space=75)
            else:  # strong
                enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=7, sigma_color=100, sigma_space=100)
    
    # 1. Strengthen highlights (vessels are bright in hp)
    enhanced = strengthen_highlights(enhanced, safe_mask, strength=highlight_strength)
    
    # 2. Enhance contrast
    enhanced = enhance_contrast(enhanced, safe_mask, alpha=contrast_alpha)
    
    # 3. Strengthen shadows (enhances overall contrast)
    enhanced = strengthen_shadows(enhanced, safe_mask, strength=shadow_strength)
    
    # 4. Sharpen (makes vessel edges crisper)
    enhanced = sharpen_image(enhanced, safe_mask, strength=sharpen_strength, sigma=sharpen_sigma)
    
    # 5. Final denoising pass (light, to clean up after sharpening)
    if denoise and denoise_method != 'nlm':  # Skip if already did NLM (it's slow)
        enhanced = reduce_noise_bilateral(enhanced, safe_mask, d=5, sigma_color=50, sigma_space=50)
    
    # 6. Remove small noise blobs (morphological cleanup)
    if remove_small_noise:
        enhanced = remove_small_noise_morph(enhanced, safe_mask, min_size=5)
    
    return enhanced


def gabor_bank(img_u8, ksize, sig, lambd, gamma=0.5, psi=0.0, ntheta=12):
    """
    A4) Gabor filter bank for vessel/ridge response.
    
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
        r = np.maximum(r, 0)  # keep ridge-like positive response
        resp_max = np.maximum(resp_max, r)
    
    return resp_max


def vessel_response_single_scale(hp, safe_mask, ksize=31, sig=5.0, lambd=10.0, gamma=0.5, ntheta=12):
    """
    A4) Single-scale vessel response (baseline).
    
    Exact params:
    - ksize = 31
    - sigma = 5.0
    - lambda = 10.0
    - gamma = 0.5
    - psi = 0
    - num_orientations = 12
    """
    resp_max = gabor_bank(hp, ksize, sig, lambd, gamma, psi=0.0, ntheta=ntheta)
    
    # Normalize to uint8
    resp_u8 = cv2.normalize(resp_max, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=safe_mask)
    
    return resp_u8, resp_max


def vessel_response_multiscale(hp, safe_mask):
    """
    B2) Multi-scale vessel response - adapts to image resolution.
    
    Scales Gabor kernel sizes based on image dimensions to handle both
    high-res and low-res images appropriately.
    
    Returns normalized and thresholded response.
    """
    h, w = hp.shape[:2]
    img_size = max(h, w)
    
    # Scale factor: 1.0 for 600px, higher for larger images
    scale = max(1.0, img_size / 600.0)
    
    # Base scales optimized for ~600px images
    base_scales = [
        # (base_ksize, base_sigma, base_lambda, gamma_val, ntheta)
        (41, 7.0, 14.0, 0.5, 10),    # Medium vessels
        (61, 11.0, 22.0, 0.6, 8),    # Thick vessels  
        (81, 15.0, 30.0, 0.7, 6),    # Major veins
    ]
    
    responses = []
    for base_ksize, base_sigma, base_lambd, gamma_val, ntheta in base_scales:
        # Scale up for larger images
        ksize = int(base_ksize * scale)
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Must be odd
        sigma = base_sigma * scale
        lambd = base_lambd * scale
        
        resp = gabor_bank(hp, ksize, sigma, lambd, gamma=gamma_val, ntheta=ntheta)
        r_min, r_max = resp.min(), resp.max()
        resp_norm = (resp - r_min) / (r_max - r_min + 1e-9)
        responses.append(resp_norm)
    
    # Max across scales
    resp = responses[0]
    for r in responses[1:]:
        resp = np.maximum(resp, r)
    
    # Adaptive threshold: higher for larger/noisier images
    threshold_pct = min(70, 50 + (scale - 1) * 10)
    vals = resp[safe_mask > 0]
    if len(vals) > 0:
        threshold = np.percentile(vals, threshold_pct)
        resp = np.where(resp > threshold, resp, 0)
    
    # Blur proportional to image size
    blur_sigma = 2.0 * scale
    resp = cv2.GaussianBlur(resp.astype(np.float32), (0, 0), blur_sigma)
    
    # Re-normalize
    r_min, r_max = resp.min(), resp.max()
    if r_max > r_min:
        resp = (resp - r_min) / (r_max - r_min)
    
    resp_u8 = (resp * 255).astype(np.uint8)
    resp_u8 = cv2.bitwise_and(resp_u8, resp_u8, mask=safe_mask)
    
    return resp_u8, resp


def enhance_vessel_map(resp_u8, safe_mask, gamma=0.5, p_low=1, p_high=99):
    """
    B1) Normalize within hand + apply gamma (makes vessels more obvious).
    
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
    # Percentile threshold
    vals = resp_u8[safe_mask > 0]
    t = np.percentile(vals, percentile_thresh)
    binv = (resp_u8 >= t) & (safe_mask > 0)
    
    # Remove small objects
    # Note: newer scikit-image uses max_size (removes objects <= max_size)
    # So to remove objects < min_size, we use max_size = min_size - 1
    binv = remove_small_objects(binv.astype(bool), max_size=min_size - 1)
    bin_u8 = (binv.astype(np.uint8) * 255)
    
    # Morphological close
    bin_u8 = cv2.morphologyEx(
        bin_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    
    # Skeletonize
    sk = skeletonize(bin_u8 > 0).astype(np.uint8) * 255
    sk = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Create red overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[sk > 0] = (0, 0, 255)
    
    return overlay, sk


def palm_vein_feature_map(gray, use_multiscale=True, enhance=True, create_overlay=True,
                          use_illumination_as_feature=False,
                          enhance_illumination=True,
                          highlight_strength=1.5,
                          contrast_alpha=1.2,
                          shadow_strength=1.2,
                          sharpen_strength=1.3,
                          sharpen_sigma=1.0,
                          denoise=True,
                          denoise_method='bilateral',
                          denoise_strength='medium',
                          remove_small_noise=True,
                          segmentation_method='gradient',
                          preprocess_for_segmentation=True,
                          otsu_bias=0.85,
                          canny_low=30,
                          canny_high=100,
                          exclude_fingers=True,
                          finger_width=60):
    """
    Complete pipeline: extract palm vein feature map.
    
    Args:
        gray: Input grayscale image
        use_multiscale: If True, use multi-scale Gabor banks (B2), else single-scale (A4)
        enhance: If True, apply robust normalization + gamma (B1) to Gabor response
        create_overlay: If True, create red overlay visualization
        use_illumination_as_feature: If True, use enhanced illumination-corrected as final feature map
        enhance_illumination: If True, apply highlight/contrast/shadow/sharpen to illumination-corrected
        highlight_strength: Strength of highlight enhancement (default 1.5)
        contrast_alpha: Contrast multiplier (default 1.2)
        shadow_strength: Shadow strengthening factor (default 1.2)
        sharpen_strength: Sharpening strength (default 1.3)
        sharpen_sigma: Sharpening blur sigma (default 1.0)
        denoise: Whether to apply denoising (default True)
        denoise_method: 'bilateral', 'median', 'nlm', or 'both' (default 'bilateral')
        denoise_strength: 'light', 'medium', or 'strong' (default 'medium')
        remove_small_noise: Whether to remove small noise blobs (default True)
        segmentation_method: 'gradient' (edge-based) or 'otsu' (threshold-based)
        preprocess_for_segmentation: Whether to preprocess image before segmentation
        otsu_bias: Bias for Otsu threshold (< 1.0 = less strict, default 0.85)
        canny_low: Low threshold for Canny edge detection (default 30)
        canny_high: High threshold for Canny edge detection (default 100)
    
    Returns:
        enhanced: Final enhanced vessel feature map (uint8)
        resp_u8: Raw vessel response map (uint8) or enhanced illumination-corrected
        safe_mask: Safe mask (hand region, avoiding boundaries)
        overlay: Red overlay visualization (BGR) if create_overlay=True, else None
        hp_enhanced: Enhanced illumination-corrected image (if use_illumination_as_feature)
    """
    # A2) Hand segmentation (now with preprocessing and method selection)
    hand_mask, safe_mask = create_hand_segmentation_mask(
        gray, 
        method=segmentation_method,
        preprocess=preprocess_for_segmentation,
        otsu_bias=otsu_bias,
        canny_low=canny_low,
        canny_high=canny_high,
        exclude_fingers_flag=exclude_fingers,
        finger_width=finger_width
    )
    
    # A3) Illumination correction (with increased sensitivity)
    hp = illumination_correction(gray, safe_mask)
    
    # Option: Use enhanced illumination-corrected as final feature map
    if use_illumination_as_feature:
        if enhance_illumination:
            hp_enhanced = enhance_illumination_corrected(
                hp, safe_mask,
                highlight_strength=highlight_strength,
                contrast_alpha=contrast_alpha,
                shadow_strength=shadow_strength,
                sharpen_strength=sharpen_strength,
                sharpen_sigma=sharpen_sigma,
                denoise=denoise,
                denoise_method=denoise_method,
                denoise_strength=denoise_strength,
                remove_small_noise=remove_small_noise
            )
        else:
            hp_enhanced = hp.copy()
        
        # Use enhanced illumination-corrected as the final feature map
        enhanced = hp_enhanced
        resp_u8 = hp_enhanced  # For consistency in return values
        
        # A5) Red overlay (optional, for visualization)
        overlay = None
        if create_overlay:
            overlay, _ = create_red_overlay(gray, enhanced, safe_mask, percentile_thresh=92.5, min_size=140)
        
        return enhanced, resp_u8, safe_mask, overlay, hp_enhanced
    
    # Original pipeline: Gabor-based vessel response
    # A4/B2) Vessel response
    if use_multiscale:
        resp_u8, resp_float = vessel_response_multiscale(hp, safe_mask)
    else:
        resp_u8, resp_float = vessel_response_single_scale(hp, safe_mask)
    
    # B1) Enhance (robust norm + gamma)
    if enhance:
        enhanced = enhance_vessel_map(resp_u8, safe_mask, gamma=0.5, p_low=1, p_high=99)
    else:
        enhanced = resp_u8.copy()
    
    # A5) Red overlay (optional, for visualization)
    overlay = None
    if create_overlay:
        overlay, _ = create_red_overlay(gray, enhanced, safe_mask, percentile_thresh=92.5, min_size=140)
    
    return enhanced, resp_u8, safe_mask, overlay, None


# -------------------------------------------------------------------
# OEG (Orientation Energy Grid) feature extraction (Option A)
# -------------------------------------------------------------------

def _pca_angle_deg_from_mask(mask_u8: np.ndarray) -> float:
    """Return PCA major-axis angle in degrees (x-axis reference)."""
    ys, xs = np.where(mask_u8 > 0)
    if xs.size < 50:
        return 0.0

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    pts -= pts.mean(axis=0, keepdims=True)

    # covariance (2x2)
    cov = (pts.T @ pts) / max(1, pts.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    v = eigvecs[:, int(np.argmax(eigvals))]  # principal axis (x,y)

    # Fix sign ambiguity to avoid random 180 flips:
    # enforce pointing roughly to +x
    if v[0] < 0:
        v = -v

    angle = np.degrees(np.arctan2(v[1], v[0]))  # atan2(y, x)
    return float(angle)


def _rotate_keep_size(img_u8: np.ndarray, mask_u8: np.ndarray, angle_deg: float):
    """Rotate image+mask about center, keep original size."""
    if abs(angle_deg) < 1e-3:
        return img_u8, mask_u8

    h, w = img_u8.shape[:2]
    center = (w * 0.5, h * 0.5)
    # rotate by -angle so principal axis aligns with +x
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    img_r = cv2.warpAffine(
        img_u8, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    mask_r = cv2.warpAffine(
        mask_u8, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    mask_r = (mask_r > 0).astype(np.uint8) * 255
    return img_r, mask_r


def _crop_and_resize(img_u8: np.ndarray, mask_u8: np.ndarray, out_size: int = 256, margin: int = 10):
    """Crop to mask bbox (+margin) then resize to (out_size, out_size)."""
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0:
        z = np.zeros((out_size, out_size), dtype=np.uint8)
        return z, z

    h, w = img_u8.shape[:2]
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)

    img_c = img_u8[y0:y1, x0:x1]
    m_c = mask_u8[y0:y1, x0:x1]

    img_r = cv2.resize(img_c, (out_size, out_size), interpolation=cv2.INTER_AREA)
    m_r = cv2.resize(m_c, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m_r = (m_r > 0).astype(np.uint8) * 255
    return img_r, m_r


def _gabor_stack(img_u8: np.ndarray,
                 ksize: int = 31,
                 sig: float = 5.0,
                 lambd: float = 10.0,
                 gamma: float = 0.5,
                 psi: float = 0.0,
                 ntheta: int = 8) -> np.ndarray:
    """
    Return stack of ReLU Gabor responses: shape [K, H, W] float32.
    """
    thetas = np.linspace(0, np.pi, ntheta, endpoint=False)
    src = img_u8.astype(np.float32)
    H, W = img_u8.shape[:2]
    stack = np.zeros((ntheta, H, W), dtype=np.float32)

    for i, th in enumerate(thetas):
        kern = cv2.getGaborKernel((ksize, ksize), sig, th, lambd, gamma, psi, ktype=cv2.CV_32F)
        kern -= kern.mean()
        r = cv2.filter2D(src, cv2.CV_32F, kern)
        r = np.maximum(r, 0)  # ReLU
        stack[i] = r

    return stack


def _grid_pool(stack: np.ndarray, mask_u8: np.ndarray, grid: int = 16) -> np.ndarray:
    """
    stack: [K,H,W] float32
    mask_u8: [H,W] uint8 0/255
    Returns: feature vector float32 length = grid*grid*K
    
    Per-cell normalization across orientations for gain invariance.
    """
    K, H, W = stack.shape
    mask = (mask_u8 > 0)

    ys = np.linspace(0, H, grid + 1, dtype=int)
    xs = np.linspace(0, W, grid + 1, dtype=int)

    feats = []
    for gy in range(grid):
        y0, y1 = ys[gy], ys[gy + 1]
        for gx in range(grid):
            x0, x1 = xs[gx], xs[gx + 1]
            m = mask[y0:y1, x0:x1]

            if not np.any(m):
                feats.extend([0.0] * K)
                continue

            cell = np.zeros((K,), dtype=np.float32)
            for k in range(K):
                v = stack[k, y0:y1, x0:x1][m]
                cell[k] = float(v.mean())

            # Per-cell normalization across orientations (key for lighting robustness)
            cell /= (np.linalg.norm(cell) + 1e-6)

            feats.extend(cell.tolist())

    return np.array(feats, dtype=np.float32)


def extract_oeg_feature_vector(gray_u8: np.ndarray,
                               safe_mask_u8: np.ndarray,
                               roi_size: int = 256,
                               roi_margin: int = 10,
                               ntheta: int = 8,
                               grid: int = 16,
                               ksize: int = 31,
                               sig: float = 5.0,
                               lambd: float = 10.0,
                               gamma: float = 0.5) -> np.ndarray:
    """
    Pipeline:
      1) hp = illumination_correction_for_features(gray, safe_mask)  # No CLAHE for invariance
      2) PCA rotate (small rotation tolerance)
      3) crop + resize ROI
      4) per-orientation Gabor stack
      5) grid pooling with per-cell orientation normalization
      6) L2 normalize
    """
    # Use simplified illumination correction (no CLAHE) for better photometric invariance
    hp = illumination_correction_for_features(gray_u8, safe_mask_u8)

    # PCA align using safe_mask
    angle = _pca_angle_deg_from_mask(safe_mask_u8)
    hp_a, mask_a = _rotate_keep_size(hp, safe_mask_u8, angle)

    # Crop + resize to canonical ROI
    hp_r, mask_r = _crop_and_resize(hp_a, mask_a, out_size=roi_size, margin=roi_margin)

    # Oriented response stack (do NOT max across orientations)
    stack = _gabor_stack(hp_r, ksize=ksize, sig=sig, lambd=lambd, gamma=gamma, psi=0.0, ntheta=ntheta)

    # Grid pooling (translation tolerant)
    vec = _grid_pool(stack, mask_r, grid=grid)

    # L2 normalize
    n = float(np.linalg.norm(vec) + 1e-6)
    vec = (vec / n).astype(np.float32)
    return vec


def main():
    parser = argparse.ArgumentParser(
        description="Extract palm vein feature maps from grayscale images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (hybrid segmentation, default settings - RECOMMENDED)
  python feature_map.py input.jpg -o output_dir

  # Best quality: illumination mode with strong enhancements
  python feature_map.py input.jpg --use-illumination --highlight-strength 2.5 \\
    --contrast-alpha 1.8 --shadow-strength 1.6 --sharpen-strength 2.2 \\
    --sharpen-sigma 0.6 --denoise-method both --denoise-strength medium --all -o output_dir

  # Adaptive segmentation (best for uneven lighting)
  python feature_map.py input.jpg -o output_dir --segmentation adaptive

  # Otsu segmentation with reduced strictness (faster, captures more)
  python feature_map.py input.jpg -o output_dir --segmentation otsu --otsu-bias 0.75

  # Multi-scale Gabor with all intermediate outputs
  python feature_map.py input.jpg -o output_dir --multiscale --all
        """
    )
    parser.add_argument("input", type=Path, help="Input grayscale image path")
    parser.add_argument("-o", "--output", type=Path, default=None,
                       help="Output directory (default: input_dir/vein_feature_maps)")
    parser.add_argument("--multiscale", action="store_true", default=True,
                       help="Use multi-scale Gabor banks (default: True)")
    parser.add_argument("--no-multiscale", dest="multiscale", action="store_false",
                       help="Use single-scale Gabor bank only")
    parser.add_argument("--enhance", action="store_true", default=True,
                       help="Apply robust normalization + gamma enhancement (default: True)")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false",
                       help="Skip enhancement step")

    # CHANGE: default overlay OFF so we don't spam files
    parser.add_argument("--overlay", action="store_true", default=False,
                       help="Create red overlay visualization (default: False)")
    parser.add_argument("--no-overlay", dest="overlay", action="store_false",
                       help="Skip overlay creation")

    parser.add_argument("--all", action="store_true",
                       help="Save all intermediate outputs (hand_mask, hp, resp_u8, etc.)")
    parser.add_argument("--use-illumination", action="store_true", default=False,
                       help="Use enhanced illumination-corrected image as final feature map (instead of Gabor response)")
    parser.add_argument("--no-enhance-illumination", dest="enhance_illumination", action="store_false", default=True,
                       help="Skip enhancement of illumination-corrected image (only if --use-illumination)")
    parser.add_argument("--highlight-strength", type=float, default=1.5,
                       help="Highlight strengthening factor (default: 1.5)")
    parser.add_argument("--contrast-alpha", type=float, default=1.2,
                       help="Contrast enhancement multiplier (default: 1.2)")
    parser.add_argument("--shadow-strength", type=float, default=1.2,
                       help="Shadow strengthening factor (default: 1.2)")
    parser.add_argument("--sharpen-strength", type=float, default=1.3,
                       help="Sharpening strength (default: 1.3)")
    parser.add_argument("--sharpen-sigma", type=float, default=1.0,
                       help="Sharpening blur sigma (default: 1.0)")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false", default=True,
                       help="Skip denoising")
    parser.add_argument("--denoise-method", type=str, default="bilateral",
                       choices=["bilateral", "median", "nlm", "both"],
                       help="Denoising method: bilateral, median, nlm (non-local means), or both (default: bilateral)")
    parser.add_argument("--denoise-strength", type=str, default="medium",
                       choices=["light", "medium", "strong"],
                       help="Denoising strength: light, medium, or strong (default: medium)")
    parser.add_argument("--no-remove-small-noise", dest="remove_small_noise", action="store_false", default=True,
                       help="Skip morphological noise removal")
    
    # Segmentation options
    parser.add_argument("--segmentation", type=str, default="hybrid",
                       choices=["hybrid", "adaptive", "otsu"],
                       help="Segmentation method: hybrid (Otsu+edges, RECOMMENDED), adaptive (for uneven lighting), or otsu (threshold-based, default: hybrid)")
    parser.add_argument("--no-preprocess-seg", dest="preprocess_seg", action="store_false", default=True,
                       help="Skip preprocessing before segmentation (not recommended)")
    parser.add_argument("--otsu-bias", type=float, default=0.85,
                       help="Otsu threshold bias (< 1.0 = less strict, captures more of hand, default: 0.85)")
    parser.add_argument("--canny-low", type=int, default=30,
                       help="Canny edge detection low threshold (default: 30)")
    parser.add_argument("--canny-high", type=int, default=100,
                       help="Canny edge detection high threshold (default: 100)")
    parser.add_argument("--finger-width", type=int, default=60,
                       help="Max finger width in pixels for exclusion (default: 60)")
    
    # ----------------------------
    # Feature vector mode (OEG)
    # ----------------------------
    parser.add_argument("--feature-vector", action="store_true", default=False,
                        help="Compute and save OEG feature vector only (no images)")
    parser.add_argument("--vec-out", type=Path, default=None,
                        help="Output path prefix for vector (default: output_dir/<stem>_oeg)")
    parser.add_argument("--vec-grid", type=int, default=16,
                        help="Grid size for pooling (default: 16)")
    parser.add_argument("--vec-ntheta", type=int, default=8,
                        help="Number of Gabor orientations (default: 8)")
    parser.add_argument("--roi-size", type=int, default=256,
                        help="Canonical ROI size (default: 256)")
    parser.add_argument("--roi-margin", type=int, default=10,
                        help="Crop margin in pixels before resize (default: 10)")

    # Gabor params for feature extraction (leave defaults unless tuning)
    parser.add_argument("--vec-ksize", type=int, default=31,
                        help="Gabor kernel size for vector extraction (default: 31)")
    parser.add_argument("--vec-sigma", type=float, default=5.0,
                        help="Gabor sigma for vector extraction (default: 5.0)")
    parser.add_argument("--vec-lambda", type=float, default=10.0,
                        help="Gabor lambda for vector extraction (default: 10.0)")
    parser.add_argument("--vec-gamma", type=float, default=0.5,
                        help="Gabor gamma for vector extraction (default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Setup output directory
    if args.output is None:
        args.output = args.input.parent / "vein_feature_maps"
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Read image
    print(f"Reading image: {args.input}")
    gray = read_image_grayscale(args.input)
    
    # ------------------------------------------------------------
    # Feature vector mode: compute OEG vector and exit
    # ------------------------------------------------------------
    if args.feature_vector:
        # Build masks using the SAME segmentation config as your pipeline
        hand_mask, safe_mask = create_hand_segmentation_mask(
            gray,
            method=args.segmentation,
            preprocess=args.preprocess_seg,
            otsu_bias=args.otsu_bias,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            exclude_fingers_flag=True,
            finger_width=args.finger_width
        )

        vec = extract_oeg_feature_vector(
            gray, safe_mask,
            roi_size=args.roi_size,
            roi_margin=args.roi_margin,
            ntheta=args.vec_ntheta,
            grid=args.vec_grid,
            ksize=args.vec_ksize,
            sig=args.vec_sigma,
            lambd=args.vec_lambda,
            gamma=args.vec_gamma
        )

        stem = args.input.stem
        if args.output is None:
            args.output = args.input.parent / "vein_feature_maps"
        args.output.mkdir(parents=True, exist_ok=True)

        prefix = args.vec_out
        if prefix is None:
            prefix = args.output / f"{stem}_oeg"

        # Save
        np.save(str(prefix) + ".npy", vec)
        np.savetxt(str(prefix) + ".txt", vec[None, :], fmt="%.8g")

        # Print full vector to terminal (one line)
        print(" ".join(f"{x:.8g}" for x in vec))
        print(f"\nSaved feature vector:")
        print(f"  npy: {str(prefix) + '.npy'}")
        print(f"  txt: {str(prefix) + '.txt'}")
        print(f"  dim: {vec.size} (grid={args.vec_grid}, ntheta={args.vec_ntheta})")

        sys.exit(0)
    
    # Run pipeline
    print(f"Running palm vein feature extraction pipeline (segmentation: {args.segmentation})...")
    enhanced, resp_u8, safe_mask, overlay, hp_enhanced = palm_vein_feature_map(
        gray,
        use_multiscale=args.multiscale,
        enhance=args.enhance,
        create_overlay=args.overlay,
        use_illumination_as_feature=args.use_illumination,
        enhance_illumination=args.enhance_illumination,
        highlight_strength=args.highlight_strength,
        contrast_alpha=args.contrast_alpha,
        shadow_strength=args.shadow_strength,
        sharpen_strength=args.sharpen_strength,
        sharpen_sigma=args.sharpen_sigma,
        denoise=args.denoise,
        denoise_method=args.denoise_method,
        denoise_strength=args.denoise_strength,
        remove_small_noise=args.remove_small_noise,
        segmentation_method=args.segmentation,
        preprocess_for_segmentation=args.preprocess_seg,
        otsu_bias=args.otsu_bias,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        exclude_fingers=True,  # Always exclude fingers
        finger_width=args.finger_width
    )
    
    # Save outputs
    stem = args.input.stem
    
    # MAIN OUTPUT (default): raw response map (most stable for downstream feature vectoring)
    cv2.imwrite(str(args.output / f"{stem}_raw_response.png"), resp_u8)
    print(f"✓ Raw response map: {args.output / f'{stem}_raw_response.png'}")

    # Optional overlay
    if args.overlay and overlay is not None:
        cv2.imwrite(str(args.output / f"{stem}_overlay.png"), overlay)
        print(f"✓ Red overlay: {args.output / f'{stem}_overlay.png'}")
    
    # Save intermediates if requested
    if args.all:
        # Re-run to get intermediates
        hand_mask, safe_mask_dbg = create_hand_segmentation_mask(
            gray, 
            method=args.segmentation,
            preprocess=args.preprocess_seg,
            otsu_bias=args.otsu_bias,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            exclude_fingers_flag=True,  # Always exclude fingers
            finger_width=args.finger_width
        )
        hp = illumination_correction(gray, safe_mask_dbg)
        
        # Always save segmentation masks for debugging
        cv2.imwrite(str(args.output / f"{stem}_01_hand_mask.png"), hand_mask)
        cv2.imwrite(str(args.output / f"{stem}_02_safe_mask.png"), safe_mask_dbg)
        
        if args.use_illumination:
            # Save illumination-corrected and enhanced versions
            cv2.imwrite(str(args.output / f"{stem}_03_illumination_corrected.png"), hp)
            if hp_enhanced is not None:
                cv2.imwrite(str(args.output / f"{stem}_03_illumination_enhanced.png"), hp_enhanced)
        else:
            # Original pipeline intermediates
            if args.multiscale:
                resp_dbg, _ = vessel_response_multiscale(hp, safe_mask_dbg)
            else:
                resp_dbg, _ = vessel_response_single_scale(hp, safe_mask_dbg)
            
            cv2.imwrite(str(args.output / f"{stem}_00_original.png"), gray)
            cv2.imwrite(str(args.output / f"{stem}_03_illumination_corrected.png"), hp)
            cv2.imwrite(str(args.output / f"{stem}_04_raw_response.png"), resp_dbg)
            
            if args.enhance:
                cv2.imwrite(str(args.output / f"{stem}_05_enhanced_feature_map.png"), enhanced)
        
        print(f"✓ Intermediate outputs saved to: {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()