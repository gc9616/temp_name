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


def create_hand_segmentation_mask(gray):
    """
    A2) Hand segmentation mask (largest bright component).
    
    Steps:
    1. Blur + Otsu thresholding
    2. Largest connected component
    3. Morphological cleanup
    4. Safe mask via distance transform (avoids boundary artifacts)
    """
    # Blur + Otsu
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hand_mask = (labels == largest).astype(np.uint8) * 255
    else:
        hand_mask = thr.copy()
    
    # Morph cleanup
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
    
    # Safe mask via distance transform (key to avoid boundary artifacts)
    # DIST_L2, maskSize 5, threshold > 6
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    safe_mask = (dist > 6).astype(np.uint8) * 255
    
    return hand_mask, safe_mask


def illumination_correction(gray, safe_mask):
    """
    A3) Illumination correction (veins dark → become bright).
    
    Parameters:
    - background blur sigma = 35.0
    - CLAHE: clipLimit = 2.0, tileGridSize = (8,8)
    - post blur sigma = 1.2
    """
    # Background blur
    bg = cv2.GaussianBlur(gray, (0, 0), 35.0)
    
    # High-pass: dark veins -> brighter
    hp = cv2.subtract(bg, gray)
    hp = cv2.bitwise_and(hp, hp, mask=safe_mask)
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hp = clahe.apply(hp)
    
    # Post blur
    hp = cv2.GaussianBlur(hp, (0, 0), 1.2)
    
    return hp


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
    B2) Multi-scale vessel response (captures thin + thick veins).
    
    Runs two Gabor banks:
    - Fine scale (thin vessels): ksize=31, sigma=5, lambda=10
    - Coarse scale (thicker vessels): ksize=51, sigma=8, lambda=18
    
    Returns normalized and maxed response.
    """
    # Fine scale (thin vessels)
    resp_fine = gabor_bank(hp, 31, 5.0, 10.0, gamma=0.5, ntheta=12)
    
    # Coarse scale (thicker vessels)
    resp_coarse = gabor_bank(hp, 51, 8.0, 18.0, gamma=0.7, ntheta=10)
    
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
                          remove_small_noise=True):
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
    
    Returns:
        enhanced: Final enhanced vessel feature map (uint8)
        resp_u8: Raw vessel response map (uint8) or enhanced illumination-corrected
        safe_mask: Safe mask (hand region, avoiding boundaries)
        overlay: Red overlay visualization (BGR) if create_overlay=True, else None
        hp_enhanced: Enhanced illumination-corrected image (if use_illumination_as_feature)
    """
    # A2) Hand segmentation
    hand_mask, safe_mask = create_hand_segmentation_mask(gray)
    
    # A3) Illumination correction
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract palm vein feature maps from grayscale images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (single-scale, enhanced)
  python palm_vein_feature_map.py input.jpg -o output_dir

  # Multi-scale with all outputs
  python palm_vein_feature_map.py input.jpg -o output_dir --multiscale --all

  # Single-scale, no enhancement, no overlay
  python palm_vein_feature_map.py input.jpg -o output_dir --no-multiscale --no-enhance --no-overlay
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
    parser.add_argument("--overlay", action="store_true", default=True,
                       help="Create red overlay visualization (default: True)")
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
    
    # Run pipeline
    print("Running palm vein feature extraction pipeline...")
    result = palm_vein_feature_map(
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
        remove_small_noise=args.remove_small_noise
    )
    enhanced, resp_u8, safe_mask, overlay, hp_enhanced = result
    
    # Save outputs
    stem = args.input.stem
    
    # Main outputs
    cv2.imwrite(str(args.output / f"{stem}_enhanced_feature_map.png"), enhanced)
    print(f"✓ Enhanced feature map: {args.output / f'{stem}_enhanced_feature_map.png'}")
    
    if args.overlay and overlay is not None:
        cv2.imwrite(str(args.output / f"{stem}_overlay.png"), overlay)
        print(f"✓ Red overlay: {args.output / f'{stem}_overlay.png'}")
    
    # Save intermediates if requested
    if args.all:
        # Re-run to get intermediates
        hand_mask, safe_mask = create_hand_segmentation_mask(gray)
        hp = illumination_correction(gray, safe_mask)
        
        if args.use_illumination:
            # Save illumination-corrected and enhanced versions
            cv2.imwrite(str(args.output / f"{stem}_03_illumination_corrected.png"), hp)
            if hp_enhanced is not None:
                cv2.imwrite(str(args.output / f"{stem}_03_illumination_enhanced.png"), hp_enhanced)
        else:
            # Original pipeline intermediates
            if args.multiscale:
                resp_u8, _ = vessel_response_multiscale(hp, safe_mask)
            else:
                resp_u8, _ = vessel_response_single_scale(hp, safe_mask)
            
            cv2.imwrite(str(args.output / f"{stem}_00_original.png"), gray)
            cv2.imwrite(str(args.output / f"{stem}_01_hand_mask.png"), hand_mask)
            cv2.imwrite(str(args.output / f"{stem}_02_safe_mask.png"), safe_mask)
            cv2.imwrite(str(args.output / f"{stem}_03_illumination_corrected.png"), hp)
            cv2.imwrite(str(args.output / f"{stem}_04_raw_response.png"), resp_u8)
        
        print(f"✓ Intermediate outputs saved to: {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()