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

    # FIX: correct OpenCV function name
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)

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

def pre_enhance_input(bw_img, safe_mask,
                      denoise=True,
                      denoise_method="bilateral",
                      denoise_strength="light",
                      apply_clahe=True,
                      clahe_clip_limit=1.5,
                      clahe_tile_grid_size=(8, 8),
                      contrast_alpha=1.05,
                      contrast_beta=0,
                      gamma=None,
                      sharpen=False,
                      sharpen_strength=0.35,
                      sharpen_sigma=1.0):
    """
    Optional filters that apply to the ORIGINAL image (or lightly-CLAHE'd original),
    before illumination correction.

    Keep this gentle for repeatability (verification). Avoid percentile boosts here.
    """
    if bw_img.dtype != np.uint8:
        img = np.clip(bw_img, 0, 255).astype(np.uint8)
    else:
        img = bw_img

    if safe_mask is None:
        safe = np.ones_like(img, dtype=np.uint8) * 255
    else:
        safe = (safe_mask > 0).astype(np.uint8) * 255

    # Always restrict to palm-safe area first
    out = cv2.bitwise_and(img, img, mask=safe)

    # 1) Gentle denoise
    if denoise:
        if denoise_method == "bilateral":
            if denoise_strength == "light":
                out = cv2.bilateralFilter(out, 5, 40, 40)
            elif denoise_strength == "medium":
                out = cv2.bilateralFilter(out, 5, 70, 70)
            else:
                out = cv2.bilateralFilter(out, 7, 95, 95)
        elif denoise_method == "median":
            k = 3 if denoise_strength == "light" else (5 if denoise_strength == "medium" else 7)
            out = cv2.medianBlur(out, k)
        elif denoise_method == "nlm":
            h = 6 if denoise_strength == "light" else (9 if denoise_strength == "medium" else 12)
            out = cv2.fastNlMeansDenoising(out, None, h=h, templateWindowSize=7, searchWindowSize=21)
        elif denoise_method == "both":
            out = cv2.medianBlur(out, 3)
            if denoise_strength == "light":
                out = cv2.bilateralFilter(out, 5, 40, 40)
            elif denoise_strength == "medium":
                out = cv2.bilateralFilter(out, 5, 70, 70)
            else:
                out = cv2.bilateralFilter(out, 7, 95, 95)
        else:
            raise ValueError(f"Unknown denoise_method: {denoise_method}")

        out = cv2.bitwise_and(out, out, mask=safe)

    # 2) Light CLAHE
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=clahe_tile_grid_size)
        out = clahe.apply(out)
        out = cv2.bitwise_and(out, out, mask=safe)

    # 3) Optional mild gamma
    if gamma is not None and abs(float(gamma) - 1.0) > 1e-6:
        x = out.astype(np.float32) / 255.0
        x = np.power(np.clip(x, 0.0, 1.0), float(gamma))
        out = (x * 255.0).astype(np.uint8)
        out = cv2.bitwise_and(out, out, mask=safe)

    # 4) Optional mild global contrast
    if abs(float(contrast_alpha) - 1.0) > 1e-6 or int(contrast_beta) != 0:
        out = cv2.convertScaleAbs(out, alpha=float(contrast_alpha), beta=int(contrast_beta))
        out = cv2.bitwise_and(out, out, mask=safe)

    # 5) Optional gentle unsharp (usually OFF for verification)
    if sharpen:
        blurred = cv2.GaussianBlur(out, (0, 0), float(sharpen_sigma))
        out_f = out.astype(np.float32)
        bl_f = blurred.astype(np.float32)
        out = np.clip(out_f + (out_f - bl_f) * float(sharpen_strength), 0, 255).astype(np.uint8)
        out = cv2.bitwise_and(out, out, mask=safe)

    return out

def illumination_correction(bw_img, safe_mask):
    """
    Make dark veins bright:

    1) Pass through large Gaussian blur to estimate background illumination
    2) 
    
    :param bw_img: Description
    :param safe_mask: Description
    """

    # FIX: correct OpenCV function name
    background = cv2.GaussianBlur(bw_img, (0, 0), 35.0)

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
    # FIX: mask keyword + argument order
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=safe_mask)

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
    # FIX: use min_size argument (max_size will error on many installs)
    binary_clean = morphology.remove_small_objects(binary.astype(bool), min_size=min_size)
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
    # FIX: use min_size argument (max_size will error on many installs)
    binv = morphology.remove_small_objects(binv.astype(bool), min_size=min_size)
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
    hand_mask, safe_mask = segmentation_hand_mask(gray)

    # (Optional) Enhance illumination in certain areas on correct img to make vessels more obvious
    # NOTE: per your intended behavior, this applies to the ORIGINAL image before illumination_correction
    gray_pre = gray
    if enhance_illumination:
        gray_pre = pre_enhance_input(
            gray, safe_mask,
            denoise=denoise,
            denoise_method=denoise_method,
            denoise_strength=denoise_strength,
            apply_clahe=True,
            clahe_clip_limit=1.5,
            clahe_tile_grid_size=(8, 8),
            contrast_alpha=1.05,
            contrast_beta=0,
            gamma=None,
            sharpen=False
        )
    
    hp = illumination_correction(gray_pre, safe_mask)
    
    # Option: Use enhanced illumination-corrected as final feature map
    if use_illumination_as_feature:
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
    gray = load_grayscale(str(args.input))
    
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
        hand_mask, safe_mask = segmentation_hand_mask(gray)

        gray_pre = gray
        if args.enhance_illumination:
            gray_pre = pre_enhance_input(
                gray, safe_mask,
                denoise=args.denoise,
                denoise_method=args.denoise_method,
                denoise_strength=args.denoise_strength,
                apply_clahe=True,
                clahe_clip_limit=1.5,
                clahe_tile_grid_size=(8, 8),
                contrast_alpha=1.05,
                contrast_beta=0,
                gamma=None,
                sharpen=False
            )

        hp = illumination_correction(gray_pre, safe_mask)
        
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
