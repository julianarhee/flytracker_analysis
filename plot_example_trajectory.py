#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Create Time: 2025-05-12 14:38:21
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-05-12 14:38:30
 # @ Description:
 '''
#%%
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils as util
import plotting as putil
#%%

plot_style='dark'
putil.set_sns_style(style=plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%

assay = 'free_behavior' #'2d_projector'

if assay == '2d_projector':
    basedir = '/Volumes/Juliana/2d_projector_analysis/circle_diffspeeds/FlyTracker'
else: 
    basedir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
    srcdir = '/Volumes/Giacomo/free_behavior_data'

figdir = os.path.join(basedir, 'plot_example_trajectory')
if plot_style == 'white':
    figdir = os.path.join(figdir, 'white')
    
if not os.path.exists(figdir):
    os.makedirs(figdir)
print('figdir:', figdir)

# %%

curr_species = 'Dele'
# --------------------------------------------------
if curr_species == 'Dyak':
    acquisition = '20240116-1015-fly1-yakWT_4do_sh_yakWT_4do_gh'
    viddir = os.path.join(srcdir, acquisition)
elif curr_species == 'Dmel':
    acquisition = '20240119-1020-fly3-melWT_4do_sh_melWT_4do_gh'
    viddir = os.path.join(srcdir, acquisition)
elif curr_species == 'Dele':
    if assay == '2d_projector':
        acquisition = '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10'
        srcdir = '/Volumes/Juliana/2d-projector/20240214'
        viddir = srcdir
    else:
        #acquisition = '20240814-0925_fly1_Dele-HK_WT_6do_gh'
        acquisition = '20240814-1023_fly2_Dele-HK_WT_6do_gh'
        srcdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/Caitlin_elehk_38mm'
        viddir = os.path.join(srcdir, acquisition)

#%%a
#curr_species = 'Dyak' if 'yak' in acquisition else 'Dmel'

yak_bouts = [
    [14900, 15650], #700],
    [16200, 17100], #*
    [18150, 18800], #* 
    [19200, 20200],
    [20500, 21500],
    [23700, 24450],
    [25800, 26440],
    [39800, 40460], #*
]
mel_bouts = [
    [6060, 7150], #200],
    [7200, 7950],
    [8800, 9450],
    [10400, 10910]
]

if acquisition == '20240814-0925_fly1_Dele-HK_WT_6do_gh':
    ele_bouts = [
        [10603, 10670], 
        [36703, 36746],
        [46451, 46527],  # maybe
    ]
elif acquisition == '20240814-1023_fly2_Dele-HK_WT_6do_gh':
    ele_bouts = [
        [6438, 6503], 
        [21161, 21243],
        [23290, 23339],
        [31294, 31363],
        [34355, 34440],
        [38387, 38468],
        [60348, 60403], #42-44 are good
        [71259, 71342],
        [77732, 77803],
        [80507, 80581],
        [92828, 92906],
        [14828, 14936], # CHASE
        [19082, 19194],
        #[19657, 19750],
        [26808, 27026],
        [27656, 27744], # chase afar?
        [28377, 28490],
        [36238, 36382]      
    ]
elif acquisition == '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10':
    ele_bouts = [
        [7449, 7510], 
        [7897, 7974],
        [8275, 8330],
        [9832, 9880],
        [10037, 10067],
        [10308, 10359],
        [20121, 20152],
        [9346, 9431], # weird int
        [9433, 9529], # CHASE
        [10118, 10202],
        [13313, 13411], # high speed chase
    ]
if curr_species == 'Dyak': 
    #start_frame = 16200 #14900
    #stop_frame = 17100 #15750
    #start_frame, stop_frame = yak_bouts[-1] #= 14900
    bouts = yak_bouts
elif curr_species == 'Dmel':
    #start_frame, stop_frame = mel_bouts[-1] #= 6060
    bouts = mel_bouts
elif curr_species == 'Dele':
    bouts = ele_bouts
    
#start_frame = 14900
#stop_frame = 15750


#%% # 
import cv2

def plot_extracted_objects(cap, ix, crop=False,
                           ratio_for_kernel=20, thresh_value=20, counter_ix=5,
                           minR_frac=0.05, maxR_frac=0.5, testing=False):
    #ix = 6439
    cap.set(1, ix)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
    h, w = gray.shape 

    # Mask image
    #ratio_for_kernel = 5
    #thresh_value = 70
    mask_two = get_mask(gray, crop=crop, 
                        ratio_for_kernel=ratio_for_kernel,
                        thresh_value=thresh_value, use_adaptive=False,
                        counter_ix=counter_ix, minR_frac=minR_frac, maxR_frac=maxR_frac
    )
    
    # 6. Build the final RGBA: copy original RGB into channels 0–2, and use mask_two as alpha
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba[..., 3]  = mask_two
    
    if testing:
        plt.figure()
        plt.imshow(rgba)

    return rgba

def video_frames_to_rgba(cap, frame_range, crop=False,
                         #min_dist=5, param1=200, param2=1, minR=None, maxR=None,
                         ratio_for_kernel=10, thresh_value=20, counter_ix=2,
                         minR_frac=0.05, maxR_frac=0.5):
    '''
    Load each frame, get mask for flies, return as list of RGBA images (alpha set my mask)
    '''
    rgba_list = []
    for ix in frame_range:
        #ix = frame_range[0]  # Start with the first frame
        # Set video to the first frame 
    #%
        #% Test here 
        #ix = 7504 #frame_range[0]
        #ix = 7515 
        #ix = frame_range[5] # 0, 1, 5
        #ix = 7454
        #print(ix)
        #ix = 6439
        rgba = plot_extracted_objects(cap, ix, crop=crop,
                                       ratio_for_kernel=ratio_for_kernel,
                                       thresh_value=thresh_value, counter_ix=counter_ix,
                                       minR_frac=minR_frac, maxR_frac=maxR_frac, 
                                       testing=False)
        #%
        # 7. Flatten the RGBA to a 2D array if want on white background
        # outim = flatten_rgba(rgba, background='white')
        # Append to the list
        rgba_list.append(rgba)    

    return rgba_list
#%%
def crop_inside_circle(img, dp=1.2, min_dist=None, param1=100, param2=30, 
                       min_radius=None, max_radius=None):
    """
    Detect the dominant circle (even if only a portion is visible) 
    and return a version of `img` where everything outside that circle 
    is zeroed out (i.e. “cropped” to the inside of the circle). 

    Args:
        img:       Input image, either single‐channel (H×W) or 3-channel (H×W×3), dtype=uint8.
        dp:        Inverse ratio of the accumulator resolution to the image resolution for HoughCircles.
        min_dist:  Minimum distance between the centers of the detected circles. Lower to get very close, touching circles
                   If None, defaults to min(h,w)/2 so it only finds one circle. 
        param1:    The higher threshold of the two passed to Canny edge detector (the lower is half).
                    Larger values will require stronger edges to detect; if vv bright, leave at 100-200, if fuzzy, lower to 50-75.
        param2:    The accumulator threshold for the circle centers at the detection stage. Lower it to get fainter circles.
                   Smaller → more (possibly spurious) detections; larger → fewer detections.
        min_radius: Minimum radius to search for (in pixels). If None, defaults to int(min(h,w)*0.3).
        max_radius: Maximum radius to search for (in pixels). If None, defaults to int(min(h,w)*0.6).

    Returns:
        masked_img:  An array with the exact same shape as `img`, dtype=uint8, but
                     with all pixels outside the detected circle set to 0.  
                     If `img` was grayscale, `masked_img` is grayscale; if `img` was color,
                     `masked_img` is color, channels 0–2 copied, but all outside‐circle pixels zeroed.
        (cx, cy, r): The detected circle’s center (x,y) and radius r (all integers).

    Raises:
        RuntimeError if no circle can be detected.
    """
    # Step 1: Convert to grayscale if necessary
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    h, w = gray.shape

    # Step 2: Provide sensible defaults for min_dist, min_radius, max_radius
    if min_dist is None:
        min_dist = min(h, w) // 2  # ensure Hough finds only one big circle
    if min_radius is None:
        min_radius = int(min(h, w) * 0.3)
    if max_radius is None:
        max_radius = int(min(h, w) * 0.6)

    # Step 3: Apply a small median blur to reduce noise (improves circle detection)
    blurred = cv2.medianBlur(gray, 5)

    # Step 4: Run HoughCircles to find candidate circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None or len(circles[0]) == 0:
        raise RuntimeError("No circle could be detected in the image.")

    # Step 5: Pick the circle with the largest radius
    circles = np.uint16(np.around(circles[0]))
    best = max(circles, key=lambda c: c[2])
    cx, cy, r = int(best[0]), int(best[1]), int(best[2])

    # Step 6: Build a binary mask (H×W) where pixels inside the circle = True
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    inside_mask = dist_sq <= (r * r)  # boolean mask

    # Step 7: Apply mask to the original image
    if len(img.shape) == 2:
        # Grayscale input
        masked_img = np.zeros_like(img)
        masked_img[inside_mask] = img[inside_mask]
    else:
        # Color input: copy each channel only where inside_mask is True
        masked_img = np.zeros_like(img)
        for ch in range(img.shape[2]):
            channel = img[..., ch]
            masked_img[..., ch][inside_mask] = channel[inside_mask]

    return masked_img, (cx, cy, r)

#%% TEST ZONE
# Take a frame and test image thresholing
# ---------------------------------------------------
testing =  True
ix = 6451
if testing == True:
    #ix = 7515 #frame_range[50]
    #ix = frame_range[0]
    cap.set(1, ix)
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
    h, w = gray.shape 

    # PROJECTOR: 
    #ratio_for_kernel = 5 #12
    #thresh_value = 70
    # ------
    # FREE-BEHAVIOR:
    ratio_for_kernel =5
    thresh_value = 30 
    #% ----
    min_dist = 1
    param1 = 300
    param2 = 1
    #minR = int(min(gray.shape)*0.05)
    #maxR = int(min(gray.shape)*0.3)

    h, w = gray.shape

    #minR = int(min(gray.shape)*0.05)
    #maxR = int(max(gray.shape)*5)
    #cropped_img, (cx, cy, r) = crop_inside_circle(gray, min_dist=min_dist, param1=param1, param2=2,
    #                                              min_radius=minR, max_radius=maxR)
    
    # 1) Build a “smaller” morphological kernel than before:
    k = int(min(h, w) / ratio_for_kernel)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 2) BlackHat → brighten the flies
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3) threshold
    _, thresh = cv2.threshold(
            blackhat, thresh_value, 255,
            cv2.THRESH_BINARY
        )

    # 3z) crop
    # 2) Crop/mask everything outside the detected circle
    min_dist = 1
    param1 = 300
    param2 = 100
    minR_frac = 0.05
    maxR_frac =0.75
    minR = int(min(gray.shape)*minR_frac)
    maxR = int(min(gray.shape)*maxR_frac)
    cropped_img, (cx, cy, r) = crop_inside_circle(thresh, min_dist=min_dist, param1=param1, param2=2,
                                                   min_radius=minR, max_radius=maxR)
    #thresh = cropped_img
 
    fig, axn = plt.subplots(1,3)
    ax=axn[0]
    ax.imshow(blackhat, cmap='gray')
    ax = axn[1]
    ax.imshow(thresh, cmap='gray')
    ax=axn[2]
    ax.imshow(cropped_img, cmap='gray')     
    #%%
    counter_ix = 4
    thresh = cropped_img.copy()
    
    # 4) Small morphological opening to remove residual specks
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

    # 5) Find all contours and keep the two largest areas
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if len(areas) > 0:
        sorted_idx = np.argsort(areas)[::-1]
        keep_idx = sorted_idx[:counter_ix]
    else:
        keep_idx = []

    # 6) Build the final mask
    mask_two = np.zeros_like(gray, dtype=np.uint8)
    for i in keep_idx:
        cv2.drawContours(mask_two, contours, i, 255, thickness=cv2.FILLED)

    fig, axn = plt.subplots(1,3)
    ax=axn[0]
    ax.imshow(blackhat, cmap='gray')
    ax = axn[1]
    ax.imshow(thresh, cmap='gray')
    ax=axn[2]
    ax.imshow(mask_two, cmap='gray') 

#%%
minR_frac = 0.05
maxR_frac =0.75
plot_extracted_objects(cap, ix, crop=False, 
                        ratio_for_kernel=ratio_for_kernel,
                        thresh_value=thresh_value, counter_ix=counter_ix,
                        minR_frac=minR_frac, maxR_frac=maxR_frac, testing=True)


#%%k 
def get_mask(gray, ratio_for_kernel=10, thresh_value=20, use_adaptive=False,
             counter_ix=2, crop=False, maxR_frac=0.5, minR_frac=0.05):
    """
    Return a binary mask (uint8, 0 or 255) that contains exactly the two largest dark blobs
    (flies) on a possibly non‐uniform circular background.

    Parameters:
    ----------
    gray : np.ndarray
        Single‐channel (H × W) uint8 grayscale input.
    ratio_for_kernel : int
        Denominator of min(h,w) to choose the BlackHat kernel radius. 
        E.g. ratio_for_kernel=20 → kernel ≈ min(h,w)/20.  Smaller → smaller kernel.
    thresh_value : int or None
        If not None, use cv2.threshold(blackhat, thresh_value, 255, THRESH_BINARY).  
        If None and use_adaptive=False, an Otsu threshold is applied automatically
        to the BlackHat output.
    use_adaptive : bool
        If True, do adaptiveThreshold on blackhat instead of a fixed/otsu threshold.
    """

    h, w = gray.shape

    # 1) Build a “smaller” morphological kernel than before:
    k = int(min(h, w) / ratio_for_kernel)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 2) BlackHat → brighten the flies
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # ---------------------------------------------------------------
    # 3a) Option A: Fixed/HARD threshold on blackhat (fast):
    if not use_adaptive:
        if thresh_value is None:
            # Use Otsu’s method if no explicit thresh_value is given
            _, thresh = cv2.threshold(
                blackhat, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, thresh = cv2.threshold(
                blackhat, thresh_value, 255,
                cv2.THRESH_BINARY
            )
    # ---------------------------------------------------------------
    # 3b) Option B: ADAPTIVE threshold (tolerates uneven lighting):
    else:
        # blockSize must be odd and roughly a bit bigger than the fly diameter
        blockSize = max(3, (k // 2) * 2 + 1)
        thresh = cv2.adaptiveThreshold(
            blackhat,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=blockSize,
            C=5
        )
    # ---------------------------------------------------------------

    if crop:
        # 2) Crop/mask everything outside the detected circle
        min_dist = 1
        param1 = 300
        param2 = 2
        minR = int(min(gray.shape)*minR_frac)
        maxR = int(min(gray.shape)*maxR_frac) #0.44)
        cropped_img, (cx, cy, r) = crop_inside_circle(thresh, min_dist=min_dist, param1=param1, param2=param2,
                                                    min_radius=minR, max_radius=maxR)
        thresh = cropped_img.copy()
    # --------------------------------------------------------------------
    
    # 4) Small morphological opening to remove residual specks
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

    # 5) Find all contours and keep the two largest areas
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(cnt) for cnt in contours]
    if len(areas) > 0:
        sorted_idx = np.argsort(areas)[::-1]
        keep_idx = sorted_idx[:counter_ix]
    else:
        keep_idx = []

    # 6) Build the final mask
    mask_two = np.zeros_like(gray, dtype=np.uint8)
    for i in keep_idx:
        cv2.drawContours(mask_two, contours, i, 255, thickness=cv2.FILLED)

    return mask_two


# def get_mask(gray): 
#     h, w = gray.shape
#     # 2. Perform a BlackHat (closing – original) with a large elliptical kernel.
#     #    This “fills in” the two dark flies so the result is nearly zero everywhere
#     #    except for bright spots at the flies’ locations.
#     #    We pick the kernel size to be about 1/10th of the smaller image dimension.
#     k = int(min(h, w) / 10)
#     if k % 2 == 0:
#         k += 1
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
#     blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
#     
#     # 3. Threshold that BlackHat result with a simple fixed threshold (fast, no Otsu).
#     #    We choose 20 so that the two bright fly regions turn white (255), while
#     #    almost everything else (background) remains zero.
#     _, thresh = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
# 
#     # 4. Remove tiny specks via a 3×3 morphological opening
#     kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
# 
#     # 5. Find all contours in the cleaned mask, then keep only the two largest
#     contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     areas = [cv2.contourArea(c) for c in contours]
#     if len(areas) > 0:
#         sorted_idx = np.argsort(areas)[::-1]
#         keep_idx = sorted_idx[:2]
#     else:
#         keep_idx = []
# 
#     mask_two = np.zeros_like(gray)
#     for i in keep_idx:
#         cv2.drawContours(mask_two, contours, i, 255, thickness=cv2.FILLED)
# 
#     return mask_two

def extract_two_darkest_blobs_from_rgba(rgba, thresh_val=240, 
                                    opening_kernel_size=5, min_blob_area=50):
    """
    Given an RGBA image (H×W×4), use the alpha channel to restrict to “foreground,” 
    then find and return a new RGBA image in which only the two darkest connected 
    blobs remain (all other pixels are made fully transparent).

    Parameters
    ----------
    rgba : np.ndarray
        Input image of shape (H, W, 4), dtype=uint8.  Channels 0–2 are BGR color; 
        channel 3 is alpha (0=transparent, 255=opaque).
    thresh_val : int, optional
        All pixels with grayscale intensity < thresh_val (0..255) are considered 
        “dark” and become candidates.  Default: 200.
    opening_kernel_size : int, optional
        Size of the elliptical kernel for morphological opening to remove noise. 
        Default: 5.
    min_blob_area : int, optional
        Ignore any connected component smaller than this area (in pixels). 
        Default: 50.

    Returns
    -------
    result_rgba : np.ndarray (H, W, 4)
        A new RGBA image (uint8).  Only the two darkest blobs from the original remain
        with their original RGB and alpha values; all other pixels are set to alpha=0.

    Raises
    ------
    RuntimeError
        If fewer than two valid blobs are found among the “foreground” pixels.
    """

    # 1) Split channels and build a masked‐grayscale image
    # -----------------------------------------------------
    # Extract BGR and alpha
    b_channel, g_channel, r_channel, a_channel = cv2.split(rgba)

    # Convert the color (BGR) to a single‐channel grayscale
    bgr = cv2.merge([b_channel, g_channel, r_channel])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Zero out (or set to white) any pixel that was fully transparent in the original
    # so that background pixels never count as “dark” blobs
    gray[a_channel == 0] = 255  # treat transparent background as white

    # 2) Threshold to pick up all “dark” foreground candidates
    # -------------------------------------------------------
    # Inverse binary: pixels < thresh_val → 255 (foreground), else 0
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 3) Remove tiny specks with morphological opening
    # ------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4) Find connected components (contours) in that cleaned mask
    # ------------------------------------------------------------
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) For each contour, measure area and “darkness” (mean grayscale value)
    # ----------------------------------------------------------------------
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area:
            continue

        # Build a mask for this single contour
        single_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Compute mean intensity over the contour region in the original gray
        mean_val = cv2.mean(gray, mask=single_mask)[0]  # returns (mean,_,_,_)
        candidates.append((mean_val, area, cnt))

    if len(candidates) < 2:
        raise RuntimeError(f"Only found {len(candidates)} valid dark blob(s); need at least 2.")

    # 6) Sort by mean_val ascending (darkest first).  If tie, larger area wins.
    # -------------------------------------------------------------------------
    candidates.sort(key=lambda x: (x[0], -x[1]))
    # Keep the two darkest
    darkest_two = [candidates[0][2], candidates[1][2]]

    # 7) Build final RGBA result: keep only those two contours
    # ---------------------------------------------------------
    H, W = gray.shape
    result_rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # For convenience, copy original RGBA into the result, but zero out everything first
    # Then “paint back” only the pixels inside the two selected contours
    # by copying from the original RGBA.
    # (Alternatively, could copy only from BGR and set alpha=255 only inside those two.)
    # We’ll do: result_rgba = 0; then for each pixel in the two contours, copy rgba[pix].

    # Build a mask (uint8) where the two contours are white (255)
    mask_two = np.zeros((H, W), dtype=np.uint8)
    for cnt in darkest_two:
        cv2.drawContours(mask_two, [cnt], -1, 255, thickness=cv2.FILLED)

    # Now copy B,G,R,A from original wherever mask_two==255
    for channel in range(4):
        result_rgba[..., channel][mask_two == 255] = rgba[..., channel][mask_two == 255]

    return result_rgba

def extract_two_darkest_blobs(img, thresh_val=240, opening_kernel_size=5, min_blob_area=50):
    """
    Given an input image (BGR or grayscale), find and return a new grayscale image
    that contains only the two darkest connected blobs, with all other pixels set to 0.

    Parameters
    ----------
    img : np.ndarray
        Input image. Can be either a 2D grayscale array (H×W) or a 3-channel BGR array (H×W×3).
    thresh_val : int, optional
        Threshold cutoff for binary inversion. Any grayscale pixel < thresh_val will be
        considered “dark” and become foreground in the binary mask. Default is 240.
    opening_kernel_size : int, optional
        Diameter of the elliptical kernel used for morphological opening to remove small noise.
        Default is 5.
    min_blob_area : int, optional
        Ignore any connected component smaller than this area in pixels. Default is 50.

    Returns
    -------
    extracted : np.ndarray (dtype=uint8)
        A single-channel (H×W) grayscale image in which only the two darkest blobs from the
        original remain (with their original gray intensities); all other pixels are set to 0.

    Raises
    ------
    RuntimeError
        If fewer than two valid dark blobs are found in the image.
    """

    # 1) Split channels and build a masked‐grayscale image
    # -----------------------------------------------------
    # Extract BGR and alpha
    b_channel, g_channel, r_channel, a_channel = cv2.split(rgba)

    # Convert the color (BGR) to a single‐channel grayscale
    bgr = cv2.merge([b_channel, g_channel, r_channel])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Zero out (or set to white) any pixel that was fully transparent in the original
    # so that background pixels never count as “dark” blobs
    gray[a_channel == 0] = 255  # treat transparent background as white

    # 2) Threshold to isolate all dark‐ish pixels (binary inverse)
    #    Any pixel < thresh_val becomes 255 (candidate foreground), else 0.
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 3) Morphological opening to remove very small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4) Find all connected components (contours) in the cleaned mask
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5) For each contour, compute its area and ignore if too small
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area:
            continue
        valid.append((area, cnt))

    # If fewer than two valid blobs remain, we cannot extract two darkest
    if len(valid) < 2:
        raise RuntimeError(f"Found only {len(valid)} valid dark blob(s); need at least 2.")

    # 6) Sort valid blobs by area (descending) and keep only the two largest
    valid.sort(key=lambda x: x[0], reverse=True)
    keep_contours = [valid[0][1], valid[1][1]]

    # 7) Build a binary mask containing just those two contours
    mask_two = np.zeros_like(gray, dtype=np.uint8)
    for cnt in keep_contours:
        cv2.drawContours(mask_two, [cnt], -1, 255, thickness=cv2.FILLED)

    # 8) Extract the two blobs from the original gray image
    extracted = cv2.bitwise_and(gray, gray, mask=mask_two)

    return extracted

def extract_two_darkest_blobs_v2(img):
    # 1. Apply a loose threshold to capture all dark-ish regions (droplet, insect, rim-shadows, debris)
    #    We use THRESH_BINARY_INV with a high cutoff so all pixels darker than 240 become white in the mask.
    _, thresh_loose = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

    # 2. Morphological opening to remove small noise (debris specks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh_loose, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Find connected components in the cleaned mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    # 4. For each component (ignoring the background label 0), compute the mean intensity
    components = []
    for lbl in range(1, num_labels):
        mask_lbl = (labels == lbl)
        if np.sum(mask_lbl) == 0:
            continue
        mean_intensity = np.mean(img[mask_lbl])
        area = stats[lbl, cv2.CC_STAT_AREA]
        components.append((lbl, mean_intensity, area))

    # 5. Sort components by mean_intensity ascending (darkest first), and pick top 2
    components_sorted = sorted(components, key=lambda x: x[1])  # sort by mean intensity
    keep_labels = [c[0] for c in components_sorted[:2]]

    # 6. Build a final mask containing only those two labels
    mask_two = np.zeros_like(img, dtype=np.uint8)
    for lbl in keep_labels:
        mask_two[labels == lbl] = 255

    # 7. Extract the two darkest blobs from the original grayscale
    extracted = cv2.bitwise_and(img, img, mask=mask_two)
    return extracted
    
def flatten_rgba(rgba, background='white'):
    """
    Flattens the RGBA image to a 2D array.
    """
    # 2. Split into B, G, R, and A channels
    b, g, r, a = cv2.split(rgba)

    # 3. Convert the color channels back to a single grayscale image
    gray_full = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)

    # 4. Build a “masked_gray” where we keep the grayscale value only if alpha > 0,
    #    otherwise set to zero.
    masked_gray = np.zeros_like(gray_full)
    masked_gray[a > 0] = gray_full[a > 0]
    if background=='white':
        masked_gray[masked_gray == 0] = 255  # Set background to white

    return masked_gray

def rgba_to_masked_gray(rgba):
    """
    Given an RGBA uint8 image of shape (H,W,4), return a 2D masked array
    `ma_gray` of shape (H,W) such that:
      - ma_gray.data[y,x] = the grayscale intensity (0..255)
      - ma_gray.mask[y,x] = True  if that pixel was transparent in RGBA
                         = False if that pixel belonged to foreground
    When you later call `imshow(ma_gray, cmap=...)`, masked pixels will be
    transparent and the rest will be colored by the colormap.
    """
    # Split into B,G,R,A channels
    b, g, r, a = cv2.split(rgba)

    # Reconstruct a single-channel grayscale from the original RGB (BGR→Gray)
    # (Since our extracted objects were always grayscale to begin with, this
    #  simply recovers their gray intensities.)
    gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)

    # Build a boolean mask: True where alpha==0 (background), False where alpha>0 (foreground)
    # Note: a is dtype=uint8 in [0..255], so we compare to zero.
    bg_mask = (a == 0)

    # Create a NumPy masked‐array.  Mask = True ⇒ “hide/transparent.”
    ma_gray = np.ma.array(gray, mask=bg_mask)

    return ma_gray


def separate_objects_by_centroid(rgba_list):
    """
    Separate objects in a list of RGBA images by their centroids.
    Returns two lists: one for each object.
    """
    
    # Lists to store the separated single-object images
    list_A = []
    list_B = []

    # To keep track of previous centroids for A and B
    prev_centroid_A = None
    prev_centroid_B = None

    for idx, rgba in enumerate(rgba_list):
        # 1) Split out the alpha channel & find contours on it
        b_channel, g_channel, r_channel, alpha_channel = cv2.split(rgba)
        # Convert alpha to binary mask (0 or 255)
        _, alpha_binary = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        
        # Find external contours (each contour corresponds to one object)
        contours, _ = cv2.findContours(alpha_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If fewer than 2 contours, skip or handle accordingly
        if len(contours) < 2:
            raise ValueError(f"Frame {idx} has fewer than 2 detected objects.")
        
        # Compute centroid and masked image for each contour
        object_info = []  # Will hold tuples of (centroid, single_obj_rgba)
        for cnt in contours:
            # Compute centroid using moments
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid = (cx, cy)
            
            # Build a mask for this single contour
            mask_single = np.zeros_like(alpha_binary)
            cv2.drawContours(mask_single, [cnt], -1, 255, thickness=cv2.FILLED)
            
            # Apply that mask to the RGBA image to extract the single-object RGBA
            single_rgba = np.zeros_like(rgba)
            single_rgba[..., 0] = b_channel * (mask_single // 255)
            single_rgba[..., 1] = g_channel * (mask_single // 255)
            single_rgba[..., 2] = r_channel * (mask_single // 255)
            single_rgba[..., 3] = mask_single
            
            object_info.append((centroid, single_rgba))
        
        # If more than 2 objects (unlikely), sort by area or pick the two largest:
        if len(object_info) > 2:
            # Sort contours by area descending, keep top 2
            areas = [cv2.contourArea(cnt) for cnt in contours]
            sorted_idx = np.argsort(areas)[::-1]
            top_two_info = [object_info[i] for i in sorted_idx[:2]]
            object_info = top_two_info
        
        # Now object_info has exactly two entries: [(centroid1, img1), (centroid2, img2)]
        (c1, img1), (c2, img2) = object_info
        
        if idx == 0:
            # For the first frame, assign based on x-coordinate (left->A, right->B)
            if c1[0] < c2[0]:
                list_A.append(img1)
                list_B.append(img2)
                prev_centroid_A = c1
                prev_centroid_B = c2
            else:
                list_A.append(img2)
                list_B.append(img1)
                prev_centroid_A = c2
                prev_centroid_B = c1
        else:
            # Compute distances from current centroids to previous
            dist1_to_A = np.hypot(c1[0] - prev_centroid_A[0], c1[1] - prev_centroid_A[1])
            dist1_to_B = np.hypot(c1[0] - prev_centroid_B[0], c1[1] - prev_centroid_B[1])
            dist2_to_A = np.hypot(c2[0] - prev_centroid_A[0], c2[1] - prev_centroid_A[1])
            dist2_to_B = np.hypot(c2[0] - prev_centroid_B[0], c2[1] - prev_centroid_B[1])
            
            # Assign the object whose centroid is closer to prev_centroid_A into A, the other into B
            if dist1_to_A + dist2_to_B < dist1_to_B + dist2_to_A:
                # Object 1 goes to A, Object 2 goes to B
                list_A.append(img1)
                list_B.append(img2)
                prev_centroid_A = c1
                prev_centroid_B = c2
            else:
                # Object  goes to A, Object 1 goes to B
                list_A.append(img2)
                list_B.append(img1)
                prev_centroid_A = c2
                prev_centroid_B = c1

    return list_A, list_B

#%%
import pickle as pkl

# Try loading frames 
frame_fpath = os.path.join(figdir, 'frames_{}.pkl'.format(acquisition))
print(frame_fpath)
create_new = False 

if create_new is False and os.path.exists(frame_fpath):
    with open(frame_fpath, 'rb') as f:
        frame_dict = pkl.load(f)
    print('Loaded existing frames from:', frame_fpath)
else:
    create_new = True 
print(create_new)
    
#%%a
#crop = False
if assay == '2d_projector':
    print("2d")
    ratio_for_kernel = 5# 10
    thresh_value = 70
    counter_ix = 5
    minR_frac=0.05
    maxR_frac=0.5
    use_adaptive = False    
    # Cropping parms, to remove bright edge of aren
    min_dist=1; param1=300; param2=2;
    #minR=None, maxR=None,
    crop=True
else:
    print("free behavior")
    ratio_for_kernel = 5 #20# 10
    thresh_value = 30 #50
    counter_ix = 5
    minR_frac=0.05
    maxR_frac=0.75
    use_adaptive = False    
    # Cropping parms, to remove bright edge of aren
    min_dist=1; param1=300; param2=2;
    #minR=None, maxR=None,
    crop=True

#%%
if create_new:
    #%%
    # Load video
    found_vidpaths = glob.glob(os.path.join(viddir, '*.avi'))
    if len(found_vidpaths) > 1:
        for vi, v in enumerate(found_vidpaths):
            print(os.path.split(v)[-1])
    vidpath = found_vidpaths[0] #[3]  # Assuming you want the first video file
    print(vidpath)

    # Load video
    cap = cv2.VideoCapture(vidpath)
    #success, image = vidcap.read()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(width, height, n_frames)
    #%%
    frame_dict = {}
    for ix, (start_frame, stop_frame) in enumerate(bouts):
        #%
        #start_frame, stop_frame = bouts[-3]
        # Make a list of frames to process
        # -----------
        #start_frame, stop_frame = bouts[0]
        #%
        #ix = 0 
        start_fram, stop_frame = bouts[ix]
        print("Processing frames {} to {}".format(start_frame, stop_frame))
        print(ix)
        frame_range = np.arange(start_frame, stop_frame) #, interval)
        rgba_list = video_frames_to_rgba(cap, frame_range, crop=crop,
                                        ratio_for_kernel = ratio_for_kernel, #12
                                        thresh_value = thresh_value,
                                        counter_ix = counter_ix,
                                        minR_frac=minR_frac, maxR_frac=maxR_frac) 
        #if assay == '2d_projector':
        print("Extracting")
        # For 2D projector, we need to crop the image to remove the bright edge
        extracted_rgba = []
        for i, rgba in enumerate(rgba_list):
            # Extract the two darkest blobs from each RGBA image
            # This will return a new RGBA image with only the two darkest blobs
            extracted_ = extract_two_darkest_blobs_from_rgba(rgba, 
                                    thresh_val=240, 
                                    opening_kernel_size=5, min_blob_area=50)
            extracted_rgba.append(extracted_)
        #extracted_rgba = [extract_two_darkest_blobs_from_rgba(rgba, thresh_val=240) for rgba in rgba_list] 
        #ext = extract_two_darkest_blobs_from_rgba(rgba_list[0], thresh_val=240, 
        #                                opening_kernel_size=5, min_blob_area=50)
        #plt.figure()
        #plt.imshow(ext, cmap='gray')
        #else:
        #    # For 3D projector, we can use the original RGBA images directly
        #    extracted_rgba = rgba_list 
        #%
        # Separate male and female
        # Now `list_A` and `list_B` each contain 10 single-object RGBA images, consistently tracked
        # left is A, right is B for first frame
        list_A, list_B = separate_objects_by_centroid(extracted_rgba)    
        #%
        #plt.figure()
        #plt.imshow(list_A[5]); plt.imshow(list_B[5])
        #% 
        frame_dict.update({ix: [list_A, list_B]})
        #%
    with open(frame_fpath, 'wb') as f:
        pkl.dump(frame_dict, f)
        
    
#%%
if curr_species == 'Dyak':
    male_is_b = [0, 1, 5]
    plot_interval = 50
elif curr_species == 'Dmel':
    male_is_b = [0, 1, 2 ]
    plot_interval = 50
elif curr_species == 'Dele':
    if acquisition == '20240814-0925_fly1_Dele-HK_WT_6do_gh':
        male_is_b = [1, 2]
    elif acquisition == '20240814-1023_fly2_Dele-HK_WT_6do_gh':
        # These are males that are black, but should be red 
        male_is_b = [1, 3, 6, 7, 10, 11, 12, 14, 15, 16] 
        #male_is_b = [0, 2, 4,  5, 8, 9, 13]
    elif acquisition == '20240214-1025_f1_Dele-wt_5do_sh_prj10_sz10x10':
        male_is_b = [0, 4, 6, 7, 8, ]
    else:
        male_is_b = []
    plot_interval = 10
    
print(male_is_b)
# A is male, is gray
male_is_a = [i for i, v in enumerate(bouts) if i not in male_is_b]

plot_interval=10

for (start_frame, end_frame), (ix, (list_A, list_B)) in zip(bouts, frame_dict.items()):
    #% testing:
    plot_interval = 5
    #start_frame, end_frame = bouts[0]
    #list_A, list_b = frame_dict[7454]
    print(start_frame, end_frame, ix)
    #%
    # Convert lists to numpy arrays
    if ix == 10:
        half_len = int(round(len(list_A)/2))
        #list_A = list_A[half_len:]
        #list_b = list_B[half_len:]
        ma_A = [rgba_to_masked_gray(rgba) for rgba in list_A[half_len::plot_interval]]
        ma_B = [rgba_to_masked_gray(rgba) for rgba in list_B[half_len::plot_interval]]
    else:
        ma_A = [rgba_to_masked_gray(rgba) for rgba in list_A[0::plot_interval]]
        ma_B = [rgba_to_masked_gray(rgba) for rgba in list_B[0::plot_interval]]

    # Plot
    alpha_values = np.linspace(0.2, 1, len(ma_A))
    fig, ax = plt.subplots(figsize=(4,4))
    
    a_cmap = 'Reds_r' if ix in male_is_a else 'gray'
    b_cmap = 'gray' if a_cmap == 'Reds_r' else 'Reds_r'
    for a, b, curr_alpha in zip(ma_A, ma_B, alpha_values):
        # Display the images with some transparency
        ax.imshow(a, cmap=a_cmap, alpha=curr_alpha, interpolation='none')
        ax.imshow(b, cmap=b_cmap, alpha=curr_alpha, interpolation='none')
    ax.set_xlim([0, 1200])
    ax.set_ylim([1200, 0])  # Invert y-axis to match video coordinates
    ax.axis('off')
    fig.text(0.1, 0.95, ix)
    
    # Set cbar title
    figname = 'overlaid_{}__{}_fr{}-{}'.format(curr_species, acquisition, start_frame, stop_frame)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)), 
                dpi=300, bbox_inches='tight')

#%%

#%%
# ===========================================================================
# Load processed data
df = util.combine_flytracker_data(acquisition, viddir)
df.head()

#%% 
# plot trajectories symbollically
# ===========================================================================
interval = 12

# Define marker size and line length
marker_length = 20  # length of line markers
circle_size = 8      # size of circle markers
female_size = 30
alpha=0.75
use_arrow = True
female_cmap = 'Greys'
male_cmap = 'viridis'
male_alpha=0.75
female_alpha=0.75

add_female_arrow = True

#%%
interval=5
for (start_frame, stop_frame) in bouts:
    #%
    #start_frame, stop_frame = bouts[0] 
    plotd = df[df['frame'].between(start_frame, stop_frame)].copy()
    plotd['ori'] = -1 * plotd['ori']

    # Calculate colors
    palette = sns.color_palette(male_cmap, plotd['frame'].nunique())
    color_mapping = dict(zip(plotd['frame'].unique(), palette))

    male_pos = plotd[plotd['id'] == 0].copy()
    female_pos = plotd[plotd['id'] == 1].copy()

    fig, ax = plt.subplots(figsize=(3,3))
    if use_arrow:
        for curr_id, plotd_ in plotd.iloc[0::interval].groupby('id'):
            # Calculate direction vectors
            u = np.cos(plotd_['ori'])
            v = np.sin(plotd_['ori'])
            colors = plotd_['frame'].map(color_mapping)

            if curr_id == 0:
                # plot male
                plt.quiver(plotd_['pos_x'].iloc[0::interval], plotd_['pos_y'].iloc[0::interval], 
                           u.iloc[0::interval], v.iloc[0::interval], 
                           angles='xy', pivot='middle', 
                    scale_units='xy', scale=0.02, color=colors, width=0.02,
                    headaxislength=8, headlength=10, headwidth=5, alpha=male_alpha)
                # plot a line
                ax.plot(plotd_['pos_x'], plotd_['pos_y'], color='gray', linewidth=0.5)
            else:
                # plot female
                if add_female_arrow:
                    plt.quiver([plotd_['pos_x'].iloc[0], plotd_['pos_x'].iloc[-1]],
                            [plotd_['pos_y'].iloc[0], plotd_['pos_y'].iloc[-1]], 
                            [u.iloc[0], u.iloc[-1]], [v.iloc[0], v.iloc[-1]], 
                            angles='xy', pivot='middle', 
                        scale_units='xy', scale=0.02, color='gray', width=0.004,
                        headaxislength=8, headlength=8, headwidth=5, alpha=male_alpha)
                # plot dots
                plot_ixs = np.arange(1, len(plotd_)-2, interval)
                sns.scatterplot(data=plotd_.iloc[plot_ixs], x='pos_x', y='pos_y', ax=ax,
                                #color='gray', marker='o',
                                hue='frame', palette=female_cmap, marker='o',
                                s=female_size, edgecolor='k', linewidth=0.1,
                                legend=False, alpha=1)
                # plot a line
                ax.plot(plotd_['pos_x'], plotd_['pos_y'], color='gray', linewidth=0.5)
    else:
        for r, row in male_pos.iloc[0::interval].iterrows():
            x, y, angle_rad, fr = row['pos_x'], row['pos_y'], row['ori'], row['frame']
            color = color_mapping[fr]

            # Compute line coordinates (rotated line segment)
            dx = marker_length * np.cos(angle_rad)
            dy = marker_length * np.sin(angle_rad)

            # Plot line ("|")
            ax.plot([x, x + dx], [y, y + dy], color=color, linewidth=1)

            # Plot circle at the starting point to indicate directionality
            ax.scatter(x + dx, y + dy, color=color, s=circle_size, 
                    edgecolor='none', zorder=3, alpha=male_alpha)

            female_row = female_pos[female_pos['frame'] == fr]  
            # Plot circle at the starting point to indicate directionality
            ax.scatter(female_row['pos_x'], female_row['pos_y'], alpha=female_alpha,
                    color=color, s=female_size, edgecolor='none', zorder=3)

    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1200])
    ax.set_aspect('equal')  
    ax.invert_yaxis()
    ax.axis('off')

    # Draw colorbar
    norm = plt.Normalize(start_frame, stop_frame)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, shrink=0.5)
    cbar.set_label('Time $\longrightarrow$', rotation=90, labelpad=5)
    # Remove ticks and tick labels from colorbar
    cbar.ax.tick_params(size=0)
    cbar.ax.set_yticklabels([])
    # Set cbar title
    figname = 'trajectory_{}__{}_fr{}-{}'.format( curr_species, acquisition, start_frame, stop_frame)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')

# %%

# %%
fig, ax = plt.subplots()

# Set rotation of marker by df column 'ori'

ax.set_aspect('equal')
ax.invert_yaxis()
#sns.scatterplot(data=plotd_, x='pos_x', y='pos_y', ax=ax,
#                hue='frame', palette='viridis', marker='|',
#                transform=plt.gca()._transforms.get_affine().rotate_deg(plotd_['ori']) + plt.gca().transData)
                
