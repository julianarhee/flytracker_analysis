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

basedir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'
figdir = os.path.join(basedir, 'plot_example_trajectory')

if not os.path.exists(figdir):
    os.makedirs(figdir)
print('figdir:', figdir)

# %%

curr_species = 'Dele'
# --------------------------------------------------
srcdir = '/Volumes/Giacomo/free_behavior_data'
if curr_species == 'Dyak':
    acquisition = '20240116-1015-fly1-yakWT_4do_sh_yakWT_4do_gh'
elif curr_species == 'Dmel':
    acquisition = '20240119-1020-fly3-melWT_4do_sh_melWT_4do_gh'
elif curr_species == 'Dele':
    #acquisition = '20240814-0925_fly1_Dele-HK_WT_6do_gh'
    acquisition = '20240814-1023_fly2_Dele-HK_WT_6do_gh'
    srcdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/Caitlin_elehk_38mm'
viddir = os.path.join(srcdir, acquisition)
df = util.combine_flytracker_data(acquisition, viddir)
df.head()

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

def video_frames_to_rgba(cap, frame_range):
    '''
    Load each frame, get mask for flies, return as list of RGBA images (alpha set my mask)
    '''
    rgba_list = []
    for ix in frame_range:
        #ix = frame_range[0]  # Start with the first frame
        # Set video to the first frame
        cap.set(1, ix)
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #COLOR_BGR2RGB)
        h, w = gray.shape 
        
        # mask image     
        mask_two = get_mask(gray)
     
        # 6. Build the final RGBA: copy original RGB into channels 0–2, and use mask_two as alpha
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgba[..., 3]  = mask_two

        # 7. Flatten the RGBA to a 2D array if want on white background
        # outim = flatten_rgba(rgba, background='white')

        rgba_list.append(rgba)    

    return rgba_list


def get_mask(gray): 
    h, w = gray.shape
    # 2. Perform a BlackHat (closing – original) with a large elliptical kernel.
    #    This “fills in” the two dark flies so the result is nearly zero everywhere
    #    except for bright spots at the flies’ locations.
    #    We pick the kernel size to be about 1/10th of the smaller image dimension.
    k = int(min(h, w) / 10)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Threshold that BlackHat result with a simple fixed threshold (fast, no Otsu).
    #    We choose 20 so that the two bright fly regions turn white (255), while
    #    almost everything else (background) remains zero.
    _, thresh = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

    # 4. Remove tiny specks via a 3×3 morphological opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)

    # 5. Find all contours in the cleaned mask, then keep only the two largest
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) > 0:
        sorted_idx = np.argsort(areas)[::-1]
        keep_idx = sorted_idx[:2]
    else:
        keep_idx = []

    mask_two = np.zeros_like(gray)
    for i in keep_idx:
        cv2.drawContours(mask_two, contours, i, 255, thickness=cv2.FILLED)

    return mask_two

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
                # Object 2 goes to A, Object 1 goes to B
                list_A.append(img2)
                list_B.append(img1)
                prev_centroid_A = c2
                prev_centroid_B = c1

    return list_A, list_B

import pickle as pkl


# Try loading frames 
frame_fpath = os.path.join(figdir, 'frames_{}.pkl'.format(acquisition))
create_new = True

if create_new is False and os.path.exists(frame_fpath):
    with open(frame_fpath, 'rb') as f:
        frame_dict = pkl.load(f)
    print('Loaded existing frames from:', frame_fpath)
else:
    create_new = True 
print(create_new)
    
#%%
if create_new:
    # Load video
    found_vidpaths = glob.glob(os.path.join(viddir, '*.avi'))
    print(found_vidpaths)
    vidpath = found_vidpaths[0]  # Assuming you want the first video file

    # Load video
    cap = cv2.VideoCapture(vidpath)
    #success, image = vidcap.read()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(width, height, n_frames)
    #%
    frame_dict = {}
    for ix, (start_frame, stop_frame) in enumerate(bouts):
    #%
        # Make a list of frames to process
        # -----------
        #start_frame, stop_frame = bouts[0]
        frame_range = np.arange(start_frame, stop_frame) #, interval)
        rgba_list = video_frames_to_rgba(cap, frame_range)
        #%
        # Separate male and female
        # Now `list_A` and `list_B` each contain 10 single-object RGBA images, consistently tracked
        # left is A, right is B for first frame
        list_A, list_B = separate_objects_by_centroid(rgba_list)    
        
        frame_dict.update({ix: [list_A, list_B]})

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
    else:
        # These are males that are black, but should be red 
        male_is_b = [1, 3, 6, 7, 10, 11, 12, 14, 15, 16] 
        #male_is_b = [0, 2, 4,  5, 8, 9, 13]
    plot_interval = 10
    
print(male_is_b)
# A is male, is gray
male_is_a = [i for i, v in enumerate(bouts) if i not in male_is_b]

plot_interval=10

for (start_frame, end_frame), (ix, (list_A, list_B)) in zip(bouts, frame_dict.items()):
    print(start_frame, end_frame, ix)
    #%
    # Convert lists to numpy arrays
    ma_A = [rgba_to_masked_gray(rgba) for rgba in list_A[0::plot_interval]]
    ma_B = [rgba_to_masked_gray(rgba) for rgba in list_B[0::plot_interval]]

    # Plot
    alpha_values = np.linspace(0.1, 1, len(ma_A))
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
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)), dpi=300, bbox_inches='tight')

#%%



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
                
