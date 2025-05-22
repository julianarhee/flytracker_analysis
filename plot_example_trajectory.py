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

srcdir = '/Volumes/Giacomo/free_behavior_data'
#acquisition = '20240116-1015-fly1-yakWT_4do_sh_yakWT_4do_gh'
acquisition = '20240119-1020-fly3-melWT_4do_sh_melWT_4do_gh'

viddir = os.path.join(srcdir, acquisition)
df = util.combine_flytracker_data(acquisition, viddir)
df.head()

#%%a
curr_species = 'Dyak' if 'yak' in acquisition else 'Dmel'

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


if curr_species == 'Dyak': 
    #start_frame = 16200 #14900
    #stop_frame = 17100 #15750
    #start_frame, stop_frame = yak_bouts[-1] #= 14900
    bouts = yak_bouts
elif curr_species == 'Dmel':
    #start_frame, stop_frame = mel_bouts[-1] #= 6060
    bouts = mel_bouts
#start_frame = 14900
#stop_frame = 15750

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
                
