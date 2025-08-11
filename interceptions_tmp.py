#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Filename:
 # @ Create Time: 2025-05-19 12:01:01
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-05-19 12:01:23
 # @ Description:
 '''
#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import utils as util
import plotting as putil

# %%
# Set plotting
plot_style='white'
putil.set_sns_style(plot_style, min_fontsize=6)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%% 
# Find actions.mat files
viddir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/Caitlin_elehk_38mm'
found_action_mats = glob.glob(os.path.join(viddir, '20*ele*', '*', '*-actions.mat'))

print(len(found_action_mats))

actions = util.load_ft_actions(found_action_mats)
int_frames = actions[actions['action']=='interception'].copy()
int_frames['acquisition'].unique()


# %%
# Load processed data for select files
basedir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker'

found_processed_mats = glob.glob(os.path.join(basedir, 'processed_mats', '*ele*_df.pkl'))
#for i in processed_mats:
#    print(os.path.split(i)[-1]) 

find_these_acqs = actions['acquisition'].unique()
processed_mats = [i for i in found_processed_mats if any(j in i for j in find_these_acqs)]

#%%
# Load processed mats
d_list = []
for p in processed_mats:
   df_ = pd.read_pickle(p)
   df_['acquisition'] = os.path.split(p)[-1].split('_df.pkl')[0]
   d_list.append(df_) 
df0 = pd.concat(d_list, ignore_index=True)


#%%
# NOTE: no theta_error for this acq (all are NaNs), check this:
acq = '20231213-1103_fly1_eleWT_5do_sh_eleWT_5do_gh'

#%%
import theta_error as th
importlib.reload(th)

f1 = df0[df0['id']==0].copy()
f1 = th.calculate_angle_metrics_focal_fly(f1, winsize=5)

#%%
f1 = th.shift_variables_by_lag(f1, lag=2)

#%%

d_list = []
for (acq, boutnumd), bout in int_frames.groupby(['acquisition', 'boutnum']): 
    start_f, end_f = bout[['start', 'end']].values[0]
   
    curr_frames = np.arange(start_f-1, end_f) 
    currdf = f1[(f1['acquisition']==acq) & (f1['frame'].isin(curr_frames))].copy()
    currdf['boutnum'] = boutnumd
    
    d_list.append(currdf) 
   
intdf = pd.concat(d_list, ignore_index=True)
 
# %%
from sklearn import linear_model
importlib.reload(th)


#%%
fig, ax = plt.subplots()
xvar = 'theta_error'
yvar = 'ang_vel_fly_shifted'
bg_color= 'k'
sns.regplot(data=intdf, x=xvar, y=yvar, ax=ax, color=bg_color,
                scatter_kws={'s': 0.5, 'color': bg_color},
                truncate=False, line_kws={'color': bg_color, 'lw': 0.5})
ax.axvline(x=0, color=bg_color, linestyle='--', lw=0.5)
ax.axhline(y=0, color=bg_color, linestyle='--', lw=0.5)
ax.set_box_aspect(1)
lr, r2 = th.get_R2_ols(intdf, xvar, yvar)
r2_str = 'OLS: y = {:.2f}x + {:.2f}\nR2={:.2f}'.format(lr.coef_[0], 
                                                            lr.intercept_,
                                                            r2)
ax.text(0.01, 1.2, r2_str, fontsize=4, transform=ax.transAxes) 
putil.annotate_regr(intdf, ax, x=xvar, y=yvar, fontsize=4,
                    xloc=0.01, yloc=1.15)
# %%
