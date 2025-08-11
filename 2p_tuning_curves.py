#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : 2p_tuning_curves.py
Created        : 2025/07/29 19:27:38
Project        : /Users/julianarhee/Repositories/flytracker_analysis
Author         : jyr
Last Modified  : 
'''
#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mat73
import utils_2p as util2p
import plotting as putil
#%%
plot_style = 'dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7] * 3 if plot_style == 'dark' else 'k'
# %%
matfile = '/Volumes/Juliana/2p-data/20250725/processed/figures/20250725_Dmel_R35D04-R22D06-RSET-GCaMP8m_f3-029/plotvars.mat'
assert os.path.exists(matfile)
mat = mat73.loadmat(matfile)
mdata = mat['plotdata']
for k, v in mdata.items():
    try:
        print(k, v.shape)
    except Exception as e:
        print("Issue with {}".format(k))
# %%
yvar = 'meantrial_tc'
mean_tc = mdata[yvar].mean(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(mean_tc, color=bg_color, lw=2)
plt.fill_between(np.arange(len(mean_tc)), 
                 mean_tc - mdata[yvar].std(axis=0), 
                 mean_tc + mdata[yvar].std(axis=0), 
                 color=bg_color, alpha=0.2)
plt.title('Mean Tuning Curve')

plt.plot(mdata['meantrial_tc_court'], 'r')

#%% TODO

# Run align_traces_save_vars.mat for 
# 20250724, 20250723, 20250722

# Run this script, aggregate across flies

# %%
import re
rootdir = '/Volumes/Juliana/2p-data'
genotype = 'R35D04-R22D06-RSET-GCaMP8m'

yvar = 'meantrial_tc'

srcdir = os.path.join(rootdir, genotype)

figdir = os.path.join(srcdir, 'figures', 'tuning_curves')
if not os.path.exists(figdir):
    os.makedirs(figdir)

#%%
#session = '20250725'
session_list = ['20250725', '20250724', '20250723', '20250722']  # List of sessions to process, can be extended
#flynum = 1

#filenum = 20
tun_list = []
df_list = []

for session in session_list:
    found_processed_matfiles = glob.glob(os.path.join(srcdir, session, 'f*', #'f{}'.format(flynum),
                           'processed', 'figures', '*', 'plotvars.mat'))
    #fly_list = sorted([os.path.split(f.split('/processed')[0])[-1] for f in found_processed_flydirs])
    #for flynum in fly_list:  
    #plot_srcdir = os.path.join(srcdir, session, 'f{}'.format(flynum), 'processed', 'figures')
    #print(plot_srcdir)
    for matfile in sorted(found_processed_matfiles):
        identifier = re.findall(r'f\d{1}\-\d{3}', matfile)[0]# 'f1-029'
        flynum = int(identifier.split('-')[0][1:])
        filenum = int(identifier.split('-')[1])
        flyid = '{}-f{}'.format(session, flynum)
        print("Processing fly {}, file {}".format(flyid, filenum))

        #assert len(found_mats) == 1, "Expected one plotvars.mat file, found: {}".format(len(found_mats))
        #matfile = found_mats[0] if found_mats else None

        assert os.path.exists(matfile)
        mat = mat73.loadmat(matfile)
        mdata = mat['plotdata']

        mean_tc = mdata[yvar].mean(axis=0)
        mean_tc = mean_tc - mean_tc.min()  # Normalize to min
        max_val = np.max(mean_tc)
        mean_val = np.mean(mean_tc)

        # Tuning
        tun_ = pd.DataFrame({
                    'max_response': max_val, 
                    'mean_response': mean_val,
                    'filenum': filenum,
                    'flynum': flynum,
                    'id': flyid, 
                    'vel': mdata['vel'],
                    },
                    index=[filenum])
        tun_list.append(tun_)

        # Traces
        df_ = pd.DataFrame({
            'dff': mean_tc - mean_tc.min(),
            'time': mdata['meantrial_time']},
            index=np.arange(len(mean_tc)))
        
        df_['filenum'] = filenum
        df_['flynum'] = flynum
        df_['vel'] = mdata['vel']
        df_['id'] = flyid
        df_list.append(df_)

tuning_df = pd.concat(tun_list, axis=0)
dff = pd.concat(df_list, axis=0)

#%%

print(dff.groupby('vel')['id'].nunique())

print(dff.dropna().groupby('vel')['id'].nunique())


#%%
# Plot dFF traces for each 
#df_ = dff[dff['flynum']==2].copy()
#%
for fnum, df_ in dff.groupby('id'):
    #%
    dff_scale = 0.1
    time_scale = 2
    fig, ax = plt.subplots(figsize=(15, 4))
    colors = sns.color_palette('viridis', n_colors=len(df_['vel'].unique()))
    #dff['vel'] = dff['vel'].astype(str)  # Ensure vel is treated as a categorical variable
    offset = 0.2
    last_t = 0
    for ci, (col, (v, d_)) in enumerate(zip(colors, df_.groupby('vel'))):
        print(v)
        #ax.plot(d_['time'], d_['dff'] + ci*offset, label=v, lw=2, color=col)
        ax.plot(d_['time'] + last_t + ci*offset, d_['dff'], label=v, lw=2, color=col)
        last_t = d_['time'].max() + ci*offset + last_t
    ax.set_title('{}'.format(fnum))
    ax.set_box_aspect(0.1)
    putil.vertical_scalebar(ax, leg_xpos=-1, leg_ypos=0, 
                            leg_scale=dff_scale, leg_unit='dF/F',
                            color='w', lw=1,fontsize=16, offset=1)
    putil.horizontal_scalebar(ax, leg_xpos=-1, leg_ypos=0, 
                            leg_scale=time_scale, leg_unit='s', offset=0.04,
                            color='w', lw=1,fontsize=16)
    #ax.set_box_aspect(1)
    ax.axis('off')
    ax.legend(title='Velocity', fontsize=16, loc='upper left', 
              bbox_to_anchor=(1,1), frameon=False)

    figname = '3reps_{}'.format(fnum)
    plt.savefig(os.path.join(figdir, figname + '.png'), dpi=300)
    print(figdir, figname)


#%%
# Get mean, ignore non-numeric values
mean_tuning = tuning_df.groupby('vel')['max_response'].mean().reset_index().rename(columns={'max_response': 'mean_response'})
# Get std, ignore NaN
std_tuning = tuning_df.groupby('vel')['max_response'].std().reset_index().rename(columns={'max_response': 'std_response'})

# Combine mean and std into a single DataFrame
mean_tuning = mean_tuning.merge(std_tuning, on='vel').dropna()

#% PLOT
same_colors = dict((k, bg_color) for k in tuning_df['id'].unique())
fig, ax = plt.subplots(figsize=(5,4))
sns.lineplot(data=tuning_df, x='vel', y='max_response', 
                hue='id', lw=0.5, palette=same_colors,
                ax=ax, legend=0)
# plot mean and std
ax.plot(mean_tuning['vel'], mean_tuning['mean_response'],
        color=bg_color, lw=2)
ax.set_xlabel('Velocity (a.u.)')
ax.set_ylabel('dF/F')
sns.despine(ax=ax, offset=4, trim=False)

ax.set_title('Mean size tuning (N={})'.format(len(tuning_df['id'].unique())),
             fontsize=16, loc='left')

figname = 'mean_tuning_curve'
plt.savefig(os.path.join(figdir, figname + '.png'), dpi=300)


#%%
ax.fill_between(mean_tuning['vel'], 
                mean_tuning['mean_response'] - mean_tuning['std_response'], 
                mean_tuning['mean_response'] + mean_tuning['std_response'], 
                color='gray', alpha=0.2, label='Std Dev')


#%%
plot_srcdir = os.path.join(srcdir, session, 'f{}'.format(flynum), 'processed', 'figures')
#print(plot_srcdir)
filenums = glob.glob(os.path.join(plot_srcdir, '*', 'plotvars.mat')) # For each filepath, find the pattern '-{:03d}' and extract the number
filenums = sorted([int(os.path.split(os.path.split(f)[0])[-1].split('-')[-1]) for f in filenums])
# Sort the file numbers
filenums.sort()
print(filenums)

#%%

#filenum = 20
tun_list = []
df_list = []
for filenum in filenums: 
    print("Processing fly {}, file {}".format(flynum, filenum))
    found_mats = glob.glob(os.path.join(plot_srcdir,
                    '*f{}-{:03d}'.format(flynum, filenum), 
                    'plotvars.mat'))
    assert len(found_mats) == 1, "Expected one plotvars.mat file, found: {}".format(len(found_mats))
    matfile = found_mats[0] if found_mats else None

    assert os.path.exists(matfile)
    mat = mat73.loadmat(matfile)
    mdata = mat['plotdata']

    mean_tc = mdata[yvar].mean(axis=0)
    mean_tc = mean_tc - mean_tc.min()  # Normalize to min
    max_val = np.max(mean_tc)
    mean_val = np.mean(mean_tc)

    # Tuning
    tun_ = pd.DataFrame({
                  'max_response': max_val, 
                  'mean_response': mean_val,
                  'filenum': filenum,
                  'flynum': flynum,
                  'vel': mdata['vel'],
                  },
                  index=[filenum])
    tun_list.append(tun_)

    # Traces
    df_ = pd.DataFrame({
        'dff': mean_tc - mean_tc.min(),
        'time': mdata['meantrial_time']},
        index=np.arange(len(mean_tc)))
    
    df_['filenum'] = filenum
    df_['flynum'] = flynum
    df_['vel'] = mdata['vel']
    df_list.append(df_)


tuning_df = pd.concat(tun_list, axis=0)
dff = pd.concat(df_list, axis=0)


# %%

fig, ax = plt.subplots(figsize=(5,4))
sns.lineplot(data=dff, x='time', y='dff', ax=ax,
             hue='vel', palette='viridis'
             )

#%%
fig, ax = plt.subplots(figsize=(5,4))
colors = sns.color_palette('viridis', n_colors=len(dff['vel'].unique()))
#dff['vel'] = dff['vel'].astype(str)  # Ensure vel is treated as a categorical variable
offset = 0.2
for ci, (col, (v, d_)) in enumerate(zip(colors, dff.groupby('vel'))):
    ax.plot(d_['time'], d_['dff'] + ci*offset, label=v, lw=2, 
            color=col)


# %%

fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=tuning_df, x='vel', y='max_response', 
                hue='vel', palette='viridis',
                style='flynum', ax=ax, s=100)

# %%
