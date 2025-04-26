
#%%
import os
import glob

import numpy as np

import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import utils as util
#import transform_data.relative_metrics as rel
import cv2
import traceback
import transform_multichamber_data as trf

#%%
import plotting as putil
plot_style='white'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%
#%
def load_jaaba_from_mat(mat_fpath, return_dict=False):
    """Get JAABA scores for a behavior from scores_behavior.mat file for a video.

    Args:
        mat_fpath (str):  Full path to scores_chasing.mat for one video.
        return_dict (bool, optional): Return full scores mat as dictionary (not just score values). Defaults to False.

    Returns:
        scores_df (pd.DataFrame): dataframe with columns as fly ids (female and male) and JAABA scores as values
        dict (if return_dict=True): All the fields of `allScores` from .mat file 
    """
    import scipy
    mat = scipy.io.loadmat(mat_fpath)
    # Also contains fields: behaviorName, jabFileNameAbs, etc. but allScores has the data
    allscores_data = mat['allScores'][0][0]
    allscores_fields = ['scores', 'tStart', 'tEnd', 'postprocessed', 'postprocessparams', 't0s', 't1s', 'scoreNorm']

    allscores = {}
    for k, v in zip(allscores_fields, allscores_data):
        if k in ['scores', 'postprocessed']:
            scores = np.vstack(v[0]).T 
            allscores[k] = scores
        else:
            allscores[k] = v[0]
    if return_dict:
        return allscores
    else:
        scores_df = pd.DataFrame(allscores['scores'], columns=range(allscores['scores'].shape[1]))     
        return scores_df

def binarize_jaaba_scores(jaaba_scores, is_threshold=0.3, isnot_threshold=0.14,
                          use_male_annotations=True):
    jaaba_binary = jaaba_scores.copy() #.asarray()
    if use_male_annotations:
        # female is not chasing (score distributions, male should be binary if courting, whereas female is not)
        jaaba_binary = use_male_scores_from_jaaba(jaaba_binary)
         
    jaaba_binary = jaaba_binary.ge(is_threshold)
    #jaaba_binary = jaaba_binary.le(isnot_treshold)
  
    return jaaba_binary

def stack_jaaba_scores(jaaba_binary):
    # Stack jaaba_binary so that each row is a frame and fly, columns are the frame number and fly number
    jaaba_binary_stack = jaaba_binary.stack().reset_index()
    jaaba_binary_stack.columns = ['frame', 'id', 'score']

    return jaaba_binary_stack

def use_male_scores_from_jaaba(jaaba_scores):
    '''
    Use chasing frames identified from male behavior and apply them for female.
    Assumes male ids are EVENS (python 0-counting, 1=male in Matlab).
    '''
    male_ids = [i for i in jaaba_scores.columns if i%2 == 0]
    for i in male_ids:
        female_id = i+1
        jaaba_scores[female_id] = jaaba_scores[i] 

    return jaaba_scores

# %
#%%

# ----------------------------------------
#experiment = 'ht_winged_vs_wingless'
experiment = '38mm_strains'
array_size = '2x2'

# parent dir where transformed data was saved
dstdir = os.path.join('/Volumes/Juliana/free_behavior_analysis', experiment)
# local dir where aggregated data saved 
local_dir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/38mm_strains'
# JAABA dir containing all the acquisitions (and scores)
jaaba_dir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/JAABA_classifiers/38mm_multichamber_winged-wingless_classifier/JAABA'

# Set output dirs
figdir = os.path.join(dstdir, 'figures', 'multichamber_strains')
if not os.path.exists(figdir):
    os.makedirs(figdir)
   
figid = os.path.join(dstdir, experiment)
print(figid)

# %% Load metadata for YAK and MEL strains
meta_fpaths = glob.glob(os.path.join(dstdir, '*.csv'))
strainmeta = trf.get_meta_data(dstdir, experiment='strains', return_all=False)

#%%
# Load aggregated processed data (from local)
aggregate_processed_datafile = os.path.join(local_dir, 
                                        '38mm_strains_df.parquet')
aggregate_processed_datafile_all = os.path.join(local_dir, 
                                        '38mm_all_df.parquet')
df0 = pd.read_parquet(aggregate_processed_datafile)
print("Loaded processed: {}".format(aggregate_processed_datafile))

df0['strain'] = df0['strain'].map(lambda x: x.replace('CS mai', 'CS Mai'))
df0['strain'] = df0['strain'].map(lambda x: x.replace('CS Mai ', 'CS Mai'))

#%%
conds = df0[['species', 'strain', 'acquisition', 'fly_pair']].drop_duplicates()
counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
print(counts)


#%%
df0['strain'] = df0['strain'].map(lambda x: x.replace(' ', '_'))

yak_strains = df0[df0['species']=='Dyak']['strain'].unique()
mel_strains = df0[df0['species']=='Dmel']['strain'].unique()
 
strain_num_dict = dict((v, i) for i, v in enumerate(yak_strains))
strain_num_dict.update(dict((v, i) for i, v in enumerate(mel_strains)))

# make dictionary where keys are strain names and values are species
df0['strain_num'] = df0['strain'].map(strain_num_dict)

strain_dict = dict( (k, 'Dyak') for k in yak_strains)
strain_dict.update(dict((k, 'Dmel') for k in mel_strains))

# %%
# Get mean velocity by condition
grouper= ['species', 'strain', 'strain_num', 'global_id']
 
mean_vel = df0[df0['sex']=='m'].groupby(grouper)['vel'].mean().reset_index()
mean_vel.head()
#mean_vel

#%%
# Create subset of cubehelix palette to exclude white but keep pink
strain_palette = sns.cubehelix_palette(rot=-1.5, light=0.8, reverse=True)
sns.palplot(strain_palette)

# %% Plot mean velocity by condition
fig, ax =plt.subplots(figsize=(5,3))
#sns.pointplot(data=df[df['sex']=='m'], ax=ax, x='winged', 
#              hue='acquisition', y='vel')
sns.barplot(data=mean_vel, ax=ax, x='species', y='vel', hue='strain',
            palette=strain_palette, width=0.8, fill=False)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
#sns.swarmplot(data=mean_vel, ax=ax, x=grouper, y='vel', 
#              hue='global_id', palette='tab20', legend=False)
#ax.set_box_aspect(1)
# rotate x-tick labels
plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_{}'.format(experiment)
plt.savefig(os.path.join(figdir, figname+'.png'))

# %% ADD JAABA INFO

def hist_jaaba_scores_male_female(jaaba_scores, ix, ax=None, 
                        curr_color='r', bg_color='k', is_threshold=0.4):
    #currcolor = grouper_palette[currcond]
    ax.hist(jaaba_scores[ix], label='male', color=curr_color)
    ax.hist(jaaba_scores[ix+1], label='female', color='darkgrey') #, legend=0)
    ax.axvline(is_threshold, color=bg_color, linestyle='--', lw=0.5)
    return ax
# ==========================================
# Load JAABA scores
# ==========================================
no_jaaba_acqs = ['20250320-1025_fly1-4_Dyak-gab_3do_gh',
                 '20250306-0917_fly1-4_Dyak-cost-abid-tai-cy_3do_gh']

acqs = [a for a in df0['acquisition'].unique() if a not in no_jaaba_acqs]

has_jaaba = True
beh_type = 'chasing'
is_threshold = 0.4
isnot_threshold = 0.2 #14

# Look at scores
cond_name = 'strain'
if has_jaaba:
    for acq, df_ in df0.groupby('acquisition'):# in acqs[0:nr*nc]:
        if acq in no_jaaba_acqs:
            continue
        mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
        jaaba_scores = load_jaaba_from_mat(mat_fpath)        
        nr=2; nc=int(df_['id'].nunique()/2/nr);
        fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                                figsize=(nc*1.5, nr*1.5))
        ix=0
        for ai, ax in enumerate(axn.flat):
            if ix not in df_['id'].unique():
                ix += 2
                continue
            currcond = df_[df_['id']==ix][cond_name].unique()[0]
            ax = hist_jaaba_scores_male_female(jaaba_scores, ix, ax=ax, 
                        curr_color='r', is_threshold=is_threshold)
            ax.set_title('{}: ids {}, {}'.format(currcond, ix, ix+1), 
                         loc='left', fontsize=4) 
            ix += 2
        fig.suptitle(acq, x=0.5, y=0.95, fontsize=8)    
        putil.label_figure(fig, figid)  
        figname = 'jaaba_scores_{}_{}'.format(beh_type, acq)
        plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))  
#%%
is_threshold = 0.4
isnot_threshold = 0.2 #14

#df0[beh_type] = False
binarize_jaaba = True
df_list = []
no_scores = []
for acq, df in df0.groupby('acquisition'):
    
    if has_jaaba:
        # Load JAABA scores
        try:
            mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 
                        'scores_{}*.mat'.format(beh_type)))[0]
        except IndexError:
            no_scores.append(acq)
            print("No JAABA scores for {}".format(acq))
            continue 
        try:
            jaaba_scores = load_jaaba_from_mat(mat_fpath)
        except Exception as e:
            traceback.print_exc()
            no_scores.append(acq)
            print("Error loading JAABA scores for {}: {}".format(acq, e))
        #
        # Find where jaaba_scores is greater than threshold value, set to 1 and otherwise 0
        if binarize_jaaba:
            jaaba_binary = binarize_jaaba_scores(jaaba_scores, is_threshold, isnot_threshold)
            jaaba_scores_stacked = stack_jaaba_scores(jaaba_binary)
        else:
            jaaba_scores_stacked = stack_jaaba_scores(jaaba_scores)
        
        # Merge jaaba_binary_stack with df
        if 'chasing' in df.columns:
            df = df.drop(columns=['chasing'])    
        df = df.merge(jaaba_scores_stacked.rename(columns={'score': beh_type}), 
                    how='inner', on=['frame', 'id']) 
        if binarize_jaaba: 
            df[beh_type] = df[beh_type].astype('bool')
        else:
            df['{}_binary'.format(beh_type)] = df[beh_type].ge(is_threshold)

    # Reassign to df0
    #df0.loc[df0['acquisition']==acq] = df
    df_list.append(df) #[~df['copulating']])
#%
ftj = pd.concat(df_list)

#%%

ftj['strain_num'].unique()


#%%
conds = ftj[['acquisition', 'species', 'strain', 'strain_num', 'fly_pair']].drop_duplicates()
counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
print(counts)
ftj['species'] = ftj['species'].astype('category')

#%%
yak = ftj[ftj['species']=='Dyak']   
mel = ftj[ftj['species']=='Dmel']

#%%
def filter_chasing(tdf, use_jaaba=True,
                   min_vel=10, max_facing_angle=np.deg2rad(20),
                   max_dist_to_other=20, max_targ_pos_theta=np.deg2rad(160),
                   min_targ_pos_theta=np.deg2rad(-160), 
                   min_wing_ang=np.deg2rad(45)): 
   
    if use_jaaba:
        chasedf = tdf[(tdf['chasing']) & (tdf['sex']=='m')
                    & (tdf['facing_angle']<=max_facing_angle)].copy()
    else:
#        chasedf = ftj[(ftj['sex']=='m') & (ftj['vel']>=min_vel)
#                & (ftj['facing_angle']<=max_facing_angle)
#                & (ftj['angle_between']<=max_angle_between)].copy()
#                #& (ftj['max_wing_ang']>=min_wing_angle)].copy()
              
        chasedf = tdf[ #(tdf['id']%2==0)
                #& (tdf['chasing'])
                (tdf['vel'] > min_vel)
                & (tdf['targ_pos_theta'] <= max_targ_pos_theta)
                & (tdf['targ_pos_theta'] >= min_targ_pos_theta)
                & (tdf['facing_angle'] <= max_facing_angle)
                & (tdf['min_wing_ang'] >= min_wing_ang)
                & (tdf['dist_to_other'] <= max_dist_to_other)].copy()
    
    #court_ = tdf[(tdf['sex']=='m') & (tdf['vel']>=min_vel)
    #         & (tdf['facing_angle']<=max_facing_angle)
    #         & (tdf['angle_between']<=max_angle_between)].copy()
    return chasedf

#%%
# Compare velocity overall vs. velocity during CHASING bouts
has_jaaba = True
#min_wing_angle = np.deg2rad(20)
#min_vel = 10
#max_facing_angle = np.deg2rad(45)
#max_angle_between = 1.6 #np.deg2rad(90)

min_vel = 10
max_facing_angle = np.deg2rad(45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

chasedf = filter_chasing(ftj, use_jaaba=has_jaaba,
                         min_vel=min_vel, max_facing_angle=max_facing_angle,
                        max_dist_to_other=max_dist_to_other,
                        max_targ_pos_theta=max_targ_pos_theta,
                        min_targ_pos_theta=min_targ_pos_theta,
                        min_wing_ang=min_wing_ang)               

mean_vel = ftj[(ftj['sex']=='m')].groupby(grouper)['vel'].mean().reset_index()
mean_vel_chasing = chasedf.groupby(grouper)['vel'].mean().reset_index()

#mean_vel['species'] = [strain_dict[m] for m in mean_vel['strain']]
#mean_vel_chasing['species'] = [strain_dict[m] for m in mean_vel_chasing['strain']]
 
fig, axn = plt.subplots(1,2, sharex=True, sharey=True)
for ai, mean_ in enumerate([mean_vel, mean_vel_chasing]):
    ax=axn[ai]
    sns.barplot(data=mean_, ax=ax, x='species', y='vel', hue='strain_num',
             palette=strain_palette, width=0.5, fill=False, legend=ai>0)
    #sns.swarmplot(data=mean_, ax=ax, x='species', y='vel', hue='strain',
    #         palette='tab20', legend=False)
    ax.set_box_aspect(1)
    ax.set_title('all' if ai==0 else 'chasing')
    ax.set_xlabel('')
    # rotate x-tick labels for current axis
    plt.setp(ax.get_xticklabels(), rotation=90)
sns.move_legend(axn[1], loc='upper left', bbox_to_anchor=(1, 1))
# Rotate x-tick labels for both axe
#plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_chasing'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# IDentify courtship frames
ftj = ftj.reset_index(drop=True)

#%%
# P(CHASING)
#plotdf = ftj[(ftj['chasing']) & (ftj['sex']=='m')].copy()
use_jaaba= False
# ----------------------------------------
min_vel = 10
max_facing_angle = np.deg2rad(45)
max_dist_to_other = 38
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang_deg = 20
min_wing_ang = np.deg2rad(min_wing_ang_deg)

if use_jaaba:
    chasedf = ftj[(ftj['sex']=='m') & (ftj['chasing']==True)
                    & (ftj['facing_angle']<=max_facing_angle)].copy()
else:
    chasedf = ftj[(ftj['sex']=='m') #& (ftjaaba['chasing']==1)
                & (ftj['vel']> min_vel)
                & (ftj['targ_pos_theta'] <= max_targ_pos_theta)
                & (ftj['targ_pos_theta'] >= min_targ_pos_theta)
                & (ftj['facing_angle'] <= max_facing_angle)
                #& (ftj['max_wing_ang'] >= min_wing_ang)
                & (ftj['dist_to_other'] <= max_dist_to_other)].copy()
ftj['chasing'] = False
ftj.loc[chasedf.index, 'chasing'] = True

orienting = ftj[(ftj['sex']=='m')
                & (ftj['facing_angle']<=max_facing_angle)].copy()
ftj['orienting'] = False
ftj.loc[orienting.index, 'orienting'] = True

#% P(SINGING)
singdf = ftj[(ftj['sex']=='m') #& (ftj['vel']>min_vel)
            & (ftj['facing_angle']<=max_facing_angle)
            & (ftj['targ_pos_theta'] <= max_targ_pos_theta)
            & (ftj['targ_pos_theta'] >= min_targ_pos_theta)
            & (ftj['max_wing_ang']>=min_wing_ang)].copy()
ftj['singing'] = False
ftj.loc[singdf.index, 'singing'] = True

ftj['behav_sum'] = ftj[['singing','chasing', 'orienting']].sum(axis=1)

ftj['courting'] = False
ftj.loc[ftj['behav_sum']>0, 'courting'] = True

ftj['{}_legend'.format(grouper)] = ftj[grouper].map(lambda x: '{} (n={})'.format(x, counts[x]))

#ftj[ftj['singing']]['max_wing_ang'].min()
 
#%%
# Binn dist_to_other
bin_size = 5 #3 
max_dist = 35 #25#np.ceil(ftjaaba['dist_to_other'].max())
dist_bins = np.arange(0, max_dist+bin_size, bin_size)
print(dist_bins)

# Cut dist_to_other into bins and assign label to new columns:
ftj['binned_dist_to_other'] = pd.cut(ftj['dist_to_other'], 
                                    bins=dist_bins, 
                                    labels=dist_bins[:-1])   
ftj['binned_dist_to_other'] = ftj['binned_dist_to_other'].astype(float)
#
#%%
# Get means of chasing and singing by binned_dist_to_other
courting = ftj[ftj['courting']==True].copy()
meanbouts_courting = courting.groupby([grouper, 'acquisition', 'fly_pair', #'behavior', 
                        'binned_dist_to_other'])[['singing', 'chasing']].mean().reset_index()

#%%
# Addd counts of each condition type
counts = meanbouts_courting[[grouper, 'acquisition', 'fly_pair']].drop_duplicates().groupby(grouper)['fly_pair'].count()

strains = meanbouts_courting[grouper].unique()

meanbouts_courting['{}_legend'.format(grouper)] = meanbouts_courting[grouper].map(lambda x: '{} (n={})'.format(x, counts[x]))
#counts[strains[0]]

#%%
# Bin dist_to_other during chasing and singing
#species_palette = {'Dmel': 'lavender', 

plot_bar = True
if grouper == 'condition':
    grouper_palette = 'cubehelix' #{'winged': 'mediumorchid',
                       #'wingless': 'lightsteelblue'}
else:
    grouper_palette = 'cubehelix'
error_type = 'ci'
plot_pairs = False

plot_type = 'bar' if plot_bar else 'point'

fig, axn = plt.subplots(1, 2, figsize=(6.5, 4), sharex=True, sharey=False)
for ai, behav in enumerate(['chasing', 'singing']):
    ax=axn[ai]
    if plot_pairs:
        sns.stripplot(data=meanbouts_courting,
                    x='binned_dist_to_other', 
                    y=behav, ax=ax, 
                    hue='{}_legend'.format(grouper), palette=grouper_palette, legend=False,
                    edgecolor='w', linewidth=0.25, dodge=True, jitter=True)
    if plot_bar:
        sns.barplot(data=meanbouts_courting,
                    x='binned_dist_to_other', 
                    y=behav, ax=ax, lw=0.5,
                    errorbar=error_type, errcolor=bg_color, errwidth=0.75,
                    hue='{}_legend'.format(grouper), palette=grouper_palette, 
                    fill=plot_pairs==False, legend=1)
    else:
        sns.pointplot(data=meanbouts_courting,
                    x='binned_dist_to_other', 
                    y=behav, ax=ax, lw=2,
                    errorbar=error_type, errwidth=0.75,
                    hue='{}_legend'.format(grouper), palette=grouper_palette, legend=1)
                    #fill=plot_pairs==False, legend=1)

    ax.set_ylim([0, 1])
    if ai==0:
        ax.legend_.remove()
    else:
        sns.move_legend(ax, loc='lower right', bbox_to_anchor=(1,1), frameon=False,
                title='', fontsize=6)
for ax in axn:
    ax.set_box_aspect(0.8)
    ax.set_xlabel('distance to other (mm)')
plt.subplots_adjust(wspace=0.5, right=0.95)

# format xticks to single digit numbers:
bin_edges = [str(int(x)) for x in dist_bins[:-1]]
bin_edges[0] = '<{}'.format(bin_size)
bin_edges[-1] = '>{}'.format(int(dist_bins[-2]))
axn[0].set_xticks(range(len(bin_edges)))
axn[0].set_xticklabels([str(x) for x in bin_edges], rotation=0)
axn[1].set_ylabel("p(singing|courtship)")

putil.label_figure(fig, figid)

figname = 'p_singing_chasing_v_binned_dist_to_other_bin-{}_{}'.format(bin_size, plot_type)
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
#hist_color = 'mediumorchid'
hist_bins = 20
#ylim = 0.2
hist_palette = 'cubehelix'

chasedf = ftj[(ftj['chasing'])].copy()
singdf = ftj[(ftj['singing'])].copy()

fig, axn = plt.subplots(1, 2, figsize=(6,3), sharex=True, sharey=True)
ax=axn[0]
ax.set_title('dist to other (chasing)')
sns.histplot(data=chasedf, ax=ax,
            x='dist_to_other', bins=hist_bins, stat='probability',
            hue='{}_legend'.format(grouper), palette=hist_palette, common_norm=False, 
            element='step', lw=0.5, alpha=0.3)
ax.legend_.remove()
#%
ax = axn[1]
ax.set_title('dist to other (singing)')
sns.histplot(data=singdf, ax=ax,
            x='dist_to_other', bins=hist_bins, stat='probability',
            hue='{}_legend'.format(grouper), palette=hist_palette, common_norm=False, 
            element='step', lw=0.5, alpha=0.3)
            #color=hist_color)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1), frameon=False, 
                title='')
for ax in axn:
    #ax.set_ylim([0, ylim])
    ax.set_box_aspect(1)
    ax.set_xlabel('interfly distance (mm)')

putil.label_figure(fig, figid)
figname = 'p_chasing_singing_v_dist_to_other_hist'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%

# ----------------------------------------------------------
# TRANSFORM DATA

#%% 
#
# Apply relative tranfsormation to 1 arena, 1 acquisition:
#acq ='20240809-0956_fly1-9_Dyak_WT_3do_gh' 
#curr_arena = 1
#df_ = df0[(df0['acquisition']==acq)
#          & (df0['fly_pair']==curr_arena)].copy()
#currdf = df0[(df0['acquisition']==acq)].copy()

#print(df_['id'].unique())

#
#t_list = []
#for (acq, curr_arena), df_ in ftj.groupby(['acquisition', 'fly_pair']):
#
#    cop_ix = df_[df_['copulating']].iloc[0].name if True in df_['copulating'].unique() else None
#    #print(cop_ix)
#    #%
#    acqdir = os.path.join(srcdir, assay, acq)
#    cap = rel.get_video_cap(acqdir, movie_fmt='.avi')
#
#    # N frames should equal size of DCL df
#    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
#    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
#    #print(n_frames, frame_width, frame_height)
#    #%
#    fps = 60.
#    df_['sec'] = df_['frame']/fps
#    #df_['sec']
#    #%
#    flyid1 = min(df_['id'].unique())
#    flyid2 = max(df_['id'].unique())
#    df_['ori'] = -1*df_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
#    transf_df = rel.do_transformations_on_df(df_, frame_width, frame_height, 
#                                cop_ix=cop_ix, flyid1=flyid1, flyid2=flyid2,
#                                verbose=True, get_relative_sizes=False)
#    #print(df_.head())
#    #print(transf_df.columns)
#    print(acq, transf_df['id'].unique())
#    
#    t_list.append(transf_df)
#
#tdf = pd.concat(t_list) 
#%%


# % Plot relative position
# min_wing_angle = np.deg2rad(45)
# min_vel =5 # 10
# max_facing_angle = np.deg2rad(25) #60) #90)
# max_angle_between = 1.6 #np.deg2rad(90)
# max_dist_to_other = 25
# max_targ_pos_theta = np.deg2rad(160)
# min_targ_pos_theta = np.deg2rad(-160)
min_vel = 10
max_facing_angle = np.deg2rad(45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

#%%
use_jaaba = False

#min_vel = 5
#max_facing_angle = np.deg2rad(25)
#max_dist_to_other = 25
#acq = '20240809-0956_fly1-9_Dyak_WT_3do_gh'
#acq = '20240809-0956_fly1-9_Dyak_WT_3do_gh'
acq ='20240812-0950_fly1-9_Dyak_WT_5do_gh'
#acq = '20240819-0945_fly1-9_Dyak_WT_5do_gh'
# Look at IDs 6, 7 (flypair 4)
#df_ = tdf[(tdf['acquisition']==acq) & (tdf['fly_pair'].isin([4]))].copy()

acq = '20250306-0917_fly1-4_Dyak-cost-abid-tai-cy_3do_gh'
#vid_df = ftj.copy()
vid_df = ftj[(ftj['acquisition']==acq)].copy()

use_jaaba = False
conditions = ftj[grouper].unique()
colors = sns.color_palette('cubehelix', n_colors=len(conditions))
cond_colors = dict(zip(conditions, colors))

fig, axn = plt.subplots(3, 3, sharex=True, sharey=True)
for ai, (flypair, df_) in enumerate(vid_df.groupby('fly_pair')):
    
    ax=axn.flat[ai]
    if use_jaaba:
        court_ = df_[(df_['sex']=='m') & (df_['chasing']==True)
                     & (df_['facing_angle']<=max_facing_angle)].copy()
    else:
        #court_ = df_[ (df_['sex']=='m')
        #            & (df_['vel'] > min_vel)
        #            & (df_['targ_pos_theta'] <= max_targ_pos_theta)
        #            & (df_['targ_pos_theta'] >= min_targ_pos_theta)
        #            & (df_['facing_angle'] <= max_facing_angle)
        #            & (df_['min_wing_ang'] >= min_wing_ang)
        #            & (df_['dist_to_other'] <= max_dist_to_other)].copy()
        court_ = df_[(df_['sex']=='m') #& (ftjaaba['chasing']==1)
                    & (df_['vel']> min_vel)
                    & (df_['targ_pos_theta'] <= max_targ_pos_theta)
                    & (df_['targ_pos_theta'] >= min_targ_pos_theta)
                    & (df_['facing_angle'] <= max_facing_angle)
                    & (df_['max_wing_ang'] >= min_wing_ang)
                    & (df_['dist_to_other'] <= max_dist_to_other)].copy()
         
         
    f2_ = df_[ (df_['frame'].isin(court_['frame']))
            & (df_['sex']=='f')].copy() #wing_ext[wing_ext['id']==1].copy()
    #fig, ax= plt.subplots()
    curr_cond = df_[grouper].unique()[0]
    #curr_pal = 'cornflowerblue' if curr_cond==conditions[0] else 'violet'
    #curr_pal = 'magma'
    sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
                color=cond_colors[curr_cond])
    ax.set_aspect(1)
    
    
#%%
# ----------------------------------------------------------
# DEFINE COURTING/SINGING - SPATIAL OCCUPANCY
# ----------------------------------------------------------
use_jaaba= False #True

if use_jaaba:
    court_ = ftj[(ftj['sex']=='m') & (ftj['chasing']==True)
                    & (ftj['facing_angle']<=max_facing_angle)].copy()
else:
    court_ = ftj[(ftj['sex']=='m') #& (ftjaaba['chasing']==1)
                & (ftj['vel']> min_vel)
                & (ftj['targ_pos_theta'] <= max_targ_pos_theta)
                & (ftj['targ_pos_theta'] >= min_targ_pos_theta)
                & (ftj['facing_angle'] <= max_facing_angle)
                & (ftj['max_wing_ang'] >= min_wing_ang)
                & (ftj['dist_to_other'] <= max_dist_to_other)].copy()
        
# Get female-centered frames: 
# Relative to the female (centered at 0,0), where is the male
f_list = []
for acq, curr_court in court_.groupby('acquisition'):
   
    # NOTE: BEFORE, this was using df, instead of df_!! 
    f2_ = ftj[ (ftj['frame'].isin(curr_court['frame']))
             & (ftj['sex']=='f')
             & (ftj['acquisition']==acq)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f2_)
f2 = pd.concat(f_list).reset_index(drop=True)

#%%
import matplotlib as mpl

spatial_cmap = 'GnBu' if plot_style=='white' else 'magma'
bins=100
stat='probability'
vmax=0.001 if stat=='probability' else 50
#f2 = f2.reset_index(drop=True)
print(grouper)

print(conds.groupby(['species', 'strain'])['fly_pair'].count())

for curr_species, f2_ in f2.groupby('species'):
    n_strains = f2_['strain'].nunique()
    # Get number of unique fly pairs by strain
    max_n_pair = f2_[['strain', 'acquisition', 'fly_pair']].drop_duplicates().groupby('strain').count().max().iloc[0]
    n_strains = f2_['strain'].nunique()

    nc = max_n_pair #int(np.ceil(max_n_pair/n_strains))
    nr = n_strains    
    fig, axn = plt.subplots(nr, nc, figsize=(5,4), sharex=True, sharey=True)
    for ri, (strain, strain_df) in enumerate(f2_.groupby('strain')):
        for ci, (flypair, d_) in enumerate(strain_df.groupby(['acquisition', 'fly_pair'])):
        #for (strain, flypair, d_) in enumerate(f2_.groupby(['acquisition', 'fly_pair'])): 
            ax = axn[ri, ci] #axn.flat[ai]
            #curr_pal = 'cornflowerblue' if curr_cond=='winged' else 'violet'
            sns.histplot(data=d_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
                        cmap=spatial_cmap, stat=stat, vmin=0, vmax=vmax, bins=bins)
            ax.set_aspect(1)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.axis('off')
            if ci==0:
                ax.text(ax.get_xlim()[0]-100, ax.get_ylim()[-1]+100, strain, 
                    ha='left', va='center', fontsize=4, rotation=0)
    plt.subplots_adjust(top=0.9, left=0.1)
    fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
    fig.text(0.01, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
    fig.suptitle('Courting, {}, jaaba={}'.format(curr_species, use_jaaba), 
                 x=0.3, y=0.95, fontsize=10) 
    for ax in axn.flat[ai+1:]:
        ax.axis('off') 
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, wspace=0.2)
    
    putil.label_figure(fig, figid) 
    figname = 'spatial_occupancy_{}_per-flypair_jaaba-{}'.format(curr_species, use_jaaba) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))    
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))    


#%%
# ALL 

cmap='GnBu' if plot_style=='white' else 'magma'
stat='probability'
vmax=0.0005 if stat=='probability' else 250
bins=100
#vmax = 0.002 if stat=='probability' else 250
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

#n_conds = f2[grouper].nunique()

#if grouper=='strain':
#    nr=2; nc=3;
#else:
#    nr=1; nc=n_conds;
nr = 2
nc = 4

def plot_occupancy(f2_, ax=None, cmap='viridis',
                   vmin=None, vmax=None, bins=100,
                   stat='probability', bg_color='w'):
    if ax is None:
        fig, ax = plt.subplots()
                
    sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
            cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=bins) # %%
    ax.plot(0, 0, 'k', markersize=5, marker='>')
    # ax.set_xlim([])
    ax.set_aspect(1)
    ax.set_title(cond, fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.axis('off')
    ax.plot(0, 0, bg_color, markersize=5, marker='>')

    return ax


for (curr_species, curr_strain), f2_ in f2.groupby(['species', 'strain']):
    fig, ax = plt.subplots()
    ax = plot_occupancy(f2_, ax=ax, cmap=cmap,
                        vmin=0, vmax=vmax, bins=bins,
                        stat=stat, bg_color='w')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_title('{}: {}'.format(curr_species, curr_strain), fontsize=8)
    figname = 'occ_{}_{}_jaaba-{}'.format(curr_species, curr_strain, use_jaaba)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    
#%% ALL on the same plot
 
#curr_species = 'Dyak'
#curr_plotd = f2[f2['species']==curr_species]
for curr_species, curr_plotd in f2.groupby('species'):
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                            figsize=(nc*3,nr*3))
    for ai, (cond, f2_) in enumerate(curr_plotd.groupby('strain')):
        ax=axn.flat[ai]

        ax = plot_occupancy(f2_, ax=ax, cmap=cmap,
                            vmin=0, vmax=vmax, bins=bins,
                            stat=stat, bg_color='w')
        ax.set_xlim([-300, 300])
        ax.set_ylim([-300, 300])
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)

    #fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
    #fig.text(0.05, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
    for ax in axn.flat[ai+1:]:
        ax.axis('off')
        
    #fig.suptitle('Courting frames, uses_jaaba={}'.format(use_jaaba))
    fig.text(0.1, 0.95, 'Male position from female-centered view (jaaba={})'.format(use_jaaba), 
            fontsize=8)
    putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                                hue_title=stat)

    putil.label_figure(fig, figid)     
    figname = 'spatial_occupancy_all-pairs_{}_jaaba-{}'.format(curr_species, use_jaaba) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir)
 
# %%
# Get male-centered frames: 
# Relative to the male (centered at 0,0), where is the female,
# i.e., where does the male keep the female?
f_list = []
for acq, curr_court in court_.groupby('acquisition'):
   
    # NOTE: BEFORE, this was using df, instead of df_!! 
    f1_ = ftj[ (ftj['frame'].isin(curr_court['frame']))
             & (ftj['sex']=='m')
             & (ftj['acquisition']==acq)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f1_)
f1 = pd.concat(f_list).reset_index(drop=True)

   #%%
if grouper=='strain':
    nr=2; nc=3;
else:
    nr=1; nc=n_conds;

fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                        figsize=(nc*3,nr*3))
#fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
for ai, (cond, f1_) in enumerate(f1.groupby(grouper)):
    ax=axn.flat[ai]
    sns.histplot(data=f1_, 
             x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
             cmap='magma',stat='probability', bins=bins, vmax=vmax, vmin=0)
    ax.set_title(cond)
    ax.set_aspect(1)
    ax.plot(0, 0, 'w', markersize=5, marker='>') 

for ax in axn.flat[ai+1:]:
    ax.axis('off')

fig.text(0.1, 0.95, 'Female position from male-centered view', fontsize=8)
putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                             hue_title=stat)

putil.label_figure(fig, figid)     
figname = 'male-perspective_all-pairs_jaaba-{}'.format(use_jaaba) 
plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir)
# %%
max_dist_to_other = 15
court_ = ftj[(ftj['sex']=='m') #& (ftjaaba['chasing']==1)
            & (ftj['vel']> 5)
            & (ftj['targ_pos_theta'] <= max_targ_pos_theta)
            & (ftj['targ_pos_theta'] >= min_targ_pos_theta)
            & (ftj['facing_angle'] <= max_facing_angle)
            #& (ftj['max_wing_ang'] >= min_wing_ang)
            & (ftj['dist_to_other'] <= max_dist_to_other)].copy()
   
court_.loc[court_['max_wing_ang']>=min_wing_angle, 'is_singing']   = True
fig, ax = plt.subplots()
sns.pointplot(data=court_,
                    x='binned_dist_to_other', 
                    y='is_singing', ax=ax, lw=2,
                    errorbar=error_type, errwidth=0.75,
                    hue='{}_legend'.format(grouper), palette=grouper_palette, legend=1)
                    #fill=plot_pairs==False, legend=1)
