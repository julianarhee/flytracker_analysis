
#%%
import os
import glob

import numpy as np

import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import utils as util

#%%

def load_feat_and_trk(acqdir):
    feat_fpath = glob.glob(os.path.join(acqdir, '202*', '*-feat.mat'))[0]
    #% Load feature mat
    feat_ = util.load_feat(acqdir) #, subfolder=None)
    trk_ = util.load_tracks(acqdir)#, subfolder=None)
    # %
    # select unique columns of trk_ using common_cols to exclude
    # columns that are common to both feat_ and trk_
    unique_cols = [c for c in trk_.columns if c not in feat_.columns]

    # merge feat_ and trk_ by index, drop duplicate columns
    feat_trk = pd.merge(feat_, trk_[unique_cols], how='inner', left_index=True, right_index=True,
                        suffixes=('', ''))
    feat_trk['acquisition'] = acq
    
    return feat_trk

def assign_sex(feat_trk):
    # Assign sex
    feat_trk['sex'] = 'f'
    feat_trk.loc[feat_trk['id'] % 2 == 0, 'sex'] = 'm'
    feat_trk['sex'] = feat_trk['sex'].astype('category')

    return feat_trk

def assign_frame_number(feat_trk):
    # Add frame
    for i, df_ in feat_trk.groupby('id'):
        # assign frame number
        feat_trk.loc[feat_trk['id']==i, 'frame'] = np.arange(len(df_))
    
    return feat_trk

def assign_conditions_to_multichamber(feat_trk, meta):
    ''' 
    Assign winged/wing (or other conditions) to high-throughput multichamber data.
    Also adds acquisition and frame numnber. Assumes even is male, odd female.
    '''
    # Assign fly pair number to feat_trk
    for i in meta['fly_num'].unique():
        feat_trk.loc[feat_trk['id']==2*i-1, 'fly_pair'] = i # this is female
        feat_trk.loc[feat_trk['id']==2*i-2, 'fly_pair'] = i # this is the male

    #% Assign conditions
    # Get pair number for wingless 
    wingless_pairs = [i-1 for i in meta[(meta['acquisition']==acq) \
                        & (meta['manipulation_male']=='wingless')]['fly_num'].unique()]
    # Assign wing or wingless for each pair
    feat_trk['winged'] = 'winged'
    feat_trk.loc[feat_trk['fly_pair'].isin(wingless_pairs), 'winged'] = 'wingless' #False
    feat_trk['winged'] = feat_trk['winged'].astype('category')

    return feat_trk

#%%
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

def binarize_jaaba_scores(jaaba_scores, is_threshold=0.3, isnot_threshold=0.14):
    jaaba_binary = jaaba_scores.copy().asarray()
    jaaba_binary.get(is_threshold) = True
    jaaba_binary.get(isnot_treshold) = False
    
    # Stack jaaba_binary so that each row is a frame and fly, columns are the frame number and fly number
    jaaba_binary_stack = jaaba_binary.stack().reset_index()
    jaaba_binary_stack.columns = ['frame', 'id', 'score']

    return jaaba_binary_stack

# %
def get_copulation_frames(mat_fpath):
    """Load -actions.mat for a video, return copulation frame indices.

    Args:
        mat_fpath (str): full path to -actions.mat

    Returns:
        cop_ix_dict (dict): keys are male and female IDs, values are copulation start frame (-1 if no cop)
    """
    import scipy
    mat = scipy.io.loadmat(mat_fpath)
    beh_names = [v[0][0]  for v in mat['behs']]
    # Get copulation events
    cop_name_ix = beh_names.index('copulation ')
    cop_ixs = [v[cop_name_ix][0][0] if len(v[cop_name_ix])>0 else -1 for v in mat['bouts']]
    cop_ix_dict = dict((k, v) for k, v in enumerate(cop_ixs))

    # Assign female ID to have same copulation index
    for k in cop_ix_dict.keys():
        if k % 2 > 0: # if odd, is female ID
            continue 
        #print(k)
        if cop_ix_dict[k] != -1:
            cop_ix_dict[k+1] = cop_ix_dict[k] # assign female to have same copulation frame
        
    return cop_ix_dict

#%%
rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
#assay = '38mm_dyad'
#acq = '20241101-0940_fly1_Dbia-WT_3do_gh'
#%
assay = 'multichamber'

acqs = ['20240805-1019_fly1-9_Dyak_WT_5do_gh', # annotated
        '20240809-0956_fly1-9_Dyak_WT_3do_gh',
        '20240812-0950_fly1-9_Dyak_WT_5do_gh',  # annotated
        '20240819-0945_fly1-9_Dyak_WT_5do_gh'    # annotated
        ]

# %% Load metadata
#meta_fpath = os.path.join(rootdir, assay, 'courtship_free_behavior_data - raw data 3x3 .csv')
meta_fpath = os.path.join(rootdir, assay, 'courtship_free_behavior_data - ht_winged_wingless.csv')
allmeta = pd.read_csv(meta_fpath)
allmeta
#%%
d_list = []
for acq in acqs:
    acqdir = os.path.join(rootdir, assay, acq)
    feat_trk = load_feat_and_trk(acqdir)
     
    # Get meta info for current acquisition
    meta = allmeta[allmeta['acquisition']==acq]

    # Check that the number of unique fly ids in feat_trk is 2x as the number of unique fly pairs in meta
    assert len(feat_trk['id'].unique()) == 2*len(meta['fly_num'].unique()), 'Incorrect fly ID to pair assignment'

    # Assign conditions
    feat_trk = assign_sex(feat_trk)
    feat_trk = assign_frame_number(feat_trk)
    feat_trk = assign_conditions_to_multichamber(feat_trk, meta)

    d_list.append(feat_trk)

df0 = pd.concat(d_list)
#%% # Reassign IDs for multi-day data
curr_id = 0
for (acq, idnum), df_ in df0.groupby(['acquisition', 'id']):
    #print(acq, idnum)
    df0.loc[(df0['acquisition']==acq) & (df0['id']==idnum), 'global_id'] = curr_id
    curr_id += 1
df0['global_id'].unique()    

print(df0['acquisition'].unique())
# %%
mean_vel = df0[df0['sex']=='m'].groupby(['winged', 'global_id'])['vel'].mean().reset_index()
#mean_vel

# %%
fig, ax =plt.subplots()
#sns.pointplot(data=df[df['sex']=='m'], ax=ax, x='winged', 
#              hue='acquisition', y='vel')
sns.barplot(data=mean_vel, ax=ax, x='winged', y='vel',
            color=[0.7]*3)
sns.swarmplot(data=mean_vel, ax=ax, x='winged', y='vel', 
              hue='global_id')

# %%
# Try loading JAABA
beh_type = 'chasing'
is_threshold = 0.3
isnot_threshold = 0.14
jaaba_dir = glob.glob(os.path.join(rootdir, 'JAABA_classifiers', 'multichamber*', 'JAABA'))[0]
#%    
#acq = '20240805-1019_fly1-9_Dyak_WT_5do_gh'
#df = df0[(df0['acquisition']==acq)].copy()

df0['copulating'] = True
#df0[beh_type] = False
df_list = []
for acq, df in df0.groupby('acquisition'):
    # Load JAABA scores
    mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
    jaaba_scores = load_jaaba_from_mat(mat_fpath)
    #
    # Find where jaaba_scores is greater than threshold value, set to 1 and otherwise 0
    jaaba_binary = binarize_jaaba_scores(jaaba_scores, is_threshold, isnot_threshold)

    # Merge jaaba_binary_stack with df
    df = df.merge(jaaba_binary.rename(columns={'score': beh_type}), how='inner', on=['frame', 'id'])
    df[beh_type] = df[beh_type].astype('bool')

    # Check for copulations
    actions_fpath = glob.glob(os.path.join(rootdir, assay, acq, '2024*', '*-actions.mat'))[0] 
    cop_dict = get_copulation_frames(actions_fpath)
     
    for i, df_ in df.groupby('id'):
        cop_ix = cop_dict[i]
        # Only take frames up to copulation
        if cop_ix == -1:
            df.loc[(df['id']==i), 'copulating'] = False
        else:
            df.loc[(df['id']==i) & (df['frame']<=cop_ix), 'copulating'] = False

    # Reassign to df0
    #df0.loc[df0['acquisition']==acq] = df
    df_list.append(df)

#%%
df0 = pd.concat(df_list)

#%%
df = df0[(df0['sex']=='m') & (~df0['copulating'])
         & (df0['chasing'])].copy()

mean_vel = df0[(df0['sex']=='m')
               & (df0['copulating']==0)].groupby(['winged', 'global_id'])['vel'].mean().reset_index()

mean_vel_chasing = df.groupby(['winged', 'global_id'])['vel'].mean().reset_index()

#%%
fig, ax =plt.subplots()
sns.barplot(data=mean_vel, ax=ax, x='winged', y='vel',
            color=[0.7]*3)
sns.swarmplot(data=mean_vel, ax=ax, x='winged', y='vel', 
              hue='global_id')

fig, ax =plt.subplots()
sns.barplot(data=mean_vel_chasing, ax=ax, x='winged', y='vel',
            color=[0.7]*3)
sns.swarmplot(data=mean_vel_chasing, ax=ax, x='winged', y='vel', 
              hue='global_id')

# %%

# Plot velocity with and without chasing
plotdf = df[(df['score']==1) & (df['sex']=='m')].copy()
fig, ax =plt.subplots()
sns.histplot(data=plotdf, ax=ax,
             x='vel', hue='winged')

fig, ax = plt.subplots()
sns.histplot(data=df[df['sex']=='m'], ax=ax,

             x='vel', hue='winged')

#%% 

import transform_data.relative_metrics as rel
import cv2

#
# Apply relative tranfsormation to 1 arena, 1 acquisition:
acq ='20240809-0956_fly1-9_Dyak_WT_3do_gh' 
curr_arena = 1
df_ = df0[(df0['acquisition']==acq)
          & (df0['fly_pair']==curr_arena)].copy()
print(df_['id'].unique())

cop_ix = df_[df_['copulating']].iloc[0].name if True in df_['copulating'].unique() else None
print(cop_ix)
#%%
acqdir = os.path.join(rootdir, assay, acq)
cap = rel.get_video_cap(acqdir, movie_fmt='.avi')

# N frames should equal size of DCL df
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

print(n_frames, frame_width, frame_height)

#%%

fps = 60.
df_['sec'] = df_['frame']/fps
df_['sec']
#%%
df_ = rel.do_transformations_on_df(df_, frame_width, frame_height, 
                             cop_ix=cop_ix)
print(df_.head())
print(df_.columns)

#%%
plotd = df_[(df_['id']==0)
            & (df_['chasing'])].copy()

fig, ax = plt.subplots()
sns.scatterplot(data=plotd, x='targ_rel_pos_x', y='targ_rel_pos_y', 
                ax=ax, s=0.1)


#%%

# %% Plot relative position

min_vel = 10
max_facing_angle = np.deg2rad(20)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang = np.deg2rad(45)

court_ = df0[ (df0['id']==0)
             & (df0['vel'] > min_vel)
             & (df0['targ_pos_theta'] <= max_targ_pos_theta)
             & (df0['targ_pos_theta'] >= min_targ_pos_theta)
             & (df0['facing_angle'] <= max_facing_angle)
             & (df0['min_wing_ang'] >= min_wing_ang)
             & (df0['dist_to_other'] <= max_dist_to_other)].copy()

# Get female-centered frames
f_list = []
for acq, df_ in court_.groupby('acquisition'):
    
    f2_ = df0[ (df0['frame'].isin(df_['frame']))
             & (df0['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f2_)
f2 = pd.concat(f_list)

#%%
acq ='20240805-1019_fly1-9_Dyak_WT_5do_gh' 
df_ = df0[df0['acquisition']==acq].copy()

f2_ = df_[ (df_['frame'].isin(court_['frame']))
          & (df_['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()

fig, ax = pl.subplots()
sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap='magma', vmin=0, vmax=20) # %%
ax.plot(0, 0, 'w', markersize=3, marker='o')
# ax.set_xlim([])
ax.set_aspect(1)
ax.set_xlim([-800, 800])
ax.set_ylim([-800, 800])