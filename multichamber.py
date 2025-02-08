
#%%
import os
import glob
import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import utils as util

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

    feat_fpath = glob.glob(os.path.join(acqdir, '2024*', '*-feat.mat'))[0]
    feat_fpath
    #%

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

    feat_trk.shape, feat_.shape, trk_.shape
    # %
    # Get meta info for current acquisition
    meta = allmeta[allmeta['acquisition']==acq]

    # Check that the number of unique fly ids in feat_trk is 2x as the number of unique fly pairs in meta
    assert len(feat_trk['id'].unique()) == 2*len(meta['fly_num'].unique()), 'Incorrect fly ID to pair assignment'

    # Assign fly pair number to feat_trk
    for i in meta['fly_num'].unique():
        feat_trk.loc[feat_trk['id']==2*i-1, 'fly_pair'] = i # this is female
        feat_trk.loc[feat_trk['id']==2*i-2, 'fly_pair'] = i # this is the male

    # Assign sex
    feat_trk['sex'] = 'f'
    feat_trk.loc[feat_trk['id'] % 2 == 0, 'sex'] = 'm'
    feat_trk['sex'] = feat_trk['sex'].astype('category')
    print(feat_trk.groupby('fly_pair')['sex'].unique())

    #% Assign conditions
    # Get pair number for wingless 
    wingless_pairs = [i-1 for i in meta[(meta['acquisition']==acq) \
                        & (meta['manipulation_male']=='wingless')]['fly_num'].unique()]
    # Assign wing or wingless for each pair
    feat_trk['winged'] = 'winged'
    feat_trk.loc[feat_trk['fly_pair'].isin(wingless_pairs), 'winged'] = 'wingless' #False

    #feat_trk['winged'] = feat_trk['winged'].astype('category')

    print(feat_trk.groupby('winged')['id'].unique())

    d_list.append(feat_trk)

df0 = pd.concat(d_list)


#%%
# Reassign IDs
curr_id = 0
for (acq, idnum), df_ in df0.groupby(['acquisition', 'id']):
    print(acq, idnum)
    df0.loc[(df0['acquisition']==acq) & (df0['id']==idnum), 'global_id'] = curr_id
    curr_id += 1

df0['global_id'].unique()    

# %%
#fig, ax =pl.subplots()
#sns.histplot(data=feat_trk[feat_trk['sex']==0], ax=ax,
#             x='vel', hue='winged')

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
jaaba_dir = glob.glob(os.path.join(rootdir, 'JAABA_classifiers', 'multichamber*', 'JAABA'))[0]

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
        print(k)
        if cop_ix_dict[k] != -1:
            cop_ix_dict[k+1] = cop_ix_dict[k] # assign female to have same copulation frame
        
    return cop_ix_dict


    
acq = '20240805-1019_fly1-9_Dyak_WT_5do_gh'
mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
jaaba_scores = load_jaaba_from_mat(mat_fpath)

#%
# Find where jaaba_scores is greater than threshold value, set to 1 and otherwise 0
is_threshold = 0.3
isnot_threshold = 0.14
jaaba_binary = jaaba_scores.copy()
jaaba_binary[jaaba_binary >= is_threshold] = 1
jaaba_binary[jaaba_binary <= isnot_threshold] = 0

#%%

# Stack jaaba_binary so that each row is a frame and fly, columns are the frame number and fly number
jaaba_binary_stack = jaaba_binary.stack().reset_index()
jaaba_binary_stack.columns = ['frame', 'id', 'score']

# %%
df = df0[(df0['acquisition']==acq)].copy()

for i, df_ in df.groupby('id'):
    # assign frame number
    df.loc[df['id']==i, 'frame'] = np.arange(len(df_))

#%%
# Merge jaaba_binary_stack with df
df = df.merge(jaaba_binary_stack, how='inner', on=['frame', 'id'])
df['score'] = df['score'].astype('category')

# %%

# Plot velocity with and without chasing
plotdf = df[(df['score']==1) & (df['sex']=='m')].copy()
fig, ax =plt.subplots()
sns.histplot(data=plotdf, ax=ax,
             x='vel', hue='winged')

fig, ax = plt.subplots()
sns.histplot(data=df[df['sex']=='m'], ax=ax,
             x='vel', hue='winged')
# %%
actions_fpath = glob.glob(os.path.join(rootdir, assay, acq, '2024*', '*-actions.mat'))[0] 
cop_dict = get_copulation_frames(actions_fpath)
cop_dict
# %%
