
#%%
import os
import glob
import pandas as pd

import utils as util

#%%
rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
assay = '38mm_dyad'
acq = '20241101-0940_fly1_Dbia-WT_3do_gh'

#%%
assay = 'multichamber'
acq = '20240805-1019_fly1-9_Dyak_WT_5do_gh'

acqdir = os.path.join(rootdir, assay, acq)

feat_fpath = glob.glob(os.path.join(acqdir, '2024*', '*-feat.mat'))[0]
feat_fpath
# %%

#% Load feature mat
feat_ = util.load_feat(acqdir) #, subfolder=None)
trk_ = util.load_tracks(acqdir)#, subfolder=None)

# %%
# select unique columns of trk_ using common_cols to exclude
# columns that are common to both feat_ and trk_
unique_cols = [c for c in trk_.columns if c not in feat_.columns]

# merge feat_ and trk_ by index, drop duplicate columns
feat_trk = pd.merge(feat_, trk_[unique_cols], how='inner', left_index=True, right_index=True,
                    suffixes=('', ''))

feat_trk.shape, feat_.shape, trk_.shape
# %%
meta_fpath = os.path.join(rootdir, assay, 'courtship_free_behavior_data - raw data 3x3 .csv')
allmeta = pd.read_csv(meta_fpath)
allmeta
# %%

# Get meta info for current acquisition
meta = allmeta[allmeta['acquisition']==acq]
print(meta.groupby('fly_num')['manipulation_male'].unique())

# Check that the number of unique fly ids in feat_trk is 2x as the number of unique fly pairs in meta
assert len(feat_trk['id'].unique()) == 2*len(meta['fly_num'].unique()), 'Incorrect fly ID to pair assignment'

# Assign fly pair number to feat_trk
for i in meta['fly_num'].unique():
    feat_trk.loc[feat_trk['id']==2*i-1, 'fly_pair'] = i # this is female
    feat_trk.loc[feat_trk['id']==2*i-2, 'fly_pair'] = i # this is the male

# Assign sex
feat_trk['sex'] = 'f'
feat_trk.loc[feat_trk['id'] % 2 == 0, 'sex'] = 'm'

print(feat_trk.groupby('fly_pair')['sex'].unique())

#%% Assign conditions

wingless_pairs = [i-1 for i in meta[(meta['acquisition']==acq) \
                    & (meta['manipulation_male']=='wingless')]['fly_num'].unique()]
print(wingless_pairs)

feat_trk['wing_male'] = 'winged'
feat_trk.loc[feat_trk['fly_pair'].isin(wingless_pairs), 'wing_male'] = 'wingless'

print(feat_trk.groupby('wing_male')['id'].unique())
# %%

