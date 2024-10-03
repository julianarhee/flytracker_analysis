#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import pandas as pd
import scipy.io
import h5py
import mat73
import numpy as np
import os
import yaml
import re

import utils as util

#%%
def aggr_matstruct_to_df(mstruct, structname='feat_fly', fps=60):
    d_list = []
    for i, fn in enumerate(file_names):

        n_flies, n_frames, n_flags = mstruct[structname][i]['data'].shape
        feat_names = mstruct[structname][i]['names']

        tmp_list=[]
        for fly_ix in range(n_flies):
            tmpdf = pd.DataFrame(data=mstruct[structname][i]['data'][fly_ix, :], 
                                columns=feat_names)
            tmpdf['id'] = fly_ix
            tmpdf['frame'] = np.arange(len(tmpdf))
            tmp_list.append(tmpdf)
        feat_ = pd.concat(tmp_list, axis=0, ignore_index=True)
        feat_['filename'] = fn
        d_list.append(feat_)

    feat = pd.concat(d_list).reset_index(drop=True)
    feat = util.add_frame_nums(feat, fps=fps)

    cols = ['_'.join(f.split(' ')) if len(f.split(' '))>0 else f for f in feat.columns]
    feat.columns = cols

    return feat


#%%
#mat_fpath = '/Volumes/Giacomo/MATLAB/projector_data_20240320.mat'
# mat_fpath = '/Volumes/Giacomo/MATLAB/projector_data_all_20240321.mat'

#mat_fpath = '/Volumes/Giacomo/MATLAB/projector_data_single_20240320.mat'

#mat_fpath = '/Volumes/Giacomo/MATLAB/projector_data_elegans_all_20240325.mat'

# Load the .mat file
#mat_fpath = '/Volumes/Giacomo/MATLAB/free_behavior_data_mel_yak_20240403.mat'
#minerva_base = '/Volumes/Giacomo/MATLAB Data'
minerva_base = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad'
fn = 'free_behavior_data_mel_yak_20240403'
#fn = 'projector_data_mel_yak_20240330'
#fn = 'projector_data_elegans_all_20240325'
mat_fpath = os.path.join(minerva_base, '{}.mat'.format(fn))
#mat = scipy.io.loadmat(mat_fpath)
mat = mat73.loadmat(mat_fpath)

#%%
assay = '38mm-dyad'
#assay = '2d-projector' 

if assay == '2d-projector':
    destdir = '/Volumes/Juliana/2d-projector-analysis/processed'
    alt_destdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/2d-projector/JAABA'
else:
    assert assay == '38mm-dyad'
    destdir = '/Volumes/Juliana/free-behavior-analysis/MF/FlyTracker/38mm_dyad'
    alt_destdir = '/Users/julianarhee/Documents/rutalab/projects/courtship/data/MF/38mm-dyad/giacomo'
if not os.path.exists(destdir):
    os.makedirs(destdir)
if not os.path.exists(alt_destdir):
    os.makedirs(alt_destdir)

#%%
mstruct = mat['exported']
n_files = len(mstruct['data'])
file_names = mstruct['name']

array_vars = ['feat_fly', 'data', 'track_fly', 'names']

d_list = []
for i, fn in enumerate(file_names):

    # Create a dataframe for each file 
    names = [n[0] for n in mstruct['names'][i]]
    df_ = pd.DataFrame(data=mstruct['data'][i], 
                 columns=names)
    df_['filename'] = fn
    df_['acquisition'] = fn

    # Add single val variables
    single_vars = dict((k, mstruct[k][i])\
                 for k in mstruct.keys() if k not in array_vars)
    for k, v in single_vars.items():
        if not isinstance(v, str) and v is not None:
            single_vars[k] = int(v) 
        elif isinstance(v, str):
            single_vars[k] = v
        elif v is None:
            if k=='age':
                age = int(re.findall(r'(\d+)do', fn)[0])
                single_vars[k] = age
    df_ = df_.assign(**single_vars)

    df_['frame'] = np.arange(len(df_))
    if 'mel' in df_['species'].unique()[0]:
        df_['strain'] = df_['species'].copy()
        df_['species'] = 'Dmel'
    else:
        df_['strain'] = df_['species'].copy()
        df_['species'] = 'Dyak'
    d_list.append(df_)
   
df = pd.concat(d_list)

#%%
#outfile = os.path.join(destdir, 'jaaba_transformed_data.pkl')

fname = os.path.splitext(os.path.split(mat_fpath)[-1])[0]
outfile = os.path.join(destdir, '{}_jaaba.pkl'.format(fname)) # os.path.splitext(os.path.split(mat_fpath)[-1])[0])

#if '20240321' in mat_fpath:
#    outfile = os.path.join(destdir, 'jaaba_transformed_data_transf.pkl')
#elif 'elegans' in mat_fpath:
#    outfile = os.path.join(destdir, 'jaaba_transformed_data_elegans.pkl')
print("Saved: {}".format(outfile))

df.to_pickle(outfile)

df.to_pickle(os.path.join(alt_destdir, os.path.split(outfile)[1]))

# %% FEAT
feat = aggr_matstruct_to_df(mstruct, structname='feat_fly')

# get units
feat_names = mstruct['feat_fly'][0]['names']
feat_units = mstruct['feat_fly'][0]['units']
unit_dict = dict(zip(feat_names, feat_units))

# save units
units_fpath = os.path.join(destdir, 'units.yaml')
with open(units_fpath, 'w') as f:
    yaml.dump(unit_dict, f)

# save df
feat_outfile = os.path.join(destdir, '{}_feat.pkl'.format(fname))

feat.to_pickle(feat_outfile)
print("Saved: {}".format(feat_outfile))

# %% TRK
trk = aggr_matstruct_to_df(mstruct, structname='track_fly')

#if 'elegans' in mat_fpath:
#    trk_outfile = os.path.join(destdir, 'trk_elegans.pkl')
#else:
trk_outfile = os.path.join(destdir, '{}_trk.pkl'.format(fname))


trk.to_pickle(trk_outfile)
print("Saved: {}".format(trk_outfile))

#%%
#feat_outfile = os.path.join(destdir, 'feat.pkl')
#trk_outfile = os.path.join(destdir, 'trk.pkl')

#feat = pd.read_pickle(feat_outfile) 
#trk = pd.read_pickle(trk_outfile)   

#print(feat.shape, trk.shape)

# %%
assert trk.shape[0] == feat.shape[0], "Number of rows in trk and feat do not match"

#%%
trk_cols = [c for c in trk.columns if c not in feat.columns]
print(trk_cols)

#%%
ftdf = pd.concat([trk[trk_cols], feat], axis=1)
print(trk.shape, ftdf.shape)

#%%
#if 'elegans' in mat_fpath:
#    ftdf_outfile = os.path.join(destdir, 'ft_elegans.pkl')
#else:
ftdf_outfile = os.path.join(destdir, '{}_flytracker.pkl'.format(fname))

ftdf.to_pickle(ftdf_outfile)
print("Saved: {}".format(ftdf_outfile))

ftdf.to_pickle(os.path.join(alt_destdir, os.path.split(ftdf_outfile)[1]))



# %%

