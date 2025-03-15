
#%%
import os
import glob

import numpy as np

import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import utils as util
import transform_data.relative_metrics as rel
import cv2

#%%

import plotting as putil
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=12)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'


#%%
def load_feat_and_trk(acqdir):
    feat_fpath = glob.glob(os.path.join(acqdir, '*', '*-feat.mat'))[0]
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

def meta_flynum_to_ft_id(array_size='3x3'):
    '''
    Converts fly_num from Metadata spreadsheet (counted top-bottom, left-right)
    to FlyTracker's counting (from left-right, top-bottom)
    '''
    if array_size=='3x3':
        flynum_to_id = {
            1: 0, #1,
            2: 6, #7,
            3: 12, #13,
            4: 2, #3,
            5: 8, #9, 
            6: 14, #15,
            7: 4, #5,
            8: 10, #11,
            9: 16, #17
        }
    elif array_size == '2x2':
        flynum_to_id = {
            1: 0, #1,
            2: 4, #5,
            3: 2, #3,
            4: 6, #7
        }
    else:
        raise ValueError('Order not recognized')
    
    return flynum_to_id

def assign_conditions_to_multichamber(feat_trk, meta, array_size='3x3'):
    ''' 
    Assign winged/wing (or other conditions) to high-throughput multichamber data.
    Also adds acquisition and frame numnber. Assumes even is male, odd female.
    NOTE: meta fly_num is 1-indexed
    '''
    # Assign fly pair number to feat_trk
    id_lut = meta_flynum_to_ft_id(array_size=array_size)

    for i in meta['fly_num'].unique():
        
        feat_trk.loc[feat_trk['id']==id_lut[i], 'fly_pair'] = i # this is MALE
        feat_trk.loc[feat_trk['id']==id_lut[i]+1, 'fly_pair'] = i # this is the FEMALE

    #% Assign conditions
    # Get pair number for wingless 
    wingless_pairs = [i for i in meta[(meta['acquisition']==acq) \
                        & (meta['manipulation_male']=='wingless')]['fly_num'].unique()]
    # Assign wing or wingless for each pair
    feat_trk['condition'] = 'winged'
    feat_trk.loc[feat_trk['fly_pair'].isin(wingless_pairs), 'condition'] = 'wingless' #False
    feat_trk['condition'] = feat_trk['condition'].astype('category')

    return feat_trk


def assign_strain_to_multichamber(feat_trk, meta, array_size='3x3'):
    ''' 
    Assign winged/wing (or other conditions) to high-throughput multichamber data.
    Also adds acquisition and frame numnber. Assumes even is male, odd female.
    '''
    # Assign fly pair number to feat_trk
    id_lut = meta_flynum_to_ft_id(array_size=array_size)

    for i in meta['fly_num'].unique():
        
        feat_trk.loc[feat_trk['id']==id_lut[i], 'fly_pair'] = i # this is MALE
        feat_trk.loc[feat_trk['id']==id_lut[i]+1, 'fly_pair'] = i # this is the FEMALE

        feat_trk.loc[feat_trk['fly_pair']==i, 'strain'] = meta[meta['fly_num']==i]['strain_male'].values[0]

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
    try:
        cop_ixs = [v[cop_name_ix][0][0] if len(v[cop_name_ix][0])>0 else -1 for v in mat['bouts']]
    except IndexError as e:
        cop_ixs = [v[cop_name_ix][0][0] if len(v[cop_name_ix])>0 else -1 for v in mat['bouts']]
    cop_ix_dict = dict((k, v) for k, v in enumerate(cop_ixs))

    # Find which IDs copulated
    copulation_ids = [k for k, v in cop_ix_dict.items() if v != -1]
    
    # Assign female ID to have same copulation index
    for k in copulation_ids: #cop_ix_dict.keys():
        if k % 2 > 0: # if odd, female was assigned cop ix in matlab
            cop_ix_dict[k-1] = cop_ix_dict[k] # Corresponding male should be 1 less
        else:
        #if cop_ix_dict[k] != -1: # male was correctly assigned (even)
            cop_ix_dict[k+1] = cop_ix_dict[k] # assign female to have same copulation frame
        
    return cop_ix_dict

#%%

# Test on 1 chamber, 38mm

srcdir = '/Volumes/Giacomo/free_behavior_data'

acq = '20240116-1100-fly2-yakWT_4do_sh_yakWT_4do_gh'
acqdir = os.path.join(srcdir, acq)
os.path.exists(acqdir)

cop_ix = None

# df_ = load_feat_and_trk(acqdir) 
# for i, d_ in df_.groupby('id'):
#     df_.loc[df_['id']==i, 'frame'] = np.arange(len(d_))
fps=60
mov_is_upstream=False
subfolder='*'
#mat_fpaths = util.get_mat_paths_for_all_vids(acqdir, subfolder=subfolder, ftype='feat')
#feat_ = util.load_feat(acqdir, subfolder=subfolder)
calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                calib_is_upstream=mov_is_upstream,
                                                subfolder=subfolder,
                                                filter_ori=True)
     
#df_ = f2[f2['acquisition']==acq].copy()
cap = rel.get_video_cap(acqdir, movie_fmt='.avi')
# N frames should equal size of DCL df
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
#print(frame_width, frame_height) # array columns x array rows
# switch ORI
trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
df_ = rel.do_transformations_on_df(trk_, frame_width, frame_height, 
                                feat_=feat_, cop_ix=cop_ix,
                                flyid1=0, flyid2=1, get_relative_sizes=False)

##%
#fps = 60.
#df_['sec'] = df_['frame']/fps
##df_['sec']
##%
#flyid1 = min(df_['id'].unique())
#flyid2 = max(df_['id'].unique())
#transf_df = rel.do_transformations_on_df(df_, frame_width, frame_height, 
#                            cop_ix=cop_ix, flyid1=flyid1, flyid2=flyid2,
#                            verbose=True, get_relative_sizes=False)


#%% LOAD PROCESSED

srcdir = '/Volumes/Juliana/free_behavior_analysis/38mm_dyad/MF/FlyTracker/processed'
single_fly_srcdir = os.path.join(os.path.split(srcdir)[0], 'processed_mats')
mat_type = 'df'
found_fns = glob.glob(os.path.join(single_fly_srcdir, '{}*{}.pkl'.format(acq, mat_type)))
print(found_fns)
fp = found_fns[0]
transf_df2 = pd.read_pickle(fp)
transf_df2.head()

#print(df_.head())
#%%

min_vel = 10
max_facing_angle = np.deg2rad(45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160)
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

transf_df = df_.copy()
df = transf_df.copy() 
court_ = df[(df['id']==0) #& (ftjaaba['chasing']==1)
            & (df['vel']> min_vel)
            & (df['targ_pos_theta'] <= max_targ_pos_theta)
            & (df['targ_pos_theta'] >= min_targ_pos_theta)
            & (df['facing_angle'] <= max_facing_angle)
            #& (df['max_wing_ang'] >= min_wing_ang)
            & (df['dist_to_other'] <= max_dist_to_other)].copy()

f2_ = df[ (df['frame'].isin(court_['frame']))
          & (df['id']==1)].copy() #wing_ext[wing_ext['id']==1].copy()
    

cmap = 'magma' #'YlOrBr'
stat = 'probability' #'count' #count'
vmax=0.0015 if stat=='probability' else 250
fig, ax = plt.subplots()
sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=100)

ax.set_box_aspect(1)




#%%
rootdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee'
srcdir = os.path.join(rootdir, 'caitlin_data')

# ----------------------------------------
#experiment = 'ht_winged_vs_wingless'
experiment = '38mm_yakstrains'
# ----------------------------------------
if experiment == 'ht_winged_vs_wingless':

    #acq = '20241101-0940_fly1_Dbia-WT_3do_gh' 
    assay = 'multichamber_20mm_winged_v_wingless' #'multichamber'
    acqs = ['20240805-1019_fly1-9_Dyak_WT_5do_gh', # annotated
            '20240809-0956_fly1-9_Dyak_WT_3do_gh',
            '20240812-0950_fly1-9_Dyak_WT_5do_gh',  # annotated
            '20240819-0945_fly1-9_Dyak_WT_5do_gh'    # annotated
            ]
    array_size = '3x3'
    figdir = os.path.join(rootdir, 'free_behavior/ht_winged_vs_wingless/figures')
elif experiment == '38mm_yakstrains':
    # STRAINS -------------------------------------------------
    assay = 'multichamber_38mm_2x2_yakstrains'
    acq_dirs = glob.glob(os.path.join(srcdir, assay, '20*'))
    acqs = [os.path.split(a)[-1] for a in acq_dirs]
    print(acqs)
#    acqs = ['20250226-1013_fly1-4_Dyak-WT_2do_gh',
#            '20250306-0917_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
#            '20250307-1034_fly1-4_Dyak-cost-abid-tai-cy_3do_gh',
#            '20250307-1145_fly1-4_Dyak-abid-tai-cy_3do_gh',
#            '20250310-0930_fly1-4_Dyak-tai-abid-cy23_5do_gh',
#            '20250310-1050_fly1-4_Dyak-tai-abid-cy23_5do_gh'
#            ]
    array_size = '2x2'
    figdir = os.path.join(rootdir, 'free_behavior/38mm_yakstrains/figures')

figid = os.path.join(srcdir, assay)
print(figid)

if not os.path.exists(figdir):
    os.makedirs(figdir)
   
processed_outdir = os.path.join(os.path.split(figdir)[0], 'processed') 
if not os.path.exists(processed_outdir):
    os.makedirs(processed_outdir)
    
# %% Load metadata
#meta_fpath = os.path.join(rootdir, assay, 'courtship_free_behavior_data - raw data 3x3 .csv')
meta_fpaths = glob.glob(os.path.join(srcdir, assay, '*.csv'))
assert len(meta_fpaths) == 1, 'More than one metadata file found'
meta_fpath = meta_fpaths[0]
allmeta = pd.read_csv(meta_fpath)
if 'acquisition' not in allmeta.columns:
    allmeta['acquisition'] = allmeta['file_name']
    
allmeta.head()
#%%
create_new=True
fps=60
mov_is_upstream=False
subfolder='*'
cop_ix=None
filter_ori = experiment!='ht_winged_vs_wingless'
print(filter_ori)

no_actions = []
d_list = []
for acq in acqs:
    acqdir = os.path.join(srcdir, assay, acq)
    #feat_trk = load_feat_and_trk(acqdir)
    print("Processing {}".format(acq))
   
    out_fpath = os.path.join(processed_outdir, '{}_df.pkl'.format(acq))
    if not create_new and os.path.exists(out_fpath):
        print("Already processed: {}".format(acq))
        continue
    calib_, trk_, feat_ = util.load_flytracker_data(acqdir, fps=fps, 
                                                calib_is_upstream=mov_is_upstream,
                                                subfolder=subfolder,
                                                filter_ori=True)
    if 'acquisition' not in trk_.columns:
        trk_['acquisition'] = acq
    # Get meta info for current acquisition
    meta = allmeta[allmeta['acquisition']==acq]

    # Check that the number of unique fly ids in feat_trk is 2x as the number of unique fly pairs in meta
    assert len(trk_['id'].unique()) == 2*len(meta['fly_num'].unique()), 'Incorrect fly ID to pair assignment'

    # Assign conditions
    trk_ = assign_sex(trk_)
    trk_ = assign_frame_number(trk_)
    if experiment=='ht_winged_vs_wingless': 
        print("Assigning winged conditions to multichamber data")
        trk_ = assign_conditions_to_multichamber(trk_, meta, array_size=array_size)
    elif experiment=='38mm_yakstrains':
        print("Assigning strain to multichamber data")
        trk_ = assign_strain_to_multichamber(trk_, meta, array_size=array_size)

     # Check for copulations
    try:
        actions_fpath = glob.glob(os.path.join(srcdir, assay, acq, '*', '*-actions.mat'))[0] 
        cop_dict = get_copulation_frames(actions_fpath)     
    except IndexError:
        no_actions.append(acq)
        cop_dict = dict((k, -1) for k in trk_['id'].unique()) 
        
    # for i, df_ in df.groupby('id'):
    #     cop_ix = cop_dict[i]
    #     # Only take frames up to copulation
    #     if cop_ix == -1:
    #         df.loc[(df['id']==i), 'copulating'] = False
    #     else:
    #         df.loc[(df['id']==i) & (df['frame']<=cop_ix), 'copulating'] = False
                   
    # N frames should equal size of DCL df
    cap = rel.get_video_cap(acqdir, movie_fmt='.avi')
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width, frame_height) # array columns x array rows
    # switch ORI
    trk_['ori'] = -1*trk_['ori'] # flip for FT to match DLC and plot with 0, 0 at bottom left
    # do transformations:
    acq_dfs = []
    for flypair, curr_trk in trk_.groupby('fly_pair'):
        curr_feat = feat_.loc[curr_trk.index].copy()
        flyid1 = curr_trk['id'].min() # even 
        flyid2 = curr_trk['id'].max() # odd
        
        cop_ix = cop_dict[flyid1]
        
        #print("Fly pair {}: ids=({}, {})".format(flypair, flyid1, flyid2))
        transf_df = rel.do_transformations_on_df(curr_trk, frame_width, frame_height, 
                                   feat_=curr_feat, cop_ix=cop_ix,
                                   flyid1=flyid1, flyid2=flyid2, get_relative_sizes=False)
        
        assert len(transf_df[transf_df['id']==flyid1]) == len(transf_df[transf_df['id']==flyid2]), 'Different number of frames for each fly'
        assert len(transf_df[transf_df['id']==flyid1]['frame']) == len(transf_df[transf_df['id']==flyid1]['frame'].unique()), 'Non-unique frames'
        assert transf_df.shape[0] == transf_df.index[-1]+1, "Bad frame indexing"
        acq_dfs.append(transf_df)
    acq_df = pd.concat(acq_dfs)
    
    # Save processed df
    #out_fpath = os.path.join(processed_outdir, '{}_df.pkl'.format(acq))
    acq_df.to_pickle(out_fpath) 
     
    d_list.append(acq_df)

df0 = pd.concat(d_list)

#% # Reassign IDs for multi-day data
curr_id = 0
for (acq, idnum), df_ in df0.groupby(['acquisition', 'id']):
    #print(acq, idnum)
    df0.loc[(df0['acquisition']==acq) & (df0['id']==idnum), 'global_id'] = curr_id
    curr_id += 1
df0['global_id'].unique()    

print(df0['acquisition'].unique())
# %%
# Get mean velocity by condition
if experiment=='ht_winged_vs_wingless':
    grouper='condition'
elif experiment=='38mm_yakstrains':
    grouper='strain'
        
mean_vel = df0[df0['sex']=='m'].groupby([grouper, 'global_id'])['vel'].mean().reset_index()
#mean_vel

# %% Plot mean velocity by condition

fig, ax =plt.subplots(figsize=(4,3))
#sns.pointplot(data=df[df['sex']=='m'], ax=ax, x='winged', 
#              hue='acquisition', y='vel')
sns.barplot(data=mean_vel, ax=ax, x=grouper, y='vel',
            color=[0.7]*3, width=0.5, fill=False)
sns.swarmplot(data=mean_vel, ax=ax, x=grouper, y='vel', 
              hue='global_id', palette='tab20', legend=False)
ax.set_box_aspect(1)
# rotate x-tick labels
plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_{}'.format(experiment)
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
c1 = 'r'
c2 = 'cornflowerblue'
if grouper == 'condition':
    grouper_palette = {'winged': 'r',
                       'wingless': 'cornflowerblue'}
else:
    grouper_palette = 'cubehelix'
error_type = 'ci'

# %% ADD JAABA INFO

has_jaaba= False
beh_type = 'chasing'
is_threshold = 0.3
isnot_threshold = 0.14
jaaba_dir = glob.glob(os.path.join(srcdir, 'JAABA_classifiers', 'multichamber*', 'JAABA'))[0]
#%    
#acq = '20240805-1019_fly1-9_Dyak_WT_5do_gh'
#df = df0[(df0['acquisition']==acq)].copy()
# First test 1 video
# acq = '20240805-1019_fly1-9_Dyak_WT_5do_gh'
# acq = '20240809-0956_fly1-9_Dyak_WT_3do_gh'
# acq = '20240812-0950_fly1-9_Dyak_WT_5do_gh'

#n_ids = jaaba_scores.shape[1]
#n_pairs = n_ids/2 # should be 9
if has_jaaba:
    nr=3; nc=3;
    for acq in acqs:
        mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
        jaaba_scores = load_jaaba_from_mat(mat_fpath)
        
        fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True)
        ix=0
        for ai, ax in enumerate(axn.flat):
            currcond = df0[(df0['acquisition']==acq)
                           & (df0['id']==ix)]['condition'].unique()[0]
            currcolor = grouper_palette[currcond]
            ax.hist(jaaba_scores[ix], label='male', color=currcolor)
            ax.hist(jaaba_scores[ix+1], label='female', color='darkgrey') #, legend=0)
            ax.axvline(is_threshold, color=bg_color, linestyle='--', lw=0.5)
            ax.set_title('{}: ids {}, {}'.format(currcond, ix, ix+1), loc='left', fontsize=4) 
            ix += 2
        legh = putil.custom_legend(
                labels=['winged-male', 'wingless-male', 'female'],
                colors=[grouper_palette['winged'], grouper_palette['wingless'], 'darkgrey'])
        ax.legend(handles=legh, frameon=False, loc='upper left', bbox_to_anchor=(1,1))
        fig.suptitle(acq, x=0.5, y=0.95, fontsize=8)
    
        putil.label_figure(fig, figid)  
        figname = 'jaaba_scores_{}_{}'.format(beh_type, acq)
        plt.savefig(os.path.join(figdir, '{}.png'.format(figname))) 
    
#%% 
#df0[beh_type] = False
binarize_jaaba = True
df_list = []
for acq, df in df0.groupby('acquisition'):
    
    if has_jaaba:
        # Load JAABA scores
        mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
        jaaba_scores = load_jaaba_from_mat(mat_fpath)
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
# Compare velocity overall vs. velocity during CHASING bouts
has_jaaba = False    
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

if has_jaaba:
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
              
mean_vel = ftj[(ftj['sex']=='m')].groupby([grouper, 'global_id'])['vel'].mean().reset_index()
mean_vel_chasing = chasedf.groupby([grouper, 'global_id'])['vel'].mean().reset_index()

fig, axn = plt.subplots(1,2, sharex=True, sharey=True)
for ai, mean_ in enumerate([mean_vel, mean_vel_chasing]):
    ax=axn[ai]
    sns.barplot(data=mean_, ax=ax, x=grouper, y='vel',
            color=[0.7]*3, width=0.5, fill=False)
    sns.swarmplot(data=mean_, ax=ax, x=grouper, y='vel', 
            hue='global_id', palette='tab20', legend=False)
    ax.set_box_aspect(1)
    ax.set_title('all' if ai==0 else 'chasing')
    ax.set_xlabel('')
    # rotate x-tick labels for current axis
    plt.setp(ax.get_xticklabels(), rotation=90)
# Rotate x-tick labels for both axe
#plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_chasing'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%% Test individual fly's chasing velocities, Male & Female
# CUM DIST -- each fly
if has_jaaba:
    curr_sex = 'm'
    winged_male_ids = ftj[(ftj['sex']==curr_sex) & (ftj['condition']=='winged')]['global_id'].unique()
    wingless_male_ids = ftj[(ftj['sex']==curr_sex) & (ftj['condition']=='wingless')]['global_id'].unique()

    c1 = 'r'
    c2 = 'cornflowerblue'
    global_id_winged = dict((k, c1) for k in winged_male_ids)
    global_id_wingless = dict((k, c2) for k in wingless_male_ids)
    # combine dictionaries
    global_id_winged_palette = {**global_id_winged, **global_id_wingless}
    # %
    # Plot velocity with and without chasing
    plotdf = ftj[(ftj['chasing']) & (ftj['sex']==curr_sex)
                & (ftj['facing_angle']<=max_facing_angle)].copy()

    fig, ax =plt.subplots()
    sns.histplot(data=plotdf, ax=ax,
                x='vel', hue='global_id', palette=global_id_winged_palette,
                cumulative=True, common_norm=False, lw=0.5, 
                fill=False, element='step', stat='probability', legend=0)
    legh = putil.custom_legend(labels=['winged', 'wingless'], colors=[c1, c2])
    ax.legend(handles=legh, loc='lower right',  frameon=False)
    ax.set_box_aspect(1)
    ax.set_xlim([0, 50])
    ax.set_title("Cum Dist of Vel during Chasing ({})".format(curr_sex))
    putil.label_figure(fig, figid)
    figname = 'cumdist_vel_chasing_winged_v_wingless_{}'.format(curr_sex)
    plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# IDentify courtship frames
ftj = ftj.reset_index(drop=True)

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
min_vel = 5
max_facing_angle = np.deg2rad(20)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(160) # male unlikely chasing if target is behind...
min_targ_pos_theta = np.deg2rad(-160)
min_wing_ang = np.deg2rad(45)

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

# DEFINE COURTING/SINGING - SPATIAL OCCUPANCY
# ----------------------------------------------------------
use_jaaba= False

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

bins=100
stat='probability'
vmax=0.001 if stat=='probability' else 50
#f2 = f2.reset_index(drop=True)
print(grouper)
for curr_cond, f2_ in f2.groupby(grouper):
    fig, axn = plt.subplots(4, 5, figsize=(5,4), sharex=True, sharey=True)
    for ai, (flypair, d_) in enumerate(f2_.groupby(['acquisition', 'fly_pair'])): 
        ax=axn.flat[ai]
        #curr_pal = 'cornflowerblue' if curr_cond=='winged' else 'violet'
        sns.histplot(data=d_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
                    cmap='magma', stat=stat, vmin=0, vmax=vmax, bins=bins)
        ax.set_aspect(1)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.subplots_adjust(top=0.9, left=0.1)
    fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
    fig.text(0.01, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
    fig.suptitle('Courting, cond={}, jaaba={}'.format(curr_cond, use_jaaba), 
                 x=0.3, y=0.95, fontsize=12) 
    for ax in axn.flat[ai+1:]:
        ax.axis('off') 
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, wspace=0.2)
    
    putil.label_figure(fig, figid) 
    figname = 'spatial_occupancy_{}_per-flypair_jaaba-{}'.format(curr_cond, use_jaaba) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))    


#%%
# ALL 

cmap='magma'
stat='probability'
vmax=0.0005 if stat=='probability' else 250
bins=100
#vmax = 0.002 if stat=='probability' else 250
norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

n_conds = f2[grouper].nunique()

if grouper=='strain':
    nr=2; nc=3;
else:
    nr=1; nc=n_conds;

fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                        figsize=(nc*3,nr*3))
for ai, (cond, f2_) in enumerate(f2.groupby(grouper)):
    ax=axn.flat[ai]
    sns.histplot(data=f2_, x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax, 
             cmap=cmap, stat=stat, vmin=0, vmax=vmax, bins=bins) # %%
    ax.plot(0, 0, 'k', markersize=5, marker='>')
    # ax.set_xlim([])
    ax.set_aspect(1)
    ax.set_title(cond, fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.plot(0, 0, 'w', markersize=5, marker='>')
ax.set_xlim([-300, 300])
ax.set_ylim([-300, 300])
plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)

fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
fig.text(0.05, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
for ax in axn.flat[ai+1:]:
    ax.axis('off')
    
#fig.suptitle('Courting frames, uses_jaaba={}'.format(use_jaaba))
fig.text(0.1, 0.95, 'Male position from female-centered view (jaaba={})'.format(use_jaaba), 
         fontsize=8)
putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                             hue_title=stat)

putil.label_figure(fig, figid)     
figname = 'spatial_occupancy_all-pairs_jaaba-{}'.format(use_jaaba) 
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
