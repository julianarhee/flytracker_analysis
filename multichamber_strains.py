
#%%
import os
import glob
import prettyprinter as pp

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
putil.set_sns_style(plot_style, min_fontsize=7)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

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


def get_path_to_jaaba_scores(jaaba_dir, acq, beh_type='chasing', score_str=''):
   
    try:
        # Load most recent scores file 
        mat_fpath = sorted(glob.glob(os.path.join(jaaba_dir, acq, 
                'scores_{}*{}.mat'.format(beh_type, score_str))))[-1]
    except IndexError:
        print("No JAABA {} scores for {}".format(beh_type, acq))
        return None 

    return mat_fpath

def add_jaaba(df, mat_fpath, beh_type='chasing', is_threshold=5, 
              binarize_jaaba=False, isnot_threshold=0.2):
    '''
    Load scores_chasing.mat file for a given df. Add scores and binarize. 
    NOTE: keeps both all fly IDs
    '''    
    acq = df['acquisition'].unique()[0] 
    try:
        jaaba_scores = load_jaaba_from_mat(mat_fpath)
    except Exception as e:
        traceback.print_exc()
        print("Error loading JAABA scores for {}: {}".format(acq, e)) 
        return None
    #
    # Find where jaaba_scores is greater than threshold value, set to 1 and otherwise 0
    if binarize_jaaba:
        jaaba_binary = binarize_jaaba_scores(jaaba_scores, 
                                is_threshold, isnot_threshold)
        jaaba_scores_stacked = stack_jaaba_scores(jaaba_binary)
    else:
        jaaba_scores_stacked = stack_jaaba_scores(jaaba_scores)
    
    # Merge jaaba_binary_stack with df
    if beh_type in df.columns:
        df = df.drop(columns=[beh_type])    
    if beh_type=='unilateral_extension':
        beh_type = 'singing'
    df = df.merge(jaaba_scores_stacked.rename(columns={'score': beh_type}), 
                how='inner', on=['frame', 'id']) 
    if binarize_jaaba: 
        df[beh_type] = df[beh_type].astype('bool')
    else:
        df['{}_binary'.format(beh_type)] = df[beh_type].ge(is_threshold)

    return df

def aggr_add_jaaba(df0, jaaba_dir, beh_types=['chasing', 'singing'],
                   jaaba_thresholds={'chasing': 5, 'singing': 10}, 
                   binarize_jaaba=False, score_species=True):
    '''
    Loop over acquisitions in df0 and add jaaba scores for each behavior type.
    
    Args:
        df0 (pd.DataFrame): Dataframe with acquisition and fly ids
        jaaba_dir (str): Path to jaaba scores directory
        beh_types (list): List of behavior types to add jaaba scores for
        jaaba_thresholds (dict): Dictionary of thresholds for each behavior type
        binarize_jaaba (bool): Whether to binarize jaaba scores
        score_species (bool): Whether to use species-specific jaaba scores 
    '''
    df_list = []
    no_scores = dict((k, []) for k in beh_types) 
    for acq, df in df0.groupby('acquisition'):
        # Iteratively add to df 
        for beh_type in beh_types:
            is_threshold = jaaba_thresholds[beh_type]

            # Get path to jaaba scores
            if score_species:
                if 'yak' in acq:
                    score_str = 'yak'
                elif 'mel' in acq:
                    score_str = 'mel'
            else:
                score_str = ''
            mat_fpath = get_path_to_jaaba_scores(jaaba_dir, acq, 
                                        beh_type=beh_type, score_str=score_str)
             
            # Load JAABA scores
            df_j = add_jaaba(df, mat_fpath, beh_type=beh_type, 
                            is_threshold=is_threshold, binarize_jaaba=binarize_jaaba) 
            if df_j is None:
                no_scores[beh_type].append(acq)
            else:
                df = df_j.copy()
        # Reassign to df0
        if any([acq in v for v in no_scores.values()]):
            continue
        df_list.append(df) #[~df['copulating']])

    ftj0 = pd.concat(df_list)
    
    return ftj0, no_scores
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
# File paths to saved processed data
aggregate_processed_datafile = os.path.join(local_dir, 
                                        '38mm_strains_df.parquet')
#aggregate_processed_datafile_all = os.path.join(local_dir, 
#                                        '38mm_all_df.parquet')
ftjaaba_datafile = os.path.join(local_dir, 
                                        '38mm_strains_ftjaaba.parquet')
ftjaaba_datafile_single_arena = os.path.join(local_dir,
                                        '38mm_single_arena_ftjaaba.parquet')
ftjaaba_datafile_all = os.path.join(local_dir, '38mm_all_ftjaaba.parquet')

#%%
# Load FTJAABA for all data (M/F, both single arena and 2x2 in 38mm arenas)
# ---------------------------------------------------------------------------
ftj= pd.read_parquet(ftjaaba_datafile_all)
print("Loaded processed: {}".format(ftjaaba_datafile))
ftj.head()

#%%
# Load and process feat-track data and combine 2x2 with single arena data
# -----------------------------------------------------------------------
new_ftjaaba = False # Load strain data 2x2 and create ftjaaba
recombine_ftjaaba_datasets = False # Load single arena data and recombine with 2x2 data
# -----------------------------------------------------------------------
if new_ftjaaba:
    df0 = pd.read_parquet(aggregate_processed_datafile)
    print("Loaded processed: {}".format(aggregate_processed_datafile))
    df0['strain'] = df0['strain'].map(lambda x: x.replace('CS mai', 'CS Mai'))
    df0['strain'] = df0['strain'].map(lambda x: x.replace('CS Mai ', 'CS Mai'))
    df0['strain'] = df0['strain'].map(lambda x: x.replace(' ', '_'))
    #%
    conds = df0[['species', 'strain', 'acquisition', 'fly_pair']].drop_duplicates()
    counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
    print(counts)

#%%
# ==========================================
# Check JAABA scores
# ==========================================
def hist_jaaba_scores_male_female(jaaba_scores, ix, ax=None, 
                        curr_color='r', bg_color='k', is_threshold=0.4):
    #currcolor = grouper_palette[currcond]
    ax.hist(jaaba_scores[ix], label='male', color=curr_color)
    ax.hist(jaaba_scores[ix+1], label='female', color='darkgrey') #, legend=0)
    ax.axvline(is_threshold, color=bg_color, linestyle='--', lw=0.5)
    return ax

no_jaaba_acqs = ['20250320-1025_fly1-4_Dyak-gab_3do_gh',
                 '20250306-0917_fly1-4_Dyak-cost-abid-tai-cy_3do_gh']

acqs = [a for a in df0['acquisition'].unique() if a not in no_jaaba_acqs]

has_jaaba = True
beh_type = 'chasing' #'unilateral_extension'
is_threshold = 5
isnot_threshold = 0.2 #14

if not os.path.exists(os.path.join(figdir, 'acqs')):
    os.makedirs(os.path.join(figdir, 'acqs'))
    
# HISTOGRAM of JAABA scores
# -----------------------------------------
cond_name = 'strain''
if has_jaaba:
    for acq, df_ in df0.groupby('acquisition'):# in acqs[0:nr*nc]:
        if acq in no_jaaba_acqs:
            continue
        mat_fpath = sorted(glob.glob(os.path.join(jaaba_dir, acq, 
                            'scores_{}*.mat'.format(beh_type))))[-1]
        jaaba_scores = load_jaaba_from_mat(mat_fpath)        
        nr=2; nc=int(df_['id'].nunique()/2/nr);
        fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                                figsize=(nc*1.7, nr*1.7))
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
        fig.suptitle(acq, x=0.5, y=0.95, fontsize=5) 
        putil.label_figure(fig, figid)  
        figname = 'jaaba_scores_{}_{}'.format(beh_type, acq)
        plt.savefig(os.path.join(figdir, 'acqs', '{}.png'.format(figname)))   
#%%
# Create FTJAABA: Load and binarize all JAABA scores
# --------------------------------------------------
if new_ftjaaba:
    has_jaaba=True
    jaaba_thresholds = {'chasing': 5, 'unilateral_extension': 10}
    isnot_threshold = 0.2 #14
    binarize_jaaba = False

    ftj0, no_scores = aggr_add_jaaba(df0, jaaba_dir, 
                                    beh_types=['chasing', 'unilateral_extension'],
                                    jaaba_thresholds=jaaba_thresholds, 
                                    binarize_jaaba=binarize_jaaba)
    print("[{}]: No JAABA for the following acquisitions:".format(beh_type))
    pp.pprint(no_scores)
    # del df0
    #%
    # Save
    print("Saving STRAIN ftjaaba to local.")
    ftj0.to_parquet(ftjaaba_datafile, engine='pyarrow',
            compression='snappy')
else: 
    #%
    # %  LOAD FTJAABA strain dataset 1
    # del df0
    ftj0 = pd.read_parquet(ftjaaba_datafile)
    print("Loaded processed: {}".format(ftjaaba_datafile))
    ftj0.head()

#%% 
# Load FTJAABA for single arena data, if needed
if recombine_ftjaaba_datasets:
    # Load second dataset
    localdir2 = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/38mm_dyad/MF/FlyTracker'
    jaaba_dir2 = '/Volumes/Giacomo/JAABA_classifiers/free_behavior'
    dfpath2 = os.path.join(localdir2, 'processed.pkl')
    tmpdf = pd.read_pickle(dfpath2)

    # Get subset of data that has been processed with jaaba
    ftjaaba_fpath_local = os.path.join(localdir2, 'ftjaaba.pkl')
    ftj2_processed = pd.read_pickle(ftjaaba_fpath_local)
    print("Loaded local processed data for other dataset.")
        
    df2 = tmpdf[tmpdf['acquisition'].isin(ftj2_processed['acquisition'].unique())]
    # Add strain info (from GG's meta)
    for acq, df_ in df2.groupby('acquisition'):
        curr_strain = ftj2_processed[ftj2_processed['acquisition']==acq]['strain'].unique()[0]
        df2.loc[df2['acquisition']==acq, 'strain'] = curr_strain

    print(df2['strain'].unique())
    print(df2['acquisition'].nunique())
    del ftj2_processed

    #%%
    # Add jaaba scores for other dataset
    jaaba_thresholds2 = {'chasing': 10, 'singing': 5}
    binarize_jaaba= False

    ftj2, no_scores2 = aggr_add_jaaba(df2, jaaba_dir2, 
                                    beh_types=['chasing', 'singing'],
                                    jaaba_thresholds=jaaba_thresholds2, 
                                    binarize_jaaba=binarize_jaaba)
    print("No JAABA for the following acquisitions:")
    pp.pprint(no_scores2)
    ftj2.head()

    #%% Load FTJAABA of other dataset

    # species  strain                               
    # Dmel     CS_Mai                                    5
    #          SD105N_(Intermediate_Usually_Aroused)    10
    # Dyak     RL_Ruta_Lab                              10
    # Name: fly_pair, dtype: int64
    #%
    ftj2.loc[ftj2['id']==0, 'sex'] = 'm' 
    ftj2.loc[ftj2['id']==1, 'sex'] = 'f' 
    ftj2['fly_pair'] = 1 # Only acquiring 1 at a time

    # Fix strain name
    ftj2.loc[ftj2['strain']=='mel-SD105N', 'strain'] = 'SD105N_(Intermediate_Usually_Aroused)'
    ftj2.loc[ftj2['strain']=='yak-WT', 'strain'] = 'RL_Ruta_Lab'
    ftj2.loc[ftj2['strain']=='mel-Canton-S', 'strain'] = 'CS_Mai'

    #%%
    # save old dataset ftjaaba
    print("Saving old 38mm (single arena) ftjaaba to local.")
    print(ftjaaba_datafile_single_arena)
    # Save
    ftj2.to_parquet(ftjaaba_datafile_single_arena, engine='pyarrow',
            compression='snappy')
        
    #%%
    # Get intersection of columns in ftj0 and ftj2
    missing_cols = np.setdiff1d(ftj0.columns, ftj2.columns)
    shared_cols = np.intersect1d(ftj0.columns, ftj2.columns)

    #%%
    del df2, tmpdf

    #%% 
    # merge ftj[shared_cols] with ftj2[shared_cols]
    ftj = pd.concat([ftj0[shared_cols], ftj2[shared_cols]], axis=0)
    ftj.head()

    #%%
    del ftj0, ftj2

    #%%
    # Save
    print("Saving ALL 38mm ftjaaba to local.")
    print(ftjaaba_datafile_single_arena)
    ftj.to_parquet(ftjaaba_datafile_all, engine='pyarrow',
            compression='snappy')

#%% ==========================================
# START HERE for analysis
# ============================================
#%%
# get counts of each 
conds = ftj[['acquisition', 'species', 'strain', 'fly_pair']].drop_duplicates()
counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
print(counts)

#%%
# check overall velocity between yak and mel 
grouper= ['species', 'strain', 'acquisition', 'fly_pair']  
mean_vel = ftj[ftj['sex']=='m'].groupby(grouper)['vel'].mean().reset_index()
mean_vel.head()

# % plot mean velocity by condition
fig, ax =plt.subplots(figsize=(3,2))
# center barplot and stripplot over each other
sns.barplot(data=mean_vel, ax=ax, x='species', y='vel', color='k', linewidth=1,
            width=0.5, fill=False)
sns.stripplot(data=mean_vel, ax=ax, x='species', y='vel', hue='strain',
            palette='PRGn', dodge=True, jitter=True, linewidth=0.5) #, legend=False)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, title='')
ax.set_xlabel('')
ax.set_box_aspect(1)
putil.label_figure(fig, figid)

figname = 'mean_vel_{}'.format(experiment)
plt.savefig(os.path.join(figdir, figname+'.png'))

#%% 
# add strain name legend for plotting
def add_legend_column_with_n(ftj, key='strain', 
                    grouper=['species', 'strain', 'acquisition', 'fly_pair']):
    conds = ftj[grouper].drop_duplicates()
    counts = conds.groupby([key])['fly_pair'].count()
    ftj['{}_legend'.format(key)] = ftj[[key]].applymap(lambda x: '{} (n={})'.format(x, counts[x]))
  
    return ftj, counts

#% add species to strain name for proper colormap grouping
ftj['strain_name'] = ['{} {}'.format(sp, st) for sp, st in zip(ftj['species'], ftj['strain'])]
ftj['strain_name'].unique()

grouper = ['species', 'strain', 'strain_name', 'acquisition', 'fly_pair']
ftj, counts = add_legend_column_with_n(ftj, key='strain_name', grouper=grouper)

ftj = ftj.reset_index(drop=True)

#%%
# save strain info
yak_strains = ftj[ftj['species']=='dyak']['strain'].unique()
mel_strains = ftj[ftj['species']=='dmel']['strain'].unique()

strains_in_group = {'Dmel': mel_strains, 'Dyak': yak_strains}
 
# make dictionary where keys are strain names and values are species
strain_dict = dict( (k, 'Dyak') for k in yak_strains)
strain_dict.update(dict((k, 'Dmel') for k in mel_strains))

#%%
# assign courting behavior (1 - JAABA)
max_facing_angle = np.deg2rad(90) #45)
max_targ_pos_theta = np.deg2rad(270) #160)
min_targ_pos_theta = np.deg2rad(-270) #160)
min_wing_ang_deg = 45
min_wing_ang = np.deg2rad(min_wing_ang_deg)

orienting_angle = 10
ftj['orienting'] = ftj['facing_angle'] #False
ftj['orienting_binary'] = False
ftj.loc[(ftj['facing_angle']<=np.deg2rad(orienting_angle)), 'orienting_binary'] = True
ftj['orienting_binary_manual'] = ftj['orienting_binary']

# find where chasing_binary is nan
ftj['chasing_binary'] = ftj['chasing_binary'].fillna(False)
ftj['orienting_binary'] = ftj['orienting_binary'].astype(int)
ftj['singing_binary'] = ftj['singing_binary'].astype(int)
ftj['chasing_binary'] = ftj['chasing_binary'].astype(int)

print(ftj['chasing_binary'].unique(), ftj['singing_binary'].unique(), ftj['orienting_binary'].unique())

ftj['behav_sum'] = ftj[['singing_binary', 'chasing_binary', 'orienting_binary']].sum(axis=1)
ftj['courting'] = False
ftj.loc[ftj['behav_sum']>0, 'courting'] = True

#%%
#% split into bouts of courtship
# d_list = []
# for acq, df_ in ftj.groupby(['species', 'acquisition', 'fly_pair']):
#     df_ = df_.reset_index(drop=True)
#     df_ = util.mat_split_courtship_bouts(df_, bout_marker='courting')
#     dur_ = util.get_bout_durs(df_, bout_varname='boutnum', return_as_df=True,
#                     timevar='sec')
#     d_list.append(df_.merge(dur_, on=['boutnum']))
# ftj = pd.concat(d_list)

subdivide = False
if subdivide: 
    # subdivide into mini bouts
    subbout_dur = 0.20
    ftj = util.subdivide_into_subbouts(ftj, bout_dur=subbout_dur, 
                                    grouper=['species', 'acquisition', 'fly_pair'])
    #%
    if 'fpath' in ftj.columns:
        ftj = ftj.drop(columns=['fpath'])

    ftjm = ftj.groupby(['species', 'strain', 'strain_nam', 'sex', 
                    'acquisition', 'fly_pair', 'subboutnum']).mean().reset_index()

#%%
def filter_chasing(tdf, use_jaaba=True, beh_type='chasing_binary',
                   min_vel=10, max_facing_angle=np.deg2rad(90),
                   max_dist_to_other=20, 
                   max_targ_pos_theta=np.deg2rad(270),
                   min_targ_pos_theta=np.deg2rad(-270), 
                   min_wing_ang=np.deg2rad(45)): 
   
    if use_jaaba:
        if 'singing' in beh_type:
            chasedf = tdf[(tdf[beh_type]==True) & (tdf['sex']=='m')
                          & (tdf['facing_angle']<=max_facing_angle)
                          & (tdf['max_wing_ang']>=min_wing_ang)].copy()
        else:
            chasedf = tdf[(tdf[beh_type]==True) & (tdf['sex']=='m')
                        & (tdf['facing_angle']<=max_facing_angle)].copy()
    else:              
        chasedf = tdf[(tdf['sex']=='m') #(tdf['id']%2==0)
                #& (tdf['chasing'])
                & (tdf['vel'] >= min_vel)
                & (tdf['targ_pos_theta'] <= max_targ_pos_theta)
                & (tdf['targ_pos_theta'] >= min_targ_pos_theta)
                & (tdf['facing_angle'] <= max_facing_angle)
                & (tdf['max_wing_ang'] >= min_wing_ang)
                & (tdf['dist_to_other'] <= max_dist_to_other)
                ].copy()

    return chasedf

def annotate_p_value_two_groups(ax, yak_behav, mel_behav, fontsize=4):
    import scipy.stats as spstats
    res = spstats.mannwhitneyu( yak_behav, mel_behav, alternative='two-sided')    
    if res.pvalue < 0.01:
        ax.annotate('**', xy=(0.5, 1), fontsize=fontsize, #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', va='center')
    elif res.pvalue < 0.05:
        ax.annotate('*', xy=(0.5, 1), fontsize=fontsize, #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', va='center')
    return res

def plot_grouped_boxplots(mean_, palette='PRGn', 
                            between_group_spacing=1.5, 
                            within_group_spacing=0.5, box_width=0.3,
                            grouper='species', lw=0.5,
                            x='strain_name', y='vel', ax=None):
    '''
    Seaborn's box plot doesn't allow custom spacing (no gap functionality). 
    Custom function to plot boxplots with custom spacing.
    
    Args:
        mean_ (pd.DataFrame): Dataframe with mean values for each group
        palette (str): Seaborn color palette
        group_spacing (float): Spacing between groups
        x_spacing (float): Spacing between boxes within a group
        grouper (str): Column name to group by
        x (str): Column name for x-axis
        y (str): Column name for y-axis
        ax (matplotlib.axes.Axes): Axes to plot on, if None create new figure and axes 
    '''
    if ax is None:
        fig, ax = plt.subplots()
        
    species_order = mean_[grouper].unique()
    strain_order = mean_[x].unique()
    strain_palette = sns.color_palette(palette, n_colors=len(strain_order))
    strain_colors = dict(zip(strain_order, strain_palette))
    # Set spacing
    #group_spacing = 1.5
    #x_spacing = 0.2
    # Set positions
    positions = {}
    x_ticks = []
    x_tick_labels = []
    x_pos = 0
    for species in species_order:
        strains = mean_[mean_[grouper] == species][x].unique()
        n = len(strains)
        # Center strains around the group midpoint
        offsets = [(i - (n - 1) / 2) * within_group_spacing for i in range(n)] 
        for i, strain in enumerate(strains):
            positions[(species, strain)] = x_pos + offsets[i] #i * within_group_spacing
        x_ticks.append(x_pos) #x_pos + (len(strains) - 1) * within_group_spacing / 2)
        x_tick_labels.append(species)
        x_pos += between_group_spacing  # move to next species group
    # Plot
    for (species, strain), pos in positions.items():
        data = mean_[(mean_[grouper] == species) & (mean_[x] == strain)][y]
        ax.boxplot(data, positions=[pos], widths=box_width, patch_artist=True,
                boxprops=dict(facecolor=strain_colors[strain], edgecolor='black',
                              linewidth=lw),
                medianprops=dict(color='black'), 
                flierprops=dict(marker='o', markersize=0, color='black'))
    # Axis labels and legend
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels) 
    # Legend
    if ai==2:
        for strain in strain_order:
            ax.plot([], [], color=strain_colors[strain], label=strain, linewidth=5)
        ax.legend(title='Strain', bbox_to_anchor=(1.05, 1), loc='upper left')
    return ax

#%%
# Add manual filter on singing
# ------------------------------------------------
min_vel = 10
max_facing_angle = np.deg2rad(90) #45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(270) #160)
min_targ_pos_theta = np.deg2rad(-270) #160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

chasedf = filter_chasing(ftj, use_jaaba=False, #beh_type='chasing_binary',
                         min_vel=8, max_facing_angle=np.deg2rad(30),
                        max_dist_to_other=max_dist_to_other,
                        max_targ_pos_theta=max_targ_pos_theta,
                        min_targ_pos_theta=min_targ_pos_theta,
                        min_wing_ang=0) 

singdf = filter_chasing(ftj, use_jaaba=False, #beh_type='singing_binary',
                        min_vel=0, min_wing_ang=np.deg2rad(30),
                        max_dist_to_other=35,
                        max_facing_angle=np.deg2rad(90))

ftj['singing_binary_manual']= False
ftj.loc[singdf.index, 'singing_binary_manual'] = True
ftj['chasing_binary_manual']= False
ftj.loc[chasedf.index, 'chasing_binary_manual'] = True

#%
ftj['behav_sum'] = ftj[['singing_binary_manual', 'chasing_binary_manual', 
                        'orienting_binary_manual']].sum(axis=1)
ftj['courting_manual'] = False
ftj.loc[ftj['behav_sum']>0, 'courting_manual'] = True
#%
ftj['behav_sum'] = ftj[['singing_binary_manual', 'chasing_binary', 
                        'orienting_binary_manual']].sum(axis=1)
ftj['courting_manual_combo'] = False
ftj.loc[ftj['behav_sum']>0, 'courting_manual_combo'] = True

#%%
# compare velocity overall vs. velocity during chasing bouts
# -----------------------------------------------------------
use_jaaba=True
grouper = ['species', 'strain_name_legend', 'acquisition', 'fly_pair']
palette = 'PRGn'
plot_type = 'box'
plot_bar = plot_type == 'strip'

min_vel = 10
max_facing_angle = np.deg2rad(90) #45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(270) #160)
min_targ_pos_theta = np.deg2rad(-270) #160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

#%
# number_of_strains_in_group = len(strains_in_group)
# within_group_spacing = 0.5
# box_width = 0.3
# (number_of_strains_in_group - 1) * within_group_spacing + box_width
# species_spacing >= 
# 

chasedf_jaaba = filter_chasing(ftj, use_jaaba=True, beh_type='chasing_binary')               
singdf_jaaba = filter_chasing(ftj, use_jaaba=True, beh_type='singing_binary_manual',
                        min_vel=0, min_wing_ang=np.deg2rad(45),
                        max_facing_angle=np.deg2rad(60))

mean_vel = ftj[(ftj['sex']=='m')].groupby(grouper)['vel'].mean().reset_index()
mean_vel_chasing = chasedf_jaaba.groupby(grouper)['vel'].mean().reset_index()
mean_vel_singing = singdf_jaaba.groupby(grouper)['vel'].mean().reset_index()

fig, axn = plt.subplots(1,3, sharex=True, sharey=False, figsize=(10,4))
for ai, (beh_, mean_) in enumerate(zip(['all', 'chasing', 'singing'], 
                                    [mean_vel, mean_vel_chasing, mean_vel_singing])):
    ax=axn[ai]
    # Center barplot and stripplot over each other
    if plot_bar:
        sns.barplot(data=mean_, ax=ax, x='species', y='vel', color='k', linewidth=1,
                width=0.5, fill=False)
    if plot_type == 'strip':
        sns.stripplot(data=mean_, ax=ax, x='species', y='vel', hue='strain_name_legend',
                    palette=palette, dodge=True, jitter=False, linewidth=0.5, 
                    legend=ai==2)
    else:
        plot_grouped_boxplots(mean_, palette=palette, 
                              between_group_spacing=5, within_group_spacing=0.6, 
                              box_width=0.5,
                              grouper='species', x='strain_name_legend', y='vel', ax=ax)         
        #sns.boxplot(data=mean_, x='species', y='vel', hue='strain_name', ax=ax,
        #      palette=palette, legend=ai==2, fliersize=0, width=1, gap=0.1)
    ax.set_title(beh_)
    ax.set_xlabel('')
    ax.set_ylabel('Mean velocity (mm/s)')
    ax.set_ylim([0, 20])
    sns.despine(ax=ax, bottom=True)
sns.move_legend(axn[2], loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
plt.subplots_adjust(wspace=0.4)
#%
#plt.xticks(rotation=90)
putil.label_figure(fig, figid)
figname = 'mean_vel_by_behavior_per_strain_{}'.format(plot_type)
#plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# mean vel by species
grouper = ['species', 'strain_name_legend']
palette = 'PRGn'

mean_vel = ftj[(ftj['sex']=='m')].groupby(grouper)['vel'].mean().reset_index()
mean_vel_chasing = chasedf_jaaba.groupby(grouper)['vel'].mean().reset_index()
mean_vel_singing = singdf_jaaba.groupby(grouper)['vel'].mean().reset_index()

fig, axn = plt.subplots(1,3, sharex=True, sharey=False, figsize=(10,4))
for ai, (beh_, mean_) in enumerate(zip(['all', 'chasing', 'singing'], 
                                    [mean_vel, mean_vel_chasing, mean_vel_singing])):
    ax=axn[ai]
    # Center barplot and stripplot over each other
    sns.barplot(data=mean_, ax=ax, x='species', y='vel', color='k', linewidth=1,
                width=0.5, fill=False)
    #ax.margins(x=0.2)
    sns.stripplot(data=mean_, ax=ax, x='species', y='vel', hue='strain_name_legend',
                palette=palette, dodge=True, jitter=False, linewidth=0.5, 
                legend=ai==2, s=10)
    yak = mean_[mean_['species']=='Dyak']['vel']
    mel = mean_[mean_['species']=='Dmel']['vel']
    res = annotate_p_value_two_groups(ax, yak, mel)
    ax.set_title(beh_)
    ax.set_xlabel('')
    print(res)
    ax.set_ylim([0, 16])
    sns.despine(ax=ax, bottom=True)
sns.move_legend(axn[2], loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
#%
#plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_by_behavior_per_species'
plt.savefig(os.path.join(figdir, figname+'.png'))


#%%
# Plot p(singing) and p(chasing) for each strain
# -------------------------------------------------
#%%
import scipy.stats as spstats
#%% Do yak have higher p(singing)? # Look at strain distribution (all pairs)
pair_plot_type = 'box'
courting_frames = True
plot_species_mean = pair_plot_type=='strip'
palette = 'PRGn'
#%
#ftj_tmp = ftj[~ftj['acquisition'].isin(single_arena_acqs)].copy()
ftj_tmp = ftj.copy()
#%
jaaba_thresholds = {'chasing': 5, 'unilateral_extension': 10}
is_threshold = jaaba_thresholds['chasing']

use_jaaba = True
jaaba_str = 'False' if not use_jaaba else 'True (thr={})'.format(is_threshold)

data_type = 'courtframes' if courting_frames else 'allframes'
jaaba_suffix = '' if use_jaaba else '_manual'
plot_vars = ['courting', 'orienting_binary{}'.format(jaaba_suffix), 
             'singing_binary_manual', #.format(jaaba_suffix), 
             'chasing_binary{}'.format(jaaba_suffix)]
ftj[plot_vars] = ftj[plot_vars].astype('int')
plot_vars.append('dist_to_other')

# Calculate means
mean_frames = ftj_tmp[ftj_tmp['sex']=='m'].groupby([
                'species', 'strain_name', 'strain_name_legend', 'acquisition', 'fly_pair' #'behavior', 
                ])[plot_vars].mean().reset_index().dropna()
mean_frames_courting = ftj_tmp[(ftj_tmp['sex']=='m') & (ftj['courting']==True)].groupby([
                'strain_name', 'strain_name_legend', 'species', 'acquisition', 'fly_pair'
                ])[plot_vars].mean().reset_index().dropna()

plotd = mean_frames_courting.copy() if courting_frames else mean_frames.copy()

to_plot = ['courting', 
           'chasing_binary{}'.format(jaaba_suffix), 
           'singing_binary_manual'] #{}'.format(jaaba_suffix)]
#%%
fig, axn = plt.subplots(1, len(to_plot), sharex=True, sharey=False, figsize=(8, 4))
for ai, behav in enumerate(to_plot):
    ax=axn[ai]
    if plot_species_mean:
        sns.barplot(data=plotd, ax=ax, x='species', y=behav, color='k', linewidth=1,
            width=0.5, fill=False)
    if pair_plot_type == 'box':
        plot_grouped_boxplots(plotd, grouper='species', x='strain_name_legend', 
                              y=behav, ax=ax, palette=palette, 
                              between_group_spacing=5, within_group_spacing=0.6, 
                              box_width=0.5)                                     
        #sns.boxplot(data=plotd, x='species', y=behav, hue='strain_name_legend', ax=ax,
        #      palette=palette, legend=ai==2, fliersize=0, width=1, gap=0.1)
        if ai==2:
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    else:
        sns.stripplot(data=plotd, x='species', y=behav, hue='strain_name_legend', ax=ax, 
              palette=palette, legend=ai==2, jitter=False, dodge=True)
        if ai==2:
            sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

    # draw statistics on plot
    yak_behav = plotd[plotd['species']=='Dyak'][behav]
    mel_behav = plotd[plotd['species']=='Dmel'][behav]
    res = annotate_p_value_two_groups(ax, yak_behav, mel_behav, fontsize=4)
    print(res)
    ax.set_xlabel('')
    if 'chasing' in behav:
        ax.set_ylabel('p(chasing|courtship)') if courting_frames else ax.set_ylabel('p(chasing)')   
    elif 'singing' in behav:
        ax.set_ylabel('p(singing|courtship)') if courting_frames else ax.set_ylabel('p(singing)')
fig.text(0.5, 0.9, 'P(behavior), courting frames only: {}, JAABA: {}'\
                    .format(courting_frames, jaaba_str), fontsize=6, ha='center')
plt.subplots_adjust(wspace=0.5, top=0.8)
sns.despine(offset=2) 
putil.label_figure(fig, jaaba_dir)
# save
figname = 'p-behaviors-flypairs_{}_bar-{}_jaaba-{}'.format(data_type, pair_plot_type, jaaba_str)
plt.savefig(os.path.join(figdir, figname+'.png'))

#%% 
# Check if strain is different between single and quad MAI
mel_strain_check = ['Dmel SD105N_(Intermediate_Usually_Aroused)', 'Dmel CS_Mai', 
                    'Dyak RL_Ruta_Lab']
beh = 'singing_binary_manual'
for curr_strain in mel_strain_check:
    single_mai = []
    quad_mai = []
    acq_with_strain = plotd[plotd['strain_name']==curr_strain]['acquisition'].unique()
    for a, d_ in plotd[plotd['acquisition'].isin(acq_with_strain)].groupby('acquisition'):
        if (d_['fly_pair'].nunique() == 1):
            single_mai.append(a)
            plotd.loc[plotd['acquisition']==a, 'n_arenas'] = 1
        else:
            quad_mai.append(a)
            plotd.loc[plotd['acquisition']==a, 'n_arenas'] = 4

    print(len(single_mai), len(quad_mai))

fig, ax = plt.subplots()
sns.boxplot(data=plotd[plotd['strain_name'].isin(mel_strain_check)], 
            x='n_arenas', y=beh, ax=ax,
            hue='strain_name', palette=palette, legend=1)
sns.stripplot(data=plotd[plotd['strain_name'].isin(mel_strain_check)], 
            x='n_arenas', y=beh, ax=ax, linewidth=0.5, dodge=True,
            hue='strain_name', palette=palette, legend=1)

ax.set_box_aspect(1)
ax.set_ylim([0, 1])
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(0, 1), frameon=False)

figname = 'new-clfs_{}_Dmel-Dyak_single-vs-quad-arenas'.format(beh)
plt.savefig(os.path.join(figdir, figname+'.png'))


#%%
# Check ONLY single acq data
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}

single_arena_acqs = plotd[plotd['n_arenas']==1]['acquisition'].unique()
currd = ftj[(ftj['acquisition'].isin(single_arena_acqs)) & (ftj['sex']=='m')].copy()

sing_var = 'singing_binary_manual'
chase_var = 'chasing_binary'

ori_conds = [None, 10, 30, 90, 180]
fig, axn = plt.subplots(3, len(ori_conds), sharex=True, sharey=True,
                        figsize=(6.5, 4))
ri = 0
sing_var = 'singing_binary_manual'
chase_var = 'chasing_binary'
var_combos = [('singing_binary', 'chasing_binary'),
              ('singing_binary_manual', 'chasing_binary_manual'),
              ('singing_binary_manual', 'chasing_binary')]
              
for ri, (sing_var, chase_var) in enumerate(var_combos):
    for ci, ori_angle in enumerate(ori_conds):
        ax=axn[ri, ci]
        if ori_angle is None:
            courting_ = currd[ (currd[chase_var]==1) | (currd[sing_var]==1)].copy()
        else: 
            currd['orienting_binary'] = 0
            currd.loc[(currd['facing_angle']<=np.deg2rad(ori_angle)), 'orienting_binary'] = True
            courting_ = currd[ (currd['orienting_binary']==True) | 
                            (currd[chase_var]==1) | (currd[sing_var]==1)].copy()
            #courting_ = currd[(currd['courting']==1)].copy()

        meanbouts_courting_nodist = courting_.groupby(['species', 'acquisition'#'behavior', 
                                                    ])[sing_var].mean().reset_index()
        sns.boxplot(data=meanbouts_courting_nodist,
                    x='species', y=sing_var, ax=ax,
                    hue='species', palette=species_palette, legend=0)
        sns.stripplot(data=meanbouts_courting_nodist,
                    x='species', y=sing_var, ax=ax, 
                    hue='acquisition', color='k', s=2, legend=0, 
                    jitter=False, dodge=True) #palette=species_palette, legend=1)
        ax.set_ylim([0, 1])
        ax.set_box_aspect(1)
        ax.set_title("ori_angle: {}\n{}\n{}".format(ori_angle, chase_var, sing_var),
                     fontsize=5)
plt.subplots_adjust(hspace=0.6, wspace=0.5)
figname = 'new-clfs_avg-by-species_test-params_set2'
plt.savefig(os.path.join(figdir, figname+'.png'))
 
      #%%
      # 
      # )


#%% Calculate whether singing_binary is significantly different between species:
mean_strains = mean_frames_courting.groupby(['species', 'strain_name'])\
                    [plot_vars].mean().reset_index().dropna()   
                
fig, axn = plt.subplots(1, len(plot_vars), sharex=True, sharey=False, figsize=(9, 4))
for ai, behav in enumerate(plot_vars):
    ax=axn[ai]
    sns.barplot(data=mean_strains, ax=ax, x='species', y=behav, color='k', linewidth=1,
            width=0.5, fill=False)
    sns.stripplot(data=mean_strains, x='species', y=behav, hue='strain_name',
              palette=palette, dodge=True, jitter=False, ax=ax, legend=0)
    
    # draw statistics on plot
    yak_behav = mean_strains[mean_strains['species']=='Dyak'][behav]
    mel_behav = mean_strains[mean_strains['species']=='Dmel'][behav]
    res = spstats.mannwhitneyu( yak_behav, mel_behav, alternative='two-sided')    
    if res.pvalue < 0.01:
        ax.annotate('**', xy=(0.5, 1), #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', va='center')
    elif res.pvalue < 0.05:
        ax.annotate('*', xy=(0.5, 1), #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', va='center')
    ax.set_xlabel('')
    if 'chasing' in behav:
        ax.set_ylabel('p(chasing|courtship)') if courting_frames else ax.set_ylabel('p(chasing)')   
    elif 'singing' in behav:
        ax.set_ylabel('p(singing|courtship)') if courting_frames else ax.set_ylabel('p(singing)')
    ax.set_xlabel('')

fig.suptitle('P(behavior), courting frames only: {}'.format(courting_frames), fontsize=6)
plt.subplots_adjust(wspace=0.5, top=0.8)
sns.despine(offset=2, bottom=True, trim=True)

putil.label_figure(fig, jaaba_dir)
figname = 'p-behaviors-strainmeans_{}_bar-{}'.format(data_type, pair_plot_type)
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# Plot box plots of dist_to_other alone
behav = 'dist_to_other'
courting_frames = True
pair_plot_type = 'box'

data_type = 'courtframes' if courting_frames else 'allframes'
plot_species_mean = False
plotd = mean_frames_courting.copy() if courting_frames else mean_frames.copy()

# plot
fig, ax = plt.subplots(figsize=(3, 1))
if plot_species_mean:
    sns.barplot(data=plotd, ax=ax, x='species', y=behav,
            color='k', linewidth=1, width=0.5, fill=False)
if pair_plot_type == 'box':    
    plot_grouped_boxplots(plotd, grouper='species', x='strain_name', 
                            y=behav, ax=ax, palette=palette, 
                            between_group_spacing=10, within_group_spacing=1.2, 
                            box_width=1, lw=0.2) 
#     sns.boxplot(data=plotd, x='species', y=behav, hue='strain_name', ax=ax,
#             palette=palette, legend=0, fliersize=0,
#             width=1, gap=0.05, dodge=0.00001, linewidth=0.25)
else:
    sns.stripplot(data=plotd, x='species', y=behav, hue='strain_name', ax=ax, 
            palette=palette, legend=0, jitter=False) #dodge=True)
ax.set_box_aspect(1.5)
ax.set_ylim([0, 20])
#ax.set_xlim([-0.5, 5])
sns.despine(trim=True, offset=2, bottom=True)
ax.set_xlabel('')
ax.set_ylabel('Interfly distance (mm)')

yak_behav = plotd[plotd['species']=='Dyak'][behav]
mel_behav = plotd[plotd['species']=='Dmel'][behav]
res = spstats.mannwhitneyu( yak_behav, mel_behav, alternative='two-sided')
print(res)                     
if res.pvalue < 0.01:
    ax.annotate('**', xy=(0.5, 0.9), #ax.get_ylim()[-1]), 
                xycoords='axes fraction', ha='center', va='center')
elif res.pvalue < 0.05:
    ax.annotate('*', xy=(0.5, 0.9), #ax.get_ylim()[-1]), 
                xycoords='axes fraction', ha='center', va='center')

figname = 'dist_to_other_{}_bar-{}'.format(data_type, pair_plot_type)
plt.savefig(os.path.join(figdir, figname+'.png'))
plt.savefig(os.path.join(figdir, figname+'.svg'))

#%% 
# Plot across SPECIES
behav = 'dist_to_other'
courting_frames = True
data_type = 'courtframes' if courting_frames else 'allframes'

means = mean_frames_courting.copy() if courting_frames else mean_frames.copy()

plotd = means.groupby(['species', 'strain_name'])\
                [behav].mean().reset_index().dropna()   
# plot
fig, ax = plt.subplots(figsize=(1, 1))
sns.barplot(data=plotd, ax=ax, x='species', y=behav,
            color='k', linewidth=1, width=0.5, fill=False)
sns.stripplot(data=plotd, x='species', y=behav, hue='strain_name', ax=ax, 
            palette=palette, legend=0, jitter=0.3, s=3, linewidth=0.1, dodge=False)
ax.set_box_aspect(1.5)
ax.set_xlabel('')
ax.set_ylim([0, 15])
ax.set_xlim([-0.5, 1.5])
ax.set_ylabel('Interfly distance (mm)')

yak_behav = plotd[plotd['species']=='Dyak'][behav]
mel_behav = plotd[plotd['species']=='Dmel'][behav]
res = spstats.mannwhitneyu( yak_behav, mel_behav, alternative='two-sided')
                     
res = spstats.mannwhitneyu( yak_behav, mel_behav, alternative='two-sided')    
if res.pvalue < 0.01:
    ax.annotate('**', xy=(0.5, 0.95), #ax.get_ylim()[-1]), 
                xycoords='axes fraction', ha='center', va='center')
elif res.pvalue < 0.05:
    ax.annotate('*', xy=(0.5, 0.95), #ax.get_ylim()[-1]), 
                xycoords='axes fraction', ha='center', va='center')
sns.despine(trim=True, offset=2, bottom=True)

figname = 'dist_to_other_{}_species-means'.format(data_type)
plt.savefig(os.path.join(figdir, figname+'.png'))
plt.savefig(os.path.join(figdir, figname+'.svg'))


#%% # Binn dist_to_other
bin_size = 5 #3 
max_dist = 30 #25#np.ceil(ftjaaba['dist_to_other'].max())
dist_bins = np.arange(0, max_dist+bin_size, bin_size)
print(dist_bins)

# Cut dist_to_other into bins and assign label to new columns:
ftj['binned_dist_to_other'] = pd.cut(ftj['dist_to_other'], 
                                    bins=dist_bins, 
                                    labels=dist_bins[:-1])   
ftj['binned_dist_to_other'] = ftj['binned_dist_to_other'].astype(float)

#%%
def add_legend_column_with_N(ftj, key='strain', 
                    grouper=['species', 'strain', 'acquisition', 'fly_pair']):
    conds = ftj[grouper].drop_duplicates()
    counts = conds.groupby([key])['fly_pair'].count()
    ftj['{}_legend'.format(key)] = ftj[[key]].applymap(lambda x: '{} (n={})'.format(x, counts[x]))
  
    return ftj, counts

grouper = ['species', 'strain', 'strain_name', 'acquisition', 'fly_pair']
ftj, counts = add_legend_column_with_N(ftj, key='strain_name', grouper=grouper)
#
#%%
# Get means of chasing and singing by binned_dist_to_other
plot_vars = ['orienting_binary', 'singing_binary_manual', 'chasing_binary']
courting = ftj[ftj['courting']==True].copy()

meanbouts_courting = courting[courting['sex']=='m'].groupby([
                    'species', 'strain', 'strain_name_legend', 'acquisition', 'fly_pair', #'behavior', 
                    'binned_dist_to_other'])[plot_vars].mean().reset_index()

meanbouts_orienting = ftj[ftj['sex']=='m'].groupby([
                    'species', 'strain', 'strain_name_legend', 'acquisition', 'fly_pair', #'behavior', 
                    'binned_dist_to_other'])[plot_vars].mean().reset_index()

#%% # Bin dist_to_other during chasing and singing
#species_palette = {'Dmel': 'lavender', 
plot_bar = True
grouper_palette = 'cubehelix'
error_type = 'ci'
plot_pairs = False

plot_type = 'bar' if plot_bar else 'point'

for curr_species, df_ in meanbouts_courting.groupby('species'): #_courting.groupby('species'):
    fig, axn = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=False)
    for ai, behav in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary_manual']):
        if behav == 'orienting_binary':
            df_ = meanbouts_orienting[meanbouts_orienting['species']==curr_species]
        ax=axn[ai]
        if plot_pairs:
            sns.stripplot(data=df_,
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, 
                        hue='{}_legend'.format('strain_name'), palette=grouper_palette, legend=False,
                        edgecolor='w', linewidth=0.25, dodge=True, jitter=True)
        if plot_bar:
            sns.barplot(data=df_,
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, lw=0.5,
                        errorbar=error_type, errcolor=bg_color, errwidth=0.75,
                        hue='{}_legend'.format('strain_name'), palette=grouper_palette, 
                        fill=plot_pairs==False, legend=1)
        else:
            sns.pointplot(data=df_, 
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, scale=0.5,
                        errorbar=error_type, errwidth=0.75,
                        hue='{}_legend'.format('strain_name'), palette=grouper_palette, legend=1)
                        #fill=plot_pairs==False, legend=1)
        ax.set_ylim([0, 0.8])
        if ai<=0:
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
    axn[1].set_ylabel('p(chasing|courtship)')
    axn[-1].set_ylabel("p(singing|courtship)")

    putil.label_figure(fig, figid)
    figname = 'p_singing_chasing_v_binned_dist_to_other_bin-{}_{}_{}'.format(bin_size, plot_type, curr_species)
    plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# just look at RL strains
#% plot
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
error_type = 'ci'

yak_strain = 'RL_Ruta_Lab'
mel_strains = ['CS_Mai', 'SD105N_(Intermediate_Usually_Aroused)']

yak = ftj[(ftj['species']=='Dyak') & (ftj['strain']==yak_strain) * (ftj['sex']=='m')]

for mel_strain in mel_strains:
    mel = ftj[(ftj['species']=='Dmel') & (ftj['strain']==mel_strain)]
    tmpdf = pd.concat([yak, mel])
    courting = tmpdf[tmpdf['courting']==1].copy()
    #%
    # average over subbout
    bout_type = 'frames' #'subboutnum'
    meanbouts_courting = courting.groupby(['species', 'acquisition', 'fly_pair', 
                            'binned_dist_to_other'])[['chasing_binary', 'singing_binary_manual']].mean().reset_index()
    # plot
    fig, axn = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(5, 3))
    for ai, behav in enumerate(['chasing_binary', 'singing_binary_manual']):
        ax=axn[ai]
        plotd = meanbouts_courting
        sns.barplot(data=plotd,
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, 
                        errorbar=error_type, errcolor=bg_color,
                        hue='species', palette=species_palette, 
                        edgecolor='none')
        if ai!=2:
            ax.legend_.remove()
    
    axn[0].set_ylabel("p(chasing|courtship)")
    axn[1].set_ylabel("p(singing|courtship)")
    fig.text(0.1, 0.9, 'Compare RL yak to Dmel {}'.format(mel_strain), fontsize=6)
    
    for ax in axn:
        ax.set_box_aspect(1)
        ax.set_xlabel('distance to other (mm)')
    # format xticks to single digit numbers:
    bin_edges = [str(int(x)) for x in dist_bins[:-1]]
    bin_edges[0] = '<{}'.format(bin_size)
    bin_edges[-1] = '>{}'.format(int(dist_bins[-2]))
    axn[0].set_xticks(range(len(bin_edges)))
    axn[0].set_xticklabels([str(x) for x in bin_edges], rotation=0)

    for ax in axn:
        ax.set_ylim([0, 0.7]) 
    sns.despine(offset=4)
    putil.label_figure(fig, local_dir) 
    # plot
    figname = 'pSinging_binned_dist_to_other_{}_{}'.format(yak_strain, mel_strain)
    plt.savefig(os.path.join(figdir, figname+'.png'))   

#%%
# Cumulative dist_to_other histograms, split by behavior:
c1 = 'purple'
c2 = 'green'
hist_bins = 100
cumulative=True
fill = cumulative is False

yak_strains = strains_in_group['Dyak']
mel_strains = strains_in_group['Dmel']
species_palette = dict((k, c2) for k in yak_strains) 
species_palette.update(dict((k, c1) for k in mel_strains))

fig, axn = plt.subplots(1, 3, figsize=(6,3), sharex=True, sharey=True)
for curr_species, df_ in ftj.groupby('species'):
    for ai, beh in enumerate(['orienting_binary', 'chasing_binary', 'singing_clean_binary']):
        plotd = df_[df_[beh]==True].copy().reset_index(drop=True)
        ax=axn[ai]
        #hist_palette = species_palette[curr_species]
        if beh == 'chasing_binary':
            title = 'dist to other (chasing)'
        elif beh == 'singing_binary':
            title = 'dist to other (singing)'
        else:
            title = 'dist to other (orienting)'
        ax.set_title(title)
        sns.histplot(data=plotd, ax=ax,
                    x='dist_to_other', bins=hist_bins, stat='probability',
                    hue='strain_name', common_norm=False,
                    element='step', lw=0.5, alpha=1, fill=fill, cumulative=cumulative,
                    #color=species_palette[curr_species])
                    palette=species_palette)
        if ai <= 1:
            ax.legend_.remove()
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
# Cumulative dist_to_other histograms during courting frames
species_palette = dict((k, c2) for k in yak_strains) 
species_palette.update(dict((k, c1) for k in mel_strains))

fig, ax = plt.subplots()
for curr_species, df_ in ftj[ftj['courting']==True].groupby('species'):

    sns.histplot(data=df_, ax=ax,                    
             x='dist_to_other', bins=hist_bins, stat='probability',
            hue='strain_name', common_norm=False,
            element='step', lw=0.5, alpha=1, fill=fill, cumulative=cumulative,
            palette=species_palette) 

legh = putil.custom_legend(labels=['Dmel', 'Dyak'], colors=[c1, c2])
ax.legend(handles=legh, loc='upper left', bbox_to_anchor=(1,1), frameon=False, 
            title='', fontsize=6)
ax.set_box_aspect(1)

figname = 'courting_v_dist_to_other_hist'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# ----------------------------------------------------------
# TRANSFORM DATA
# ----------------------------------------------------------
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
min_wing_ang_deg = 40
min_wing_ang = np.deg2rad(min_wing_ang_deg)
    
#%%
# ----------------------------------------------------------
# DEFINE COURTING/SINGING - SPATIAL OCCUPANCY
# ----------------------------------------------------------
use_jaaba= True #True
#court_behavior = 'singing_clean_binary'
court_behavior = 'chasing_binary'
max_facing_angle = np.deg2rad(90)
min_wing_ang = np.deg2rad(30)

if use_jaaba:
    court_ = ftj[(ftj['sex']=='m') & (ftj[court_behavior]==True)
#                    & (ftj['facing_angle']<=max_facing_angle)
                    & (ftj['max_wing_ang']>=min_wing_ang)].copy()
else:
    court_ = ftj[(ftj['sex']=='m') #& (ftjaaba['chasing']==1)
                & (ftj['vel']> min_vel)
                & (ftj['targ_pos_theta'] <= max_targ_pos_theta)
                & (ftj['targ_pos_theta'] >= min_targ_pos_theta)
                & (ftj['facing_angle'] <= max_facing_angle)
                & (ftj['max_wing_ang'] >= min_wing_ang)
                & (ftj['dist_to_other'] <= max_dist_to_other)].copy()
#%%
court_behavior = 'chasing_binary_wing{}'.format( int(round(np.rad2deg(min_wing_ang))))
print(court_behavior)
 
#%%        
# Get female-centered frames: 
# Relative to the female (centered at 0,0), where is the male
f_list = []
for (sp, strain, acq, fp), curr_court in court_.groupby(['species', 'strain', 'acquisition', 'fly_pair']):
   
    # NOTE: BEFORE, this was using df, instead of df_!! 
    f2_ = ftj[ (ftj['species']==sp) & (ftj['strain']==strain)
             & (ftj['fly_pair']==fp)
             & (ftj['frame'].isin(curr_court['frame']))
             & (ftj['sex']=='f')
             & (ftj['acquisition']==acq)].copy() #wing_ext[wing_ext['id']==1].copy()
    f_list.append(f2_)
f2 = pd.concat(f_list).reset_index(drop=True)

#%%

ftj[ (ftj['species']==sp) & (ftj['strain']==strain)
             & (ftj['fly_pair']==fp)
             & (ftj['frame'].isin(curr_court['frame']))
             & (ftj['sex']=='f')
             & (ftj['acquisition']==acq)].shape

#%%
f2_conds = f2[['species', 'strain', 'acquisition', 'fly_pair']].drop_duplicates()
f2_counts = f2_conds.groupby(['species', 'strain'])['fly_pair'].count()
print(f2_counts)

#%%
# Plot position of MALE relative to female (centered)
import matplotlib as mpl

spatial_cmap = 'GnBu' if plot_style=='white' else 'magma'
bins=100
stat='probability'
vmax=0.001 if stat=='probability' else 50
jaaba_thresholds = {'chasing': 5, 'unilateral_extension': 10}
is_threshold = jaaba_thresholds['unilateral_extension']

jaaba_str = 'False' if not use_jaaba else 'True (thr={})'.format(is_threshold)
#f2 = f2.reset_index(drop=True)
#print(grouper)
print(conds.groupby(['species', 'strain'])['fly_pair'].count())

for curr_species, f2_ in f2.groupby('species'):
    n_strains = f2_['strain'].nunique()
    # Get number of unique fly pairs by strain
    max_n_pair = f2_[['strain', 'acquisition', 'fly_pair']].drop_duplicates().groupby('strain').count().max().iloc[0]
    n_strains = f2_['strain'].nunique()

    nc = max_n_pair #int(np.ceil(max_n_pair/n_strains))
    nr = n_strains    
    fig, axn = plt.subplots(nr, nc, figsize=(10, 8), sharex=True, sharey=True)
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
                ax.text(ax.get_xlim()[0]-100, ax.get_ylim()[-1]+400, strain, 
                    ha='left', va='center', fontsize=4, rotation=0)
            datestr = flypair[0].split('_')[0]
            fpair = int(flypair[1])
            maleid = d_['id'].unique()[0]
            ax.text(ax.get_xlim()[0]-100, ax.get_ylim()[-1]+100, 
                    '{}\nflypair={}, id={}'.format(datestr, fpair, maleid),
                    ha='left', va='center', fontsize=4, rotation=0)
            ax.set_xlim([-500, 500])
            ax.set_ylim([-500, 500])
        
    plt.subplots_adjust(top=0.9, left=0.1)
    #fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
    #fig.text(0.01, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
    fig.suptitle('{}, {}, jaaba={}'.format(court_behavior, curr_species, jaaba_str), 
                 x=0.3, y=0.95, fontsize=10) 
    for ax in axn.flat[ai+1:]:
        ax.axis('off') 
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, wspace=0.2)

   
    putil.label_figure(fig, figid) 
    figname = 'male-rel-pos_{}_per-flypair_{}-jaaba-{}'.format(curr_species, court_behavior, jaaba_str) 
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
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.axis('off')
    ax.plot(0, 0, bg_color, markersize=5, marker='>')

    return ax

#%%
# Plot occupancy for each species, strain, flypair -- individual plots

for (curr_species, curr_strain), f2_ in f2.groupby(['species', 'strain']):
    fig, ax = plt.subplots()
    ax = plot_occupancy(f2_, ax=ax, cmap=cmap,
                        vmin=0, vmax=vmax, bins=bins,
                        stat=stat, bg_color='w')
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_title('{}: {}'.format(curr_species, curr_strain), fontsize=8)
    figname = 'occ_{}_{}_{}-jaaba-{}'.format(curr_species, curr_strain, court_behavior, jaaba_str)
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    
#%% ALL on the same plot
vmax=0.001
nr=2
nc=4
#curr_species = 'Dyak'
#curr_plotd = f2[f2['species']==curr_species]
for curr_species, curr_plotd in f2.groupby('species'):
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                            figsize=(nc*3,nr*3))
    for ai, (strain, f2_) in enumerate(curr_plotd.groupby('strain')):
        ax=axn.flat[ai]

        ax = plot_occupancy(f2_, ax=ax, cmap=cmap,
                            vmin=0, vmax=vmax, bins=bins,
                            stat=stat, bg_color=bg_color)
        ax.set_xlim([-300, 300])
        ax.set_ylim([-300, 300])
        ax.set_title(strain, fontsize=4)
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
    figname = 'male-rel-pos_all-pairs_{}_{}-jaaba-{}'.format(curr_species, court_behavior, jaaba_str) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
    print(figdir)
 
# %%
# Get male-centered frames: 
# Relative to the male (centered at 0,0), where is the female,
# i.e., where does the male keep the female?

# f_list = []
# for acq, curr_court in court_.groupby('acquisition'):
#    
#     # NOTE: BEFORE, this was using df, instead of df_!! 
#     f1_ = ftj[ (ftj['frame'].isin(curr_court['frame']))
#              & (ftj['sex']=='m')
#              & (ftj['acquisition']==acq)].copy() #wing_ext[wing_ext['id']==1].copy()
#     f_list.append(f1_)
# f1 = pd.concat(f_list).reset_index(drop=True)

f1 = court_.copy()
 
   #%%
#if grouper=='strain':
nr=2; nc=4;

for sp, f1_strains in f1.groupby('species'):
    fig, axn = plt.subplots(nr, nc, sharex=True, sharey=True,
                            figsize=(nc*3,nr*3))
    #fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
    for ai, (cond, f1_) in enumerate(f1_strains.groupby('strain')):
        ax=axn.flat[ai]
        sns.histplot(data=f1_, 
                x='targ_rel_pos_x', y='targ_rel_pos_y', ax=ax,
                cmap=cmap,stat='probability', bins=bins, vmax=vmax, vmin=0)
        ax.set_title(cond)
        ax.set_aspect(1)
        ax.plot(0, 0, 'w', markersize=5, marker='>') 

    for ax in axn.flat[ai+1:]:
        ax.axis('off')
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    
    fig.text(0.1, 0.95, 'Female position from male-centered view', fontsize=8)
    putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                                hue_title=stat)

    putil.label_figure(fig, figid)     
    figname = 'female-rel-pos_{}_all-pairs_{}-jaaba-{}'.format(sp, court_behavior, jaaba_str) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir)

#%%
