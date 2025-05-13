
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

#%
# Create subset of cubehelix palette to exclude white but keep pink
strain_palette = sns.cubehelix_palette(rot=-1.5, light=0.8, reverse=True)
sns.palplot(strain_palette)

# %%
# Get mean velocity by condition
grouper= ['species', 'strain', 'strain_num', 'global_id']
 
mean_vel = df0[df0['sex']=='m'].groupby(grouper)['vel'].mean().reset_index()
mean_vel.head()
#mean_vel

# %% Plot mean velocity by condition
fig, ax =plt.subplots(figsize=(5,3))
#sns.pointplot(data=df[df['sex']=='m'], ax=ax, x='winged', 
#              hue='acquisition', y='vel')
# Center barplot and stripplot over each other
sns.barplot(data=mean_vel, ax=ax, x='species', y='vel', color='k', linewidth=1,
            width=0.5, fill=False)
#ax.margins(x=0.2)
sns.stripplot(data=mean_vel, ax=ax, x='species', y='vel', hue='strain',
            palette=strain_palette, dodge=True, jitter=True) #, legend=False)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
ax.set_box_aspect(1)
#sns.swarmplot(data=mean_vel, ax=ax, x=grouper, y='vel', 
#              hue='global_id', palette='tab20', legend=False)
#ax.set_box_aspect(1)
# rotate x-tick labels
#plt.xticks(rotation=90)
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

#%%
# ==========================================
# Load JAABA scores
# ==========================================
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
cond_name = 'strain'
if has_jaaba:
    for acq, df_ in df0.groupby('acquisition'):# in acqs[0:nr*nc]:
        if acq in no_jaaba_acqs:
            continue
        mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 'scores_{}*.mat'.format(beh_type)))[0]
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
has_jaaba=True
#is_threshold = 5 #0.4
jaaba_thresholds = {'chasing': 5, 'unilateral_extension': 10}
isnot_threshold = 0.2 #14
beh_type = 'chasing'

#df0[beh_type] = False
binarize_jaaba = False
df_list = []
no_scores = dict((k, []) for k in ['chasing', 'unilateral_extension'])
for acq, df in df0.groupby('acquisition'):

    for beh_type in ['chasing', 'unilateral_extension']:
        is_threshold = jaaba_thresholds[beh_type]
        if has_jaaba:
            # Load JAABA scores
            try:
                mat_fpath = glob.glob(os.path.join(jaaba_dir, acq, 
                            'scores_{}*.mat'.format(beh_type)))[0]
            except IndexError:
                no_scores[beh_type].append(acq)
                print("No JAABA {} scores for {}".format(beh_type, acq))
                continue 
            try:
                jaaba_scores = load_jaaba_from_mat(mat_fpath)
            except Exception as e:
                traceback.print_exc()
                no_scores[beh_type].append(acq)
                print("Error loading JAABA scores for {}: {}".format(acq, e))
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
            df = df.merge(jaaba_scores_stacked.rename(columns={'score': beh_type}), 
                        how='inner', on=['frame', 'id']) 
            if binarize_jaaba: 
                df[beh_type] = df[beh_type].astype('bool')
            else:
                df['{}_binary'.format(beh_type)] = df[beh_type].ge(is_threshold)

    # Reassign to df0
    #df0.loc[df0['acquisition']==acq] = df
    df_list.append(df) #[~df['copulating']])

ftj = pd.concat(df_list)
# 
print("[{}]: No JAABA for the following acquisitions:".format(beh_type))
pp.pprint(no_scores)
#%%

ftj = ftj.rename(columns={'unilateral_extension_binary': 'singing_binary',
                          'unilateral_extension': 'singing'}) 
#%%
max_facing_angle = np.deg2rad(90) #45)
max_targ_pos_theta = np.deg2rad(270) #160)
min_targ_pos_theta = np.deg2rad(-270) #160)
min_wing_ang_deg = 45
min_wing_ang = np.deg2rad(min_wing_ang_deg)

#orienting = ftj[(ftj['sex']=='m')
#                & (ftj['facing_angle']<=max_facing_angle)].copy()
orienting_angle = 30
ftj['orienting'] = ftj['facing_angle'] #False
ftj['orienting_binary'] = False
ftj.loc[(ftj['sex']=='m')
    & (ftj['facing_angle']<=np.deg2rad(orienting_angle)), 'orienting_binary'] = True
#ftj['orienting'] = False
#ftj.loc[orienting.index, 'orienting'] = True

#% P(SINGING)
#singdf = 
# ftj['singing'] = False
# ftj.loc[(ftj['sex']=='m') ##  (ftj['vel']>min_vel)
            #  (ftj['facing_angle']<=45)
            #  (ftj['targ_pos_theta'] <= 160) #max_targ_pos_theta)
            #  (ftj['targ_pos_theta'] >= -160) #min_targ_pos_theta)
            #  (ftj['max_wing_ang']>=min_wing_ang), 'singing'] = True
#ftj['singing'] = False
#ftj.loc[singdf.index, 'singing'] = True

ftj['behav_sum'] = ftj[['singing_binary','chasing_binary', 'orienting_binary']].sum(axis=1)

ftj['courting'] = False
ftj.loc[ftj['behav_sum']>0, 'courting'] = True

# Find where chasing_binary is nan
ftj['chasing_binary'] = ftj['chasing_binary'].fillna(False)
print(ftj['chasing_binary'].unique(), ftj['singing_binary'].unique(), ftj['orienting_binary'].unique())


#%%
#% Split into bouts of courtship
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
    # Subdivide into mini bouts
    subbout_dur = 0.20
    ftj = util.subdivide_into_subbouts(ftj, bout_dur=subbout_dur, 
                                    grouper=['species', 'acquisition', 'fly_pair'])
    #%
    if 'fpath' in ftj.columns:
        ftj = ftj.drop(columns=['fpath'])

    ftjm = ftj.groupby(['species', 'strain', 'strain_num', 'sex', 
                    'acquisition', 'fly_pair', 'subboutnum']).mean().reset_index()

#%%
ftj['chasing'].unique()

#%%
conds = ftj[['acquisition', 'species', 'strain', 'strain_num', 'fly_pair']].drop_duplicates()
counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
print(counts)
ftj['species'] = ftj['species'].astype('category')

#%%
#yak = ftj[ftj['species']=='Dyak']   
#mel = ftj[ftj['species']=='Dmel']

#%%
def filter_chasing(tdf, use_jaaba=True, beh_type='chasing_binary',
                   min_vel=10, max_facing_angle=np.deg2rad(20),
                   max_dist_to_other=20, max_targ_pos_theta=np.deg2rad(160),
                   min_targ_pos_theta=np.deg2rad(-160), 
                   min_wing_ang=np.deg2rad(45)): 
   
    if use_jaaba:
        chasedf = tdf[(tdf[beh_type]==True) & (tdf['sex']=='m')
                    & (tdf['facing_angle']<=max_facing_angle)].copy()
    else:
#        chasedf = ftj[(ftj['sex']=='m') & (ftj['vel']>=min_vel)
#                & (ftj['facing_angle']<=max_facing_angle)
#                & (ftj['angle_between']<=max_angle_between)].copy()
#                #& (ftj['max_wing_ang']>=min_wing_angle)].copy()
              
        chasedf = tdf[(tdf['sex']=='m') #(tdf['id']%2==0)
                #& (tdf['chasing'])
                & (tdf['vel'] > min_vel)
                & (tdf['targ_pos_theta'] <= max_targ_pos_theta)
                & (tdf['targ_pos_theta'] >= min_targ_pos_theta)
                & (tdf['facing_angle'] <= max_facing_angle)
                & (tdf['max_wing_ang'] >= min_wing_ang)
                & (tdf['dist_to_other'] <= max_dist_to_other)
                ].copy()
    
    #court_ = tdf[(tdf['sex']=='m') & (tdf['vel']>=min_vel)
    #         & (tdf['facing_angle']<=max_facing_angle)
    #         & (tdf['angle_between']<=max_angle_between)].copy()
    return chasedf

#%%
# Compare velocity overall vs. velocity during CHASING bouts
# -----------------------------------------------------------
#min_wing_angle = np.deg2rad(20)
#min_vel = 10
#max_facing_angle = np.deg2rad(45)
#max_angle_between = 1.6 #np.deg2rad(90)

use_jaaba=True
grouper = ['species', 'strain', 'strain_num', 'global_id']

min_vel = 10
max_facing_angle = np.deg2rad(90) #45)
max_dist_to_other = 20
max_targ_pos_theta = np.deg2rad(270) #160)
min_targ_pos_theta = np.deg2rad(-270) #160)
min_wing_ang_deg = 30
min_wing_ang = np.deg2rad(min_wing_ang_deg)

chasedf = filter_chasing(ftj, use_jaaba=True, beh_type='chasing_binary',
                         min_vel=min_vel, max_facing_angle=max_facing_angle,
                        max_dist_to_other=max_dist_to_other,
                        max_targ_pos_theta=max_targ_pos_theta,
                        min_targ_pos_theta=min_targ_pos_theta,
                        min_wing_ang=min_wing_ang)               
singdf = filter_chasing(ftj, use_jaaba=True, beh_type='singing_binary',
                        min_vel=0)

mean_vel = ftj[(ftj['sex']=='m')].groupby(grouper)['vel'].mean().reset_index()
mean_vel_chasing = chasedf.groupby(grouper)['vel'].mean().reset_index()
mean_vel_singing = singdf.groupby(grouper)['vel'].mean().reset_index()

fig, axn = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,4))
for ai, (beh_, mean_) in enumerate(zip(['all', 'chasing', 'singing'], 
                                       [mean_vel, mean_vel_chasing, mean_vel_singing])):
    ax=axn[ai]
    # Center barplot and stripplot over each other
    sns.barplot(data=mean_, ax=ax, x='species', y='vel', color='k', linewidth=1,
                width=0.5, fill=False)
    #ax.margins(x=0.2)
    sns.stripplot(data=mean_, ax=ax, x='species', y='vel', hue='strain',
                palette='cubehelix', dodge=True, jitter=False, linewidth=0.5, 
                legend=ai==2)
    ax.set_title(beh_)
    ax.set_xlabel('')
sns.move_legend(axn[2], loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
#%
#plt.xticks(rotation=90)
putil.label_figure(fig, figid)

figname = 'mean_vel_by_behavior_per_strain'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
# IDentify courtship frames
ftj = ftj.reset_index(drop=True)

#%%
import scipy.stats as spstats

#%% Do yak have higher p(singing)?

plot_vars = ['orienting_binary', 'singing_binary', 'chasing_binary', 'courting']
ftj[plot_vars] = ftj[plot_vars].astype('int')
plot_vars.append('dist_to_other')

mean_frames = ftj.groupby([
                'species', 'strain', 'strain_num', 'acquisition', 'fly_pair', #'behavior', 
                ])[plot_vars].mean().reset_index().dropna()
mean_frames_courting = ftj[ftj['courting']==True].groupby([
                'strain', 'species', 'acquisition', 'fly_pair'#'behavior',
                ])[plot_vars].mean().reset_index().dropna()

plotd = mean_frames_courting.copy()

fig, axn = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(8, 4))
for ai, behav in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary']):
    ax=axn[ai]
    sns.barplot(data=plotd, ax=ax, x='species', y=behav, color='k', linewidth=1,
            width=0.5, fill=False)
    sns.stripplot(data=plotd, x='species', y=behav, hue='strain',
              palette='cubehelix', dodge=True, jitter=False, ax=ax, legend=0)

    # draw statistics on plot
    yak_behav = plotd[plotd['species']=='Dyak'][behav]
    mel_behav = plotd[plotd['species']=='Dmel'][behav]
    res = spstats.mannwhitneyu( yak_behav, mel_behav, 
                        alternative='two-sided')    
    if res.pvalue < 0.01:
        ax.annotate('**', xy=(0.5, 1.1), #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', 
                    va='center')
    elif res.pvalue < 0.05:
        ax.annotate('*', xy=(0.5, 1.1), #ax.get_ylim()[-1]), 
                    xycoords='axes fraction', ha='center', 
                    va='center')
    ax.set_xlabel('')
    print(res)
sns.despine(offset=2) 
#spstats.ttest_ind(yak_behav, mel_behav) 
figname = 'p-behaviors_bar-strip'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
#% Calculate whether singing_binary is significantly different between species:
mean_strains = mean_frames_courting.groupby(['species', 'strain'])\
                    [plot_vars].median().reset_index().dropna()   
fig, axn = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(8, 4))
for ai, behav in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary']):
    ax=axn[ai]
    sns.barplot(data=mean_strains, ax=ax, x='species', y=behav, color='k', linewidth=1,
            width=0.5, fill=False)
    sns.stripplot(data=mean_strains, x='species', y=behav, hue='strain',
              palette='cubehelix', dodge=True, jitter=False, ax=ax, legend=0)

#%
beh = 'singing_binary'
yak_behav = mean_strains[mean_strains['species']=='Dyak'][beh]
mel_behav = mean_strains[mean_strains['species']=='Dmel'][beh]
res = spstats.mannwhitneyu( yak_behav, mel_behav, 
                     alternative='two-sided')
print(res)

spstats.ttest_ind(yak_behav, mel_behav)

#%%
beh = 'dist_to_other'

plotd = mean_frames_courting.copy()
fig, ax = plt.subplots()
sns.barplot(data=plotd, ax=ax, x='species', y=beh,
            color='k', linewidth=1,
        width=0.5, fill=False)
sns.stripplot(data=plotd, x='species', y=beh, 
              hue='strain',
            palette='cubehelix', dodge=True, jitter=False, ax=ax, legend=0)
ax.set_box_aspect(1)
ax.set_xlabel('')

yak_behav = plotd[plotd['species']=='Dyak'][beh]
mel_behav = plotd[plotd['species']=='Dmel'][beh]
res = spstats.mannwhitneyu( yak_behav, mel_behav, 
                     alternative='two-sided')
print(res)
spstats.ttest_ind(yak_behav, mel_behav)

figname = 'dist_to_other_bar-strip'
plt.savefig(os.path.join(figdir, figname+'.png'))

#%%
yak_behav = mean_strains[mean_strains['species']=='Dyak'][beh]
mel_behav = mean_strains[mean_strains['species']=='Dmel'][beh]
res = spstats.mannwhitneyu( yak_behav, mel_behav, 
                     alternative='two-sided')
print(res)
spstats.ttest_ind(yak_behav, mel_behav)



#%%



 
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
#
#%
# Get means of chasing and singing by binned_dist_to_other
plot_vars = ['orienting_binary', 'singing_binary', 'chasing_binary']
courting = ftj[ftj['courting']==True].copy()

meanbouts_courting = courting.groupby([
                    'species', 'strain', 'strain_num', 'acquisition', 'fly_pair', #'behavior', 
                    'binned_dist_to_other'])[plot_vars].mean().reset_index()

meanbouts_orienting = ftj.groupby([
                    'species', 'strain', 'strain_num', 'acquisition', 'fly_pair', #'behavior', 
                    'binned_dist_to_other'])[plot_vars].mean().reset_index()
#%
grouper = ['species', 'strain', 'strain_num', 'acquisition', 'fly_pair']
# Addd counts of each condition type
#counts = meanbouts_courting[['species', 'strain', 'acquisition', 'fly_pair']]\
#                    .drop_duplicates().groupby(['strain'])['fly_pair'].count()
strains = meanbouts_courting['strain'].unique() #meanbouts_courting[grouper].unique()

# Add counts of each species strain to dataframe as legend
meanbouts_courting['{}_legend'.format('strain')] = meanbouts_courting[['strain']].applymap(lambda x: '{} (n={})'.format(x, counts[x]))
meanbouts_orienting['{}_legend'.format('strain')] = meanbouts_orienting[['strain']].applymap(lambda x: '{} (n={})'.format(x, counts[x]))

#counts[strains[0]]

counts = conds.groupby(['strain'])['fly_pair'].count()
print(counts)
ftj['{}_legend'.format('strain')] = ftj[['strain']].applymap(lambda x: '{} (n={})'.format(x, counts[x]))
#%%
# Bin dist_to_other during chasing and singing
#species_palette = {'Dmel': 'lavender', 

plot_bar = False
grouper_palette = 'cubehelix'
error_type = 'ci'
plot_pairs = False

plot_type = 'bar' if plot_bar else 'point'

for curr_species, df_ in meanbouts_courting.groupby('species'): #_courting.groupby('species'):
    fig, axn = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=False)
    for ai, behav in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary']):
        if behav == 'orienting_binary':
            df_ = meanbouts_orienting[meanbouts_orienting['species']==curr_species]
        ax=axn[ai]
        if plot_pairs:
            sns.stripplot(data=df_,
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, 
                        hue='{}_legend'.format('strain'), palette=grouper_palette, legend=False,
                        edgecolor='w', linewidth=0.25, dodge=True, jitter=True)
        if plot_bar:
            sns.barplot(data=df_,
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, lw=0.5,
                        errorbar=error_type, errcolor=bg_color, errwidth=0.75,
                        hue='{}_legend'.format('strain'), palette=grouper_palette, 
                        fill=plot_pairs==False, legend=1)
        else:
            sns.pointplot(data=df_, 
                        x='binned_dist_to_other', 
                        y=behav, ax=ax, scale=0.5,
                        errorbar=error_type, errwidth=0.75,
                        hue='{}_legend'.format('strain'), palette=grouper_palette, legend=1)
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
yak = ftj[(ftj['species']=='Dyak') & (ftj['strain']=='RL_Ruta_Lab')]
mel = ftj[(ftj['species']=='Dmel') & (ftj['strain']=='CS_Mai')]
tmpdf = pd.concat([yak, mel])

courting = tmpdf[tmpdf['courting']==1].copy()
print(courting.shape)
#courting.loc[courting['chasing_binary']==1, 'behavior'] = 'chasing'
#courting.loc[courting['singing_binary']==1, 'behavior'] = 'singing'
#%
# average over subbout
bout_type = 'frames' #'subboutnum'
meanbouts_courting = courting.groupby(['species', 'acquisition', 'fly_pair', 
                        'binned_dist_to_other'])[['orienting_binary', 'chasing_binary', 'singing_binary']].mean().reset_index()
meanbouts_courting.head()

#%%
# Bin dist_to_other during chasing and singing
species_palette = {'Dmel': 'lavender', 
                   'Dyak': 'mediumorchid'}
error_type = 'ci'

fig, axn = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(8, 4))
for ai, behav in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary']):
    ax=axn[ai]
    if behav in ['orienting_binary', 'chasing_binary']:
        plotd = meanbouts_orienting
    else:
        plotd = meanbouts_courting
    sns.barplot(data=plotd,
                    x='binned_dist_to_other', 
                    y=behav, ax=ax, 
                    errorbar=error_type, errcolor=bg_color,
                    hue='species', palette=species_palette, 
                    edgecolor='none')
    if ai!=2:
        ax.legend_.remove()
   
axn[0].set_ylabel('p(orienting)') 
#axn[1].set_ylabel("p(chasing|courtship)")
axn[1].set_ylabel("p(chasing)")
axn[2].set_ylabel("p(singing|courtship)")

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


#%%
#hist_color = 'mediumorchid'
c1 = 'r'
c2 = 'b'
hist_bins = 100
cumulative=True
fill = cumulative is False
#ylim = 0.2
hist_palette = 'cubehelix'

#species_palettes = {'Dmel': 'red', 'Dyak': 'blue'}
species_palette = dict((k, c2) for k in yak_strains) 
species_palette.update(dict((k, c1) for k in mel_strains))

chasedf = ftj[(ftj['chasing_binary'])].copy().reset_index(drop=True)
singdf = ftj[(ftj['singing_binary'])].copy().reset_index(drop=True)

grouper = 'strain'

fig, axn = plt.subplots(1, 3, figsize=(6,3), sharex=True, sharey=True)

for curr_species, df_ in ftj.groupby('species'):
    for ai, beh in enumerate(['orienting_binary', 'chasing_binary', 'singing_binary']):
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
                    hue='strain', common_norm=False,
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

c1 = 'mediumturquoise'
c2 = 'mediumorchid'
species_palette = dict((k, c2) for k in yak_strains) 
species_palette.update(dict((k, c1) for k in mel_strains))

fig, ax = plt.subplots()
for curr_species, df_ in ftj[ftj['courting']==True].groupby('species'):

    sns.histplot(data=df_, ax=ax,                    
             x='dist_to_other', bins=hist_bins, stat='probability',
            hue='strain', common_norm=False,
            element='step', lw=0.5, alpha=1, fill=fill, cumulative=cumulative,
            palette=species_palette) 

legh = putil.custom_legend(labels=['Dmel', 'Dyak'], colors=[c1, c2])
ax.legend(handles=legh, loc='upper left', bbox_to_anchor=(1,1), frameon=False, 
            title='', fontsize=6)
ax.set_box_aspect(1)

figname = 'courting_v_dist_to_other_hist'
plt.savefig(os.path.join(figdir, figname+'.png'))
#%%
# mean dist_to_other during courtship for each strain?
plotd = ftj[(ftj['courting']==True)].copy()

fig, ax = plt.subplots()
sns.barplot(data=plotd, y='dist_to_other', x='species',
            hue='strain', palette=species_palette, ax=ax)


#%%


fig, ax = plt.subplots()
#plotd = ftj[(ftj['courting']==True) & (ftj['species']=='Dyak')].copy()
plotd = ftj[(ftj['courting']==True)].copy()
sns.pointplot(data=plotd, ax=ax,
              x='binned_dist_to_other', y='singing_binary', #y='chasing_binary', 
              hue='strain', palette=species_palette)


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
min_wing_ang_deg = 40
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
use_jaaba= True #True

if use_jaaba:
    court_ = ftj[(ftj['sex']=='m') & (ftj['chasing']==True)
                    & (ftj['facing_angle']<=max_facing_angle)
                    & (ftj['max_wing_ang']>=min_wing_ang)].copy()
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
 
    plt.subplots_adjust(top=0.9, left=0.1)
    #fig.text(0.5, 0.05, 'targ_rel_pos_x', ha='center')
    #fig.text(0.01, 0.5, 'targ_rel_pos_y', va='center', rotation='vertical')
    fig.suptitle('Courting, {}, jaaba={}'.format(curr_species, jaaba_str), 
                 x=0.3, y=0.95, fontsize=10) 
    for ax in axn.flat[ai+1:]:
        ax.axis('off') 
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, wspace=0.2)
    
    putil.label_figure(fig, figid) 
    figname = 'spatial_occupancy_{}_per-flypair_jaaba-{}'.format(curr_species, jaaba_str) 
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
    figname = 'occ_{}_{}_jaaba-{}'.format(curr_species, curr_strain, jaaba_str)
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
    figname = 'spatial_occupancy_all-pairs_{}_jaaba-{}'.format(curr_species, jaaba_str) 
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

    fig.text(0.1, 0.95, 'Female position from male-centered view', fontsize=8)
    putil.colorbar_from_mappable(ax, norm=norm, cmap=cmap, axes=[0.92, 0.3, 0.01, 0.4],
                                hue_title=stat)

    putil.label_figure(fig, figid)     
    figname = 'male-perspective_{}_all-pairs_jaaba-{}'.format(sp, jaaba_str) 
    plt.savefig(os.path.join(figdir, '{}.png'.format(figname)))
print(figdir)

#%%
