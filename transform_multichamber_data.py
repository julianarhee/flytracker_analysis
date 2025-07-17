#/usr/bin/env python3
# # -*- coding: utf-8 -*-
'''
 # @ Author: Juliana Rhee
 # @ Create Time: 2025-04-26 16:10:52
 # @ Modified by: Juliana Rhee
 # @ Modified time: 2025-04-26 16:33:36
 # @ Description: Process multichamber data for relative metrics. Save each to dstdir, save aggregate to local dir. 
 NOTE: This script needs the metadata (saved as .csv) to dstdir. 
 
 '''
#%%
import os
import glob
import argparse

import numpy as np

import pandas as pd
import polars as pl

import seaborn as sns
import matplotlib.pyplot as plt

import utils as util
import transform_data.relative_metrics as rel
import cv2
import traceback

#%%
#import plotting as putil
#plot_style='white'
#putil.set_sns_style(plot_style, min_fontsize=18)
#bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%

def get_meta_data_for_experiment(meta_fpath):
    '''
    Get metadata for a specific experiment from dstdir.
    
    Args:
        dstdir (str): Directory where metadata files are stored.
        experiment (str): Name of the experiment to find metadata for. Parent folder of all acquisition folders. 
        Should be the same name as .csv tab on meta data google sheets.
    '''
#     meta_fpaths = glob.glob(os.path.join(dstdir, '*{}.csv'.format(experiment)))
#     if len(meta_fpaths) > 1:
#         print("Found multiple metadata files: {}".format(meta_fpaths))
#     elif len(meta_fpaths) == 0:
#         raise FileNotFoundError('No metadata file found for experiment {}'.format(experiment))
#     else:
#         print("Found metadata file: {}".format(meta_fpaths[0]))
#     meta_fpath = meta_fpaths[0]
     allmeta = pd.read_csv(meta_fpath)
    if 'acquisition' not in allmeta.columns:
        allmeta['acquisition'] = allmeta['file_name']
        
    return allmeta

def get_google_sheets_name(meta_fpath):
    
    return os.path.splitext(os.path.split(meta_fpath)[-1].split('- ')[-1])[0]


def get_all_meta_data(dstdir, experiment='strains', return_all=False):
    '''
    Gets metadata for all 2x2 (or other) multichamber assays.
    Assumes .csv files are in dstdir, and have 2x2 in name.
    Loads all bec winged vs. wingless contains pairs that can be used for strain comparisons.
    Args:
        dstdir (str): Directory where metadata files are stored.
        experiment (str): Name of the experiment to find metadata for. Parent folder of all acquisition folders. 
                          Should be the same name as .csv tab on meta data google sheets.
        return_all (bool): If True, returns all_metadata, strain_metadata, otherwise only strain metadata.
    '''
    meta_fpaths = glob.glob(os.path.join(dstdir, '*.csv'))
    meta_list = []
    for meta_fpath in meta_fpaths:
        print(meta_fpath)
        meta_ = pd.read_csv(meta_fpath)
        last_col = 'tracked and checked for swaps and copulation annotated'
        last_col_ix = meta_.columns.tolist().index(last_col) 
        incl_cols = [i for i in meta_.columns[:last_col_ix+1] if i!='notes']
        meta_ = meta_[incl_cols]     
        meta_[last_col] = [i in [1, 'yes'] for i in meta_[last_col]]
        if 'acquisition' not in meta_.columns:
            meta_['acquisition'] = meta_['file_name'] 
            
        # Include meta file name (this also identifies the acquisition data source) 
        # especially for subset of 2x2 data moved to JAABA_classifiers subfolder
        experiment_tabname = get_google_sheets_name(meta_fpath)
        meta_['experiment'] = experiment_tabname
        
        # Add 
        meta_list.append(meta_) 
    allmeta0 = pd.concat(meta_list) 
    allmeta0['species_male'].unique()
    
    # Specific to wingless vs winged 2x2 data (WT strain data for Dmel/Dyak can be re-used)
    if experiment=='strains':
        # Only select WINGED pairs
        strainmeta = allmeta0[(allmeta0['manipulation_male']!='wingless')
                    & (allmeta0[last_col]==1)].copy()
        # check names
        # Replace 'CS mai' with "CS Mai" in allmeta
        strainmeta['strain_male'] = strainmeta['strain_male'].map(lambda x: x.replace('CS mai', 'CS Mai'))
        strainmeta['strain_male'] = strainmeta['strain_male'].map(lambda x: x.replace('CS Mai ', 'CS Mai'))

    if return_all:
        return allmeta0, strainmeta
    
    return strainmeta

#%
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
    NOTE: meta fly_num is 1-indexed. fly_pair is 1-indexed, and follows the google spreadsheet meta data.
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

def transform_multichamber_data(acqdir, feat_, trk_, cop_dict):
    '''
    For multi-arena acquisition, transform the data to get relative positions
    '''
    cap = rel.get_video_cap(acqdir, movie_fmt='.avi')
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

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
        transf_df = rel.do_transformations_on_df(curr_trk, 
                                frame_width, frame_height, 
                                feat_=curr_feat, cop_ix=cop_ix,
                                flyid1=flyid1, flyid2=flyid2, 
                                get_relative_sizes=False)
        
        assert len(transf_df[transf_df['id']==flyid1]) == len(transf_df[transf_df['id']==flyid2]), 'Different number of frames for each fly'
        assert len(transf_df[transf_df['id']==flyid1]['frame']) == len(transf_df[transf_df['id']==flyid1]['frame'].unique()), 'Non-unique frames'
        assert transf_df.shape[0] == transf_df.index[-1]+1, "Bad frame indexing"
        acq_dfs.append(transf_df)

    acq_df = pd.concat(acq_dfs)

    return acq_df

def process_multichamber_acquisitions(acq_dirs, processed_outdir, allmeta0,
                                        fps=60, mov_is_upstream=False,
                                        subfolder='*', filter_ori=True,
                                        array_size='3x3'):  
    '''
    Load and aggregate transformed data for multichamber assays.
    '''
    no_actions = []
    d_list = []
       
    for acqdir in acq_dirs:
        acq = os.path.split(acqdir)[-1]
        out_fpath = os.path.join(processed_outdir, '{}_df.parquet'.format(acq))
         
        print("Processing acquisition: {}".format(acq))
        calib_, trk_, feat_ = util.load_flytracker_data(acqdir, 
                                    fps=fps, 
                                    calib_is_upstream=mov_is_upstream,
                                    subfolder=subfolder,
                                    filter_ori=True)
        if 'acquisition' not in trk_.columns:
            trk_['acquisition'] = acq
        # Get meta info for current acquisition
        meta = allmeta0[allmeta0['acquisition']==acq]

        # Check that the number of unique fly ids in feat_trk is 2x as the number of unique fly pairs in meta
        assert len(trk_['id'].unique()) == 2*len(meta['fly_num'].unique()), 'Incorrect fly ID to pair assignment'

        # Assign conditions
        trk_ = assign_sex(trk_)
        trk_ = assign_frame_number(trk_)
        print("Assigning strain to multichamber data")
        trk_ = assign_strain_to_multichamber(trk_, meta, array_size=array_size)

        # Check for copulations
        try:
            actions_fpath = glob.glob(os.path.join(acqdir, '*', '*-actions.mat'))[0] 
            cop_dict = get_copulation_frames(actions_fpath)     
        except IndexError:
            print("No actions file for {}".format(acq))
            no_actions.append(acq)
            cop_dict = dict((k, -1) for k in trk_['id'].unique()) 
            
        # for i, df_ in df.groupby('id'):
        #     cop_ix = cop_dict[i]
        #     # Only take frames up to copulation
        #     if cop_ix == -1:
        #         df.loc[(df['id']==i), 'copulating'] = False
        #     else:
        #         df.loc[(df['id']==i) & (df['frame']<=cop_ix), 'copulating'] = False
        acq_df = transform_multichamber_data(acqdir, feat_, trk_, cop_dict)   
        
        # Save processed df
        acq_df.to_parquet(out_fpath, engine='pyarrow', compression='snappy')


#%%

    #%%
def load_aggregate_datafile(save_fpath):
    
    df0 = pd.read_parquet(save_fpath)

    return df0

def check_for_processed_file(processed_outdir, acqs):
    '''
    Check if processed data file exists for a given acquisition.
    Args:
        processed_outdir (str): Directory where processed data files are stored (processed_mats).
        acq (str): Acquisition name to check for.
    Returns:
        bool: True if processed file exists, False otherwise.
    '''
    not_found = []
    for acq in acqs:
        out_fpath = os.path.join(processed_outdir, '{}_df.parquet'.format(acq))
        if not os.path.exists(out_fpath):
            not_found.appedn(acq)
    return not_found

def cycle_and_load_processed_acquisitions(processed_outdir, acqs=None):
    ''' Cycle through all processed acquisition directories and load data.
    Args:
        acq_dirs (list): List of acquisition directories to process.
        processed_outdir (str): Directory where processed data files are stored (processed_mats).
        acqs (list, optional): List of acquisition names to process. If None, processes all found in processed_outdir.
    Returns:
        df0 (pd.DataFrame): Aggregated DataFrame containing all processed data.
    '''
    d_list = []
    if acqs is None:
        # Load/aggregate ALL processed data found in processed_outdir
        out_fpaths = glob.glob(os.path.join(processed_outdir, '*_df.parquet'))
        print("Found {} processed files.".format(len(acqs)))
    else:
        print("Processing {} acquisitions.".format(len(acqs)))
        out_fpaths = [os.path.join(processed_outdir, '{}_df.parquet'.format(acq)) for acq in acqs]  
        
    for out_fpath in out_fpaths:
        acq = os.path.split(f)[-1].replace('_df.parquet', '') 
        if os.path.exists(out_fpath):
            acq_df = pd.read_parquet(out_fpath)
            d_list.append(acq_df)
        else:
            print("No processed data for {}".format(acq))
    df0 = pd.concat(d_list) 

     #% # Reassign IDs for multi-day data
    curr_id = 0
    for (acq, idnum), df_ in df0.groupby(['acquisition', 'id']):
        #print(acq, idnum)
        df0.loc[(df0['acquisition']==acq) & (df0['id']==idnum), 'global_id'] = curr_id
        curr_id += 1
    df0['global_id'].unique()    
    print(df0['acquisition'].unique())
    #%
    # Assign species to each line based on what file_name is
    df0['species'] = df0['acquisition'].map(lambda x: 'Dyak' if 'yak' in x else 'Dmel')    
    #%
    #aggregate_processed_datafile = os.path.join(processed_outdir, '38mm_strains_df.parquet')

    return df0

def aggregate_and_save_acquisitions(processed_outdir, save_fpath, acqs=None):
    '''
    Aggregate processed data from multichamber assays and save to local directory.
    Args:
        processed_outdir (str): Directory where processed data files are stored (processed_mats).
        savedir (str): Directory where aggregated data will be saved as parquet file.
    '''
#     aggregate_processed_datafile = os.path.join(savedir, 
#                                             '38mm_strains_df.parquet')
#     aggregate_processed_datafile_all = os.path.join(savedir, 
#                                             '38mm_all_df.parquet')


    # Aggregate all processed data
    df0 = cycle_and_load_processed_acquisitions(processed_outdir, 
                                                        acqs=acqs) 
    # Save to local directory
    df0.to_parquet(save_fpath, engine='pyarrow', compression='snappy')        
    print("Saved aggregated data to: {}".format(save_fpath))

    return df0
    
    
#%% 
def aggregate_main_from_meta(dstdir, array_size='2x2', save_fname='38mm_2x2_strains',
                        create_new=True, fps=60, 
                        rootdir='/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data',
                        remake_acquisition=False):
    '''
    Load data from metadata and process multichamber acquisitions.
    Args:
        rootdir (str): Parent directory of experiment folder that in turn contains acquisiton folders.
        # rootdir default is dropbox caitlin_data: need this to get all the different sources of meta data
        dstdir (str): Directory where processed data will be saved.
        array_size (str): Size of the multichamber array (e.g., '2x2', '3x3').
        experiment (str): Name of the experiment to process (e.g., 'strains').
        create_new (bool): If True, creates new aggregate data.
        fps (float): Frame rate of the acquisition.
    '''
    save_fpath = os.path.join(dstdir, '{}.parquet'.format(save_fname))

    if (not create_new) and os.path.exists(save_fpath):
        df0 = load_aggregate_datafile(save_fpath)
        return df0
    
    # % Load metadata for YAK and MEL strains
    allmeta0, strainmeta = get_all_meta_data(dstdir, experiment='strains' 
                                             return_all=True) 
    #%
    # STRAINS -------------------------------------------------
    in_jaaba = ['2x2 winged vs. wingless', '2x2 yak strains']
    all_acq_dirs = []
    for i, row in strainmeta.iterrows():
        if row['experiment'] in in_jaaba:
            jaaba_srcdir = os.path.join(rootdir, 'JAABA_classifiers', '38mm_multichamber_winged-wingless_classifier')
            acq_dir = os.path.join(jaaba_srcdir, row['file_name'])
        else:
            acq_dir = os.path.join(rootdir, row['experiment'], row['file_name'])  
        all_acq_dirs.append(acq_dir)                 
    #all_acq_dirs = [os.path.join(srcdir, ac) for ac \
    #                                in strainmeta['file_name'].unique()]
    
    non_existing = [a for a in all_acq_dirs if not os.path.exists(a)]
    if len(non_existing)>0:
        print("Non-existing directories: ")
        for n in non_existing:
            print(os.path.split(n)[-1])
    acq_dirs = [a for a in all_acq_dirs if os.path.exists(a)]
    acqs = [os.path.split(a)[-1] for a in acq_dirs]

    # Set output dirs
    processed_outdir = os.path.join(dstdir, 'processed_mats') 
    if not os.path.exists(processed_outdir):
        os.makedirs(processed_outdir) 
    print("Output saved to:", processed_outdir) 
    
    # Check for processed files
    not_found = check_for_processed_file(processed_outdir, acqs=acqs)

    #%
    if remake_acquisition or len(not_found) > 0:
        #% 
        print("The following acquisitions were not found in processed directory:") 
        process_multichamber_acquisitions(acq_dirs, processed_outdir, allmeta0,
                                            remake_acquisition=remake_acquisition, 
                                            fps=fps, mov_is_upstream=False,
                                            filter_ori=True,
                                            array_size=array_size)
        create_new = True
    
    if create_new:
        # Aggregate all processed data    
        print("Saving ALL 2x2 data to local.")
        df0 = aggregate_and_save_acquisitions(processed_outdir, save_fpath, acqs=acqs) 
    
    return df0  
#%% 



#%%

#minerva_base = '/Volumes/Juliana'
#srcdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data'
#print(srcdir)
#experiment = '38mm_strains'
#dstdir = os.path.join(minerva_base, 'free_behavior_analysis', experiment)
#array_size = '2x2'
#local_dir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior/38mm_strains'
#srcdir = os.path.join('/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data',
#                    'JAABA_classifiers', 
#                    '38mm_multichamber_winged-wingless_classifier', 'JAABA')

#create_new=True
#fps=60

dstdir = '/Volumes/Juliana/free_behavior_analysis/38mm_strains'
srcdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/JAABA_classifiers/38mm_multichamber_winged-wingless_classifier/JAABA'
localdir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior_analysis/38mm_strains'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process Multichamber FlyTracker data for relative metrics.')
    parser.add_argument('--dstdir', type=str, help='Directory to save processed data.')    
    parser.add_argument('--srcdir', type=str, help='Parent dir of acquisition folders.')
    parser.add_argument('--localdir', type=str, help='Local dir to save aggregate output.')
    parser.add_argument('--meta', type=str, help='Full path to metadata .csv file.')
    parser.add_argument('--savename', type=str, default='test', help='Save name of df saved to parquet file (aggregate).')

    parser.add_argument('--single', type=bool, default=False, help='Cycle and process all acqs from 1 source (default: False).')

    parser.add_argument('--new', type=bool, default=False, help='Create new aggregate data (default: True).')
    parser.add_argument('--remake', type=bool, default=False, help='Transform all data anew (default: False).')   
    parser.add_argument('--fps', type=float, default=60.0, help='Acquisition frame rate (default: 60).')   
    parser.add_argument('--array', type=str, default='2x2', help='Size of multichamber array (default: 2x2).')   
  
    args = parser.parse_args()
    # 
    dstdir = args.dstdir
    srcdir = args.srcdir
    meta_fpath = args.metapath
    create_new = args.new
    remake_acquisition = args.remake
    localdir = args.localdir
    fps = args.fps
    array_size = args.array
    save_fname = args.savename

    single_source = args.single
    if single_source:
        # Load metadata
        acq_dirs = [os.path.join(srcdir, i) for i in os.listdir(srcdir) if os.path.isdir(os.path.join(srcdir, i))]
        allmeta0 = get_meta_data_for_experiment(meta_fpath)

        process_multichamber_acquisitions(acq_dirs, dstdir, allmeta0,
                                        fps=fps, mov_is_upstream=False,
                                        subfolder='*', filter_ori=True,
                                        array_size=array_size) 

    else: 
        df0 = aggregate_main_from_meta(dstdir, array_size=array_size, 
                        save_fname=save_fname,
                        create_new=create_new, fps=fps, 
                        rootdir='/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data',
                        remake_acquisition=remake_acquisition)

        conds = df0[['species', 'strain', 'acquisition', 'fly_pair']].drop_duplicates()
        counts = conds.groupby(['species', 'strain'])['fly_pair'].count()
        print(counts)

    
    