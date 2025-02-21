#!/usr/bin/env python3
"""
@File    :   concatenate_videos.py
@Time    :   2025/02/20 16:25:55
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

This script is used to concatenate a bunch of subvideos 
into a single video.

% python concatenate_videos.py 
    --basedir '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector' 
    --session 20250219-1559
    --movie_fmt avi
"""
#%%
import os
import glob
import subprocess
import argparse

def get_acquisition_dirs(session, basdir):
    """
    Get all the session directories in the base directory
    """
    acquisition_dirs = glob.glob(os.path.join(basdir, '{}*'.format(session)))

    return acquisition_dirs

def get_subvideo_fpaths(acquisition_dir, movie_fmt='avi'):
    """
    Get all the subvideo file paths in the current acquisitiondirectory.
    Assumes the subvideos are currently in the acquisition directory.
    """
    acquisition_prefix = os.path.split(acquisition_dir)[-1]
    subvideo_fpaths = sorted(glob.glob(os.path.join(acquisition_dir, '{}*.{}'.format(acquisition_prefix, movie_fmt))))

    return subvideo_fpaths

def move_to_subvideo_dir(subvideo_fpaths, session_dir):
    """
    Move all the subvideo file paths to the subvideo directory
    """
    subvideo_dir = os.path.join(session_dir, 'subvideos')
    if not os.path.exists(subvideo_dir):
        os.makedirs(subvideo_dir)

    for subvideo_fpath in subvideo_fpaths:
        subprocess.run(['mv', subvideo_fpath, subvideo_dir])

    return subvideo_dir

#%%

basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector'
session = '20250219-1559'
movie_fmt = 'avi'

def do_concat_for_acquisition(acquisition_dir):

    # Generate filename for concatenated video from acquisition_dir basename
    acquisition_basename = os.path.split(acquisition_dir)[-1]
    print(acquisition_basename)
    outpath = os.path.join(acquisition_dir, '{}.{}'.format(acquisition_basename, movie_fmt))

    # Check if already exists
    if os.path.exists(outpath):
        print('Concatenated video already exists: %s' % outpath)
        return

    # %
    subvideo_list = get_subvideo_fpaths(acquisition_dir, movie_fmt=movie_fmt)
    #%

    try:
        # Write subvideo list to text file, include path in single quotes:
        with open(os.path.join(acquisition_dir, 'subvideo_list.txt'), 'w') as f:
            for subvideo in subvideo_list:
                f.write('file ' + "'" + subvideo + "'" + '\n')
        #%
        #%
        # Use ffmpeg to concatenate the subvideos specified in the text file
        #concat_cmd = 'ffmpeg -f concat -safe 0 -i ' + "'{}'" %s -c copy %s' % (os.path.join(acquisition_dir, 'subvideo_list.txt'), os.path.join(acquisition_dir, 'concatenated_video.avi'))
        input = os.path.join(acquisition_dir, 'subvideo_list.txt')
        concat_cmd = 'ffmpeg -f concat -safe 0 -i ' + "'{}'".format(input) + ' -c copy ' + "'{}'".format(outpath) 
        #print(concat_cmd)
        subprocess.run(concat_cmd, shell=True)
    except Exception as e:
        print('Error concatenating videos: %s' % e)
        return
    #%

    if os.path.exists(outpath):
        # Move subvideos to subvideo dir
        subvideo_dir = move_to_subvideo_dir(subvideo_list, acquisition_dir)
        print('... Subvideos moved to: %s' % subvideo_dir)




# %%

# Run as main script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process FlyTracker data for relative metrics.') 
    parser.add_argument('--basedir', type=str, help='Directory containing acquisition dirs.')  
    parser.add_argument('--movie_fmt', type=str, default='avi', help='Movie format (default: avi).')
    parser.add_argument('--session', type=str, default='yyyymmdd', help='Session date (yyyymmdd or yyyymmdd-HHMM).')

    args = parser.parse_args()
    # 
    basedir = args.basedir
    session = args.session
    movie_fmt = args.movie_fmt

    acq_dirs = get_acquisition_dirs(session, basedir)

    #acquisition_dir = acq_dirs[0]
    #print(acquisition_dir)

    for acquisition_dir in acq_dirs:
        do_concat_for_acquisition(acquisition_dir)
    
    #acquisition_dir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/caitlin_data/38mm_projector/20250219-1016_fly1_Dyak_WT_5do_gh'
    #concatenate_videos(acquisition_dir)
    print('Done!')