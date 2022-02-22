#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   create_movie_clip.py
@Time    :   2021/11/30 14:52:55
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

Crop movies

python crop_movies.py -E single_20mm_1x1 --fmt avi -s '00:05.7' -e 100 --crop -A 20220203-0951_sant_3do_sh --submovie -N 0 --cat
 
'''

#%%
import sys
import os
import re
import glob
import optparse
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#from concatenate_movies import concatenate_subvideos

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

def get_start_and_end_sec(start_time, end_time, time_in_sec=False):
    '''
    Get start and end time in sec.
    
    Args:
    -----
    start_time, end_time: (str or float)
        If str, must be format:  'MM:SS.ms' (estimated with VLC)
        If int, assumes all the way to end (100)
       
    time_in_sec: (bool)
        If INT provided for either start_time or end_time, specifies whether the number is in seconds or minutes
        Set to False if in minutes (e.g., 100 minutes to get full video, a big number). 
    '''
    tstamps=[] 
    for tstamp in [start_time, end_time]:
        if isinstance(tstamp, str):
            # Get start time in sec 
            minutes, secs = [float(i) for i in tstamp.split(':')]
            #print(minutes, secs)
            tstamp_sec = minutes*60. + secs
        else:
            tstamp_sec = float(tstamp) if time_in_sec else float(tstamp)*60. 
        tstamps.append(tstamp_sec) 

    return tstamps[0], tstamps[1]


def do_clip(movie_fpath, start_time, end_time, movie_prefix=None, movie_num=0, time_in_sec=False, 
            fmt=None, verbose=False):
    '''
    Crop specified movie.
    
    Args:
    -----
    input_movie: (str)
        Full path to movie to crop.
    
    start_time, end_time: (str or float)
        If str, must be format:  'MM:SS.ms' (estimated with VLC)
        If int, assumes all the way to end (100)
       
    time_in_sec: (bool)
        If INT provided for either start_time or end_time, specifies whether the number is in seconds or minutes
        Set to False if in minutes (e.g., 100 minutes to get full video, a big number). 

    '''

    acqdir = os.path.split(movie_fpath)[0]
    clip_dir = os.path.join(acqdir, 'clips')
    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)
        
        
    # Format start/end time
    start_time = start_time if (isinstance(start_time, str) and ':' in start_time) \
            else float(start_time) 
    end_time = end_time if (isinstance(end_time, str) and ':' in end_time) \
            else float(end_time)       

    print("    Cropping: %s" % movie_fpath)

    # Crop
    # -------------------------------------------------------------------- 
    existing_clips = glob.glob(os.path.join(acqdir, 'clips', '*.%s' % fmt))
    nclips = len(existing_clips)
    movie_basename = os.path.split(os.path.splitext(movie_fpath)[0])[-1]
   
    prefix = '%s' % (movie_prefix) if movie_prefix is not None else 'clip%03d' % nclips 
    start_str = ''.join(start_time.split(':'))
    end_str = ''.join(end_time.split(':'))
    timestr = '%s-%s' % (start_str, end_str)
    output_fname = '%s_%s__%s.%s' % (prefix, timestr, movie_basename, fmt)
    output_movie = os.path.join(clip_dir, output_fname)
    
    # Specify how much time to keep
    start_time_sec, end_time_sec = get_start_and_end_sec(start_time, end_time, 
                                                 time_in_sec=time_in_sec)
    if verbose:
        print('    Start/end time (sec): %.2f, %.2f' % (start_time_sec, end_time_sec))
 
    # Cropy movie and save
    t = time.time()
    ffmpeg_extract_subclip(movie_fpath, start_time_sec, end_time_sec, 
                           targetname=output_movie)
    elapsed = time.time() - t
    print(elapsed)

    return


#%%
def extract_options(options):

    parser = optparse.OptionParser()
    parser.add_option('--rootdir', action="store", 
                      dest="rootdir", default='/mnt/sda/Videos', 
                      help="out path directory [default: /mnt/sda/Videos]")

    parser.add_option('-E', '--assay', action="store", 
                      dest="assay", default='single_20mm_1x1', 
                      help="Name of dir containing acquisition subdirs, e.g., ")
  
    parser.add_option('-A', '--acquisition', action="store", 
                       dest='acquisition', default=None, 
                       help='Name of acquisition, or dir containing .avi/mp4 files to concatenate (default grabs all dirs found in SESSION dir)')

    parser.add_option('-s', '--start', action="store", 
                      dest="start_time", default=None, 
                      help="Start time (sec or min). If str, MM:SS.ms. Specify NONE to not crop.")
    parser.add_option('-e', '--end', action="store", 
                      dest="end_time", default=0, 
                      help="End time (sec or min). If str, MM:SS.ms. Specify large number and time_in_sec=False to go to end.")
    parser.add_option('--seconds', action="store_true", 
                      dest="time_in_sec", default=False, 
                      help="If number provided for start or end time, specifies whether is in MIN or SEC.")

    parser.add_option('--fmt', '-f', action="store", 
                      dest="movie_format", default='avi', 
                      help="Movie format (default: avi)")

    parser.add_option('--verbose', '-v', action="store_true", 
                      dest="verbose", default=False, 
                      help="Flag to print verbosely")
    parser.add_option('--prefix', '-p', action="store", 
                      dest="movie_prefix", default=None, 
                      help="Prefix to save movie clip")

    (options, args) = parser.parse_args()

    return options

rootdir = '/mnt/sda/Videos'
assay = 'single_20mm_triad_2x1'
acquisition = '20220203-1101_triad_yak_7do_sh'
is_submovie=True
movie_num=0
start_time0 = '00:05'
end_time=100
time_in_sec=False
delete_submovies=False

#%%
if __name__ == '__main__':

    opts = extract_options(sys.argv[1:])

    rootdir = opts.rootdir
    assay = opts.assay
    acquisition = opts.acquisition
          
    start_time = opts.start_time #'00:06.75'
    end_time = opts.end_time # 100 #'32:38.0'
    time_in_sec = opts.time_in_sec

    fmt = opts.movie_format   
    verbose=opts.verbose

    movie_prefix=opts.movie_prefix
    
    
    # Get src dirs
    basedir = os.path.join(rootdir, assay) 
    acqdir = os.path.join(basedir, acquisition)    
    
    found_movies = glob.glob(os.path.join(acqdir, '*.%s' % fmt))
    assert len(found_movies)==1, "Movie not found: %s" % str(found_movies) 
    movie_fpath = found_movies[0]
    
    
    do_clip(movie_fpath, start_time, end_time, movie_prefix=movie_prefix, time_in_sec=time_in_sec, 
                fmt=fmt, verbose=verbose)

