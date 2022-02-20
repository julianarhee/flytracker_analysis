#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   convert_frames_to_video.py
@Time    :   2022/02/20 15:17:41
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
'''


import os
import glob
import cv2
import re
import numpy as np
import pylab as pl
import pandas as pd

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

rootdir = '/mnt/sda/Videos/sound-chamber' 

def select_acquisition(rootdir = '/mnt/sda/Videos/sound-chamber'):
    found_acqs = sorted(glob.glob(os.path.join(rootdir, '2022*', 'frames')), key=natsort)
    acquisitions = sorted([os.path.split(os.path.split(f)[0])[-1] for f in found_acqs], key=natsort)
    
    #needs_movie = sorted([f for f in found_acqs if len(glob.glob(os.path.join(os.path.split(f)[0], '*.mp4')))==0], key=natsort)
    needs_movie =sorted([os.path.join(rootdir, a, 'frames') for a in acquisitions if not os.path.exists(os.path.join(rootdir, a, '%s.mp4' % a))], 
                        key=natsort)     
    for i, f in enumerate(needs_movie):
        acq = f.split('%s/' % rootdir)[-1].split('/frames')[0] 
        print(i, acq)

    sel = input("Select IX of acq: ")
    acq_ix = int(sel)
    sel_dir = needs_movie[acq_ix]

    acquisition_dir=None
    confirm = input("Confirm dir -- %s -- (sel Y/n): " % sel_dir)
    if confirm=='Y':
        frame_dir = sel_dir
    
    return frame_dir

def check_frames(frame_dir):
    '''
    Check format of frames saved from acquisition (in frame_dir).
    Convert to .png if .npz.
    
    Args:
    -----
    frame_dir: (str)
        Dir containing the individual frames (or npz files) froma acquisition. 
    
    Returns:
    --------
    src_dir: (str)
        Source dir for frame files to combine into movie.
    '''
    ftype=None
    src_dir=None 
    found_files = sorted(os.listdir(frame_dir), key=natsort)
    testf = found_files[0]
    
    if testf.endswith('npz'):
        ftype = 'npz' 
        
    elif testf.endswith('png'):
        ftype = 'png'
        src_dir = frame_dir
        
    elif testf.endswith('tiff'):
        ftype = 'tiff'
    else:
        print("Unknow file type: %s" % testf)        

    if ftype=='npz':
        src_dir = npz_to_png(frame_dir)
            
    return src_dir

def npz_to_png(frame_dir):
    '''
    Converts all .npz in FRAME_DIR and converts to .png.
    Save to 'frames-tmp'
    
    Args:
    -----
    frame_dir: (str)
        Directory containing all the .npz files (assumes subdir is 'frames')
        
    Returns:
    --------
    dst_dir: (str)
        Directory to which all the .png files were saved
    '''
    
    print("Converting npz to png...")
    fpaths = sorted(glob.glob(os.path.join(frame_dir, '*.npz')), key=natsort)
   
    base_dir = frame_dir.split('/frames')[0]
    dst_dir = os.path.join(base_dir, 'frames-tmp')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    do_conversion=True
    if os.path.exists(dst_dir):
        n_found = len(glob.glob(os.path.join(dst_dir, '*.png')))
        if n_found==len(fpaths):
            print("PNGs all exist. Returning.")
            do_conversion=False
            
    if do_conversion:
        print("Saving to tmp dir: %s" % dst_dir) 
        for fpath in fpaths:
            f = np.load(fpath)
            im = f['arr_0']
            fnum = int(os.path.splitext(os.path.split(fpath)[-1])[0])
            outf = os.path.join(dst_dir, '%06d.png' % fnum)
            cv2.imwrite(outf, im)
     
    return dst_dir

def get_framerate(frame_dir):
    
    acquisition_dir = os.path.split(frame_dir)[0]
    performance_info = os.path.join(acquisition_dir, 'performance.txt')
    metadata = pd.read_csv(performance_info, sep="\t")
    fps = float(metadata['frame_rate'])
    return fps

def write_frames_to_mp4(movname, frame_dir, fps=None):
    '''
    Create mp4 movie from pngs found in frame_dir.
    '''
    acq_dir = os.path.split(frame_dir)[0]
    outfile = os.path.join(acq_dir, '%s.mp4' % movname)
    print("Writing movie: %s (FPS=%i Hz)" % (outfile, fps))

    if fps is None:
        cmd='ffmpeg -y -i ' + frame_dir+'/%06d.png -vcodec libx264 -f mp4 -pix_fmt yuv420p ' + outfile
    else:    
        cmd='ffmpeg -y -r %.2f ' % fps + '-i ' + frame_dir+'/%06d.png -vcodec libx264 -f mp4 -pix_fmt yuv420p ' + outfile
    print(cmd)
    os.system(cmd)
     
 
if __name__=='__main__':
    frame_dir = select_acquisition()
    print(frame_dir)
    
    src_dir = check_frames(frame_dir)     
    print("Frames in: %s" % src_dir)

    fps = get_framerate(src_dir)
    acquisition = os.path.split(src_dir.split('/frames')[0])[-1] 
    print("ACQ: %s" % acquisition)
 
    write_frames_to_mp4(acquisition, src_dir, fps=fps)
    
    
    