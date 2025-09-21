#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   2p_utils.py
@Time    :   2024/10/05 17:26:48
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com

'''

import os
import glob
import scipy

import numpy as np
import pandas as pd
import mat73
import traceback

#%%

def load_virmen_meta(genotype, session, flyid, filenum, rootdir='/Volumes/juliana/2p-data'):
    '''
    Loads experiment metadata from _exper.mat file.
    Arguments:
        genotype -- string
        session -- string
        flyid -- string
        filenum -- integer
        rootdir -- string
    Returns:
        exper -- pandas dataframe
    '''
    #exper = scipy.io.loadmat(exper_fpath, struct_as_record=False)
    #exper['exper'][0,0]._fieldnames
    #exper['exper'][0,0].__dict__['variables'][0,0].__dict__['sphere_radius'][0]

    srcdir = os.path.join(rootdir, genotype)
    exper_fpaths = glob.glob(os.path.join(srcdir, session, flyid, 'virmen', 
                            '{}_*{}_{:03d}_exper.mat'.format(session, flyid, filenum)))
    exper_fpath = exper_fpaths[0]
    exper = load_custom_mat(exper_fpath) #, struct_as_record=False, squeeze_me=True)
    return exper

def load_mat_vr(mat_fpath, is_vr_file=True):
    '''
    Loads .mat file from session virmen folder and converts it to pandas dataframe.
    Note: Only works for _vr.mat files.
    '''
    mat = scipy.io.loadmat(mat_fpath, struct_as_record=is_vr_file)    
    vr = mat['vr'][0, 0] # Call vr.dtype to get other fields
    
    # Get storevars
    expr = pd.DataFrame(data=vr['storevars'],      
                        columns=[i[0] for i in vr['varnames'][0]])
    # This stuff along with other experiment info can prob go into class
    expr['angsize'] = float(vr['angsize'])
    expr['arcangle'] = float(vr['arcAngle']) 
   
    return expr

def load_expr_from_mat(session, acqnum, rootdir='/Volumes/juliana/2p-data'):
    '''
    Loads .mat file from session virmen folder and converts it to pandas dataframe.
    Saves to session/processed/matlab-files folder as .csv file.

    Arguments:
        session -- YYYYMMDD
        acqnum -- numeric 

    Keyword Arguments:
        rootdir -- _description_ (default: {'/Volumes/juliana/2p-data'})

    Returns:
        df
    '''
    virmen_dir = os.path.join(rootdir, session, 'virmen')
    mat_files = glob.glob(os.path.join(virmen_dir, '*{:03d}.mat'.format(acqnum)))

    assert len(mat_files) == 1, "Expected exactly one .mat file for acquisition {}, found {}".format(acqnum, len(mat_files))
    mat_fpath = mat_files[0]
    #mat_fpath = processed_mats[0]
    #print(mat_fpath)

    acq = os.path.splitext(os.path.split(mat_fpath)[1])[0]
    #print(acq)
    #%
    # Load mat
    #mat = mat73.loadmat(mat_fpath)
    mat = scipy.io.loadmat(mat_fpath)
    mdata = mat['expr']

    columns = [
        'toc', # 1
        'counter',
        'pos_x',
        'pos_y',
        'heading', # 5
        'target_x',
        'target_y',
        'clock_H', # 8
        'clock_M',
        'clock_S',
        'pulse_sent', # 11
        'ang_size',
        'moving',
        'visible',
        'frame_counter',                  # 1 (15) (FT PARAMS, 15-38)
        'delta_rotation_vector_cam_x',
        'delta_rotation_vector_cam_y',
        'delta_rotation_vector_cam_z',
        'delta_rotation_error_score',     # 5 (19)
        'delta_rotation_vector_lab_x',
        'delta_rotation_vector_lab_y',
        'delta_rotation_vector_lab_z',
        'absolute_rotation_vector_cam_x', # 9-11
        'absolute_rotation_vector_cam_y', 
        'absolute_rotation_vector_cam_z', 
        'absolute_rotation_vector_lab_x', # 12-14
        'absolute_rotation_vector_lab_y', 
        'absolute_rotation_vector_lab_z', 
        'integrated_pos_lab_x',  # 15 (29): should be same/similar to pos_x
        'intergrated_pos_lab_y', # 16 (30): should be same/similar to pos_y
        'animal_heading_lab',    # 17 (31): should be the same/similar to heading 
        'animal_movement_direction_lab', # 18 (32) - maybe this is traveling dir?
        'animal_movement_speed_lab',     # 19 (33)
        'lateral_motion_x', # 20
        'lateral_motion_y', # 21
        'timestamp',        # 22
        'sequence_counter', # 23 (# 37)
        'delta_timestamp',  # 24 (#38)
        #'alt.timestamp',   # 25
        'target_vel' # 39  
        ]

    df = pd.DataFrame(data=mat['expr'], columns=columns)
    df['acquisition'] = acq
    df['session'] = session
    df['acqnum'] = acqnum

    # Save to csv
    savedir = os.path.join(rootdir, session, 'processed', 'matlab-files')

    fname = 'virft_{}.csv'.format(acq)
    out_fpath = os.path.join(savedir, fname)

    df.to_csv(out_fpath, index=False)

    return df

#%%
def load_custom_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            #print(key)
            if key in ['exper']:
                #print("Exper")
                d[key] = _exper_todict(d[key]) # matobj #[key])
            else:
                if isinstance(d[key], scipy.io.matlab.mat_struct):
                    d[key] = _todict(d[key])
                elif isinstance(d[key], np.ndarray):
                    d[key] = _toarray(d[key])
        return d

    def _exper_todict(matobj):
        """
        Recursively constructs a dictionary from matlab object
        scipy.io.matlab._mio5_params.mat_struct.

        """
        mdict={}
        for strg in matobj._fieldnames:
            #print(strg)
            if strg in ['codeText']:
                continue
            elem = matobj.__dict__[strg] #if len(matobj.__dict__[strg])==0 else matobj.__dict__[strg][0][0] 
            try:
                if isinstance(elem, scipy.io.matlab.mat_struct):
                    mdict[strg] = _todict(elem)
                elif isinstance(elem, scipy.io.matlab.MatlabOpaque):
                    mdict[strg] = elem
                elif isinstance(elem, scipy.io.matlab.MatlabFunction):
                    mdict[strg] = elem.tolist().__dict__['function_handle'].__dict__['function']
                elif isinstance(elem, np.ndarray):
                    mdict[strg] = _toarray(elem)
                else:
                    mdict[strg] = elem
                #if isinstance(mdict[strg], np.ndarray) and len(mdict[strg])==1:
                #    mdict[strg] = mdict[strg][0]    
            except Exception as e:
                print("Error in ({})".format(strg))
                traceback.print_exc()
        return mdict #d['exper']

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    mat = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)

    #data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    d = _check_vars(mat)

    return d #_check_vars(data)


def load_caimin_matlab(mat_fpath, verbose=False):
    '''
    Load .mat file from CaImAn extract_rois.m (saed as mat73) and return a dict with extracted ROI traces.

    Arguments:
        mat_fpath -- _description_

    Keyword Arguments:
        verbose -- _description_ (default: {False})

    Returns:
        mdata (dict)
    '''
    mat = mat73.loadmat(mat_fpath)
    mdata = mat['plotdata']

    if verbose:
        for k, v in mdata.items():
            try:
                print(k, v.shape)
            except Exception as e:
                print("Issue with {}".format(k))
    return mdata


#%%%

def compute_ti(heading_diff, female_angle, behavior_time):
    # Parameters
    trial_length = 180
    hemi_length = int(trial_length / 2)
    trialStarts = range(len(behavior_time) - 1)
    
    vigor = []
    fidelity = []
    start_times = []

    # Compute tracking index
    for i in trialStarts:
        if i > hemi_length and i < (len(heading_diff) - hemi_length): #(len(behavior_time) - hemi_length):
            x = heading_diff[(i - hemi_length):(i + hemi_length + 1)]
            f = female_angle[(i - hemi_length):(i + hemi_length + 1)]
            t = behavior_time[(i - hemi_length):(i + hemi_length + 1)]
            start_times.append(behavior_time[i])
            mat = np.corrcoef(x, f)  # Correlation matrix between stim and male turning
            fid = mat[0, 1]  # Store correlation coefficient
        elif i <= hemi_length:  # At the very start of trial
            x = heading_diff[:i + 1]
            f = female_angle[:i + 1]
            t = behavior_time[:i + 1]
            start_times.append(behavior_time[i])
            fid = 0
        else:  # At the very end of trial
            x = heading_diff[i:]
            f = female_angle[i:len(behavior_time) - 1]
            t = behavior_time[i:len(behavior_time) - 1]
            start_times.append(behavior_time[i])
            fid = 0

        fidelity.append(fid)  # Correlation
        vigor.append(np.sum(x * np.sign(f)))  # Net turning in direction of target

    vigor = np.array(vigor)
    fidelity = np.array(fidelity)
    R = vigor / np.max(vigor)  # Normalize vigor across all periods
    R = R * fidelity  # Tracking index

    return R, vigor, fidelity


#%%

import xml.etree.ElementTree as ET

# Read the XML file
def xml_to_dict(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    return root

#%
def print_xml_structure(element, indent=0):
    print('  ' * indent + element.tag)
    for child in element:
        print_xml_structure(child, indent + 1)

def get_timestamps_from_xml(xml_fname, verbose=False):
    # Load and parse the XML
    # metaIm = xml_to_dict(xml_fname)
    metaIm = ET.parse(xml_fname).getroot()
    if verbose:
        print_xml_structure(metaIm)

    #%
    # Load and parse the XML file
    #tree = ET.parse(xml_fname)
    #root = tree.getroot()

    # Find the imaging start time in the XML structure
    sequence_element = metaIm.find(".//Sequence")
    if sequence_element is not None and 'time' in sequence_element.attrib:
        imStart_str = sequence_element.attrib['time'].split(':')
        imStart = [float(imStart_str[0]), float(imStart_str[1]), float(imStart_str[2])]
    else:
        raise ValueError("Could not find the 'time' attribute in the XML structure.")

    #%
    # Find start times of behavior and imaging
    #imStart_str = metaIm.find(".//PVScan/Sequence").attrib['time'].split(':')
    #imStart = [float(imStart_str[0]), float(imStart_str[1]), float(imStart_str[2])]

    # Get the number of frames and their absolute times
    #nIms = len(metaIm.findall(".//Sequence/Frame"))
    #absFrameTimes = []
    #for i in range(nIms):
    #    frame = metaIm.find(f".//Sequence/Frame[{i+1}]")
    #    absFrameTimes.append(float(frame.attrib['absoluteTime']))
    absFrameTimes = [float(i.attrib['absoluteTime']) for i in metaIm.findall(".//Sequence/Frame")]

    # Extract behavior start time and bTime
    # behStart = ft[['clock_H', 'clock_M', 'clock_S']].iloc[0] #expr[0, 7:10]
    return absFrameTimes


def get_pulse_frames(voltage_fname):
    
    from scipy.signal import find_peaks

    # Read voltage CSV file (skipping the first row and column)
    voltage = np.genfromtxt(voltage_fname, delimiter=',', skip_header=1, usecols=1)

    # Find voltage peaks for syncing
    peaks, _ = find_peaks(np.diff(voltage), height=1)
    id_vals = peaks + 1  # Adjust indices to match MATLAB's 1-based indexing
    pulseRCD = id_vals / 1000  # Convert to seconds

    return pulseRCD

def get_pulse_offset(pulseRCD, pulseSent):
    # Adjust pulse lengths if they don't match
    if len(pulseRCD) > len(pulseSent):
        pulseRCD = pulseRCD[:len(pulseSent)]
    elif len(pulseSent) > len(pulseRCD):
        pulseSent = pulseSent[:len(pulseRCD)]

    # Calculate the offset
    offset = pulseSent - pulseRCD
    mean_offset = np.mean(offset)

    return mean_offset

#%%

# Visualizing fov
def adjust_image_contrast(img, clip_limit=2.0, tile_size=10):#(10,10)):
    import cv2
    img[img<-50] = 0
    normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit
    img8 = cv2.convertScaleAbs(normed)

    # Equalize hist:
    tg = tile_size if isinstance(tile_size, tuple) else (tile_size, tile_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tg)
    eq = clahe.apply(img8)

    return eq