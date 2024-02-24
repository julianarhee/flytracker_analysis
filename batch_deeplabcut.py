#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Feb 24 10:43:38 2024

@author: julianarhee
"""

import os
import glob
import re
import deeplabcut

def get_videos(folder, vid_type='.avi'):
    found_vidpaths = glob.glob(os.path.join(folder, '*{}'.format(vid_type)))
    return found_vidpaths

project = 'projector-1dot-jyr-2024-02-18'
rootdir = '/rugpfs/fs0/ruta_lab/scratch/jrhee/DeepLabCut'

project_path = os.path.join(rootdir, project)
config_path = os.path.join(project_path, 'config.yaml')

assay = 'projector-1dot'
data_basepath = os.path.join('/rugpfs/fs0/ruta_lab/store/jrhee/DeepLabCut', assay, 'videos') #data'
destfolder = os.path.join(os.path.split(data_basepath)[0], 'analyzed_videos', project)
if not os.path.exists(destfolder):
    os.makedirs(destfolder)

found_vidpaths = sorted(get_videos(data_basepath))

included_acqs = [('20240222', 'fly1'),
                 ('20240222', 'fly3'),
                 ('20240222', 'fly7'),
                 ('20240216', 'fly1'),
                 ('20240216', 'fly3'),
                 ('20240216', 'fly7'),
                 ('20240214', 'f1')]


#vids = ['20240222-1055_fly1_Dyak_sP1-ChR_2do_sh',
    #'20240222-1145_fly3_Dyak_sP1-ChR_2do_sh',
    #'20240222-1604_fly7_Dmel_sP1-ChR_2do_sh',

#    '20240216-1252_fly1_Dyak_sP1-ChR_2do_sh',
#    '20240216-1422_fly3_Dmel_sP1-ChR_2do_sh',
#    '20240216-1541_fly7_Dmel_sP1-ChR_2do_sh']


# NEW new
# ['20240212-1130_fly2_Dyak_sP1-ChR_3do_sh',
#  '20240212-1210_fly3_Dmel_sP1-ChR_3do_sh',
#  '20240212-1345_fly4_Dyak_sP1-ChR_3do_sh',
#  '20240212-1515_fly6_Dyak_sP1-ChR_3do_sh',
#  '20240215-1722_fly1_Dmel_sP1-ChR_3do_sh',
#  '20240215-1801_fly2_Dyak_sP1-ChR_3do_sh',
#  '20240216-1252_fly1_Dyak_sP1-ChR_2do_sh',
#  '20240222-1145_fly3_Dyak_sP1-ChR_2do_sh',
#  '20240222-1604_fly7_Dmel_sP1-ChR_2do_sh',
#  '20240224-1032_fly2_Dmel_sP1-ChR_2do_sh']


videos_to_analyze=[]
for v in included_acqs:
    pattern = "({date}-\d+_{fly})".format(date=v[0], fly=v[1])
    curr_vpath = [vp for vp in found_vidpaths if len(re.findall(pattern, vp))>0][0]
    videos_to_analyze.append(curr_vpath)

print("Found {} videos to analyze.".format(len(videos_to_analyze)))


deeplabcut.analyze_videos(config_path, videos_to_analyze, videotype='', trainingsetindex=0, destfolder=destfolder)
