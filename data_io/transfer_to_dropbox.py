#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import glob
import shutil


#%%

dst_dir_base = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/data-science-center/data'


#%%
basedir = '/Volumes/Extreme Pro/DLC_models'
found_models = [f for f in os.listdir(basedir) if not f.startswith('.')]
found_models.append('38mm-dyad-jyr-2024-02-23')

for i, v in enumerate(found_models):
    print(i, v)


# Model types:
model_types = {
                '20mm-MMF-ventral-abdomen': ['MMF-fh-2024-01-15',
                                             'MMF-Rufei-2022-02-15'],
                '20mm-MF-ventral-abdomen':  ['MF-Rufei-2022-08-26'],
                '20mm-MMF-ventral-legs':    ['MMFv2-Rufei-2022-09-27'],
                '20mm-MF-dorsal-soundchamber':  ['SCF-Tom-2023-02-09'],
                '38mm-MF-dorsal-20mm':      ['38mm-dyad-jyr-2024-02-23']
            }

#%%
# -------------------------------------------------------------------- 
# Copy ALL labeled-data contents, preserve the project info under each category
# --------------------------------------------------------------------

for curr_model in found_models:
    print(curr_model)

    # add additional model:
    if curr_model in ['38mm-dyad-jyr-2024-02-23']:
        experimenter = curr_model.split('-')[2]
        basedir = '/Users/julianarhee/DeepLabCut'
    else:
        experimenter = curr_model.split('-')[1]
        basedir = '/Volumes/Extreme Pro/DLC_models'
    print(experimenter)

    #label_files = glob.glob(os.path.join(basedir, curr_model, 'labeled-data', '*', 
    #                                    'CollectedData_{}.csv'.format(experimenter)))
    #print(len(label_files))

    # get curr dst dir
    model_type_name = [k for k, v in model_types.items() if curr_model in v][0]
    dst_dir = os.path.join(dst_dir_base, model_type_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # save model name
    curr_dst_dir = os.path.join(dst_dir, curr_model)

    #label_data = os.listdir(os.path.join(basedir, curr_model, 'labeled-data'))

    label_data = os.path.join(basedir, curr_model, 'labeled-data')
    video_data = os.path.join(basedir, curr_model, 'videos')

    #for ld in label_data:
    #    if ld.startswith('.'):
    #        continue

    #existing_dirs = os.listdir(curr_dst_dir)
    #src_path = os.path.join(basedir, curr_model, 'labeled-data') #, ld)
    #dst_path = os.path.join(curr_dst_dir, ld) 
    dst_labels = os.path.join(curr_dst_dir, 'labeled-data')
    if not os.path.exists(dst_labels): #dst_path):
        os.makedirs(dst_labels) #dst_path)
    dst_videos = os.path.join(curr_dst_dir, 'videos')
    if not os.path.exists(dst_videos):
        os.makedirs(dst_videos)

    #shutil.copytree(src_path, dst_path, dirs_exist_ok=True ) #, curr_dst_dir)
    shutil.copytree(label_data, dst_labels, dirs_exist_ok=True ) #, curr_dst_dir)
    shutil.copytree(video_data, dst_videos, dirs_exist_ok=True ) #, curr_dst_dir)






   #%% only copy the .csv or .h5

    for lf in label_files:
        src_vid_dir, fname = os.path.split(lf)

        #fpath_suffix = lf.split('labeled-data/')[-1]
        src_vid_folder = os.path.split(src_vid_dir)[-1]
        curr_dst_dir = os.path.join(dst_dir, src_vid_folder)
        dst_fpath = os.path.join(dst_dir, src_vid_folder, fname)

        # Create video subdir if needed
        #curr_dst_dir = os.path.split(dst_fpath)[0]
        if not os.path.exists(curr_dst_dir):
            os.makedirs(curr_dst_dir)

        existing_fp = glob.glob(os.path.join(curr_dst_dir, '*.csv'))
        if len(existing_fp)>0:
            if existing_fp[0] != dst_fpath:
                print('skipping: {}'.format(dst_fpath))
                continue
        # copy files
        shutil.copy(lf, dst_fpath)

    #%
        # copy labeled imgs if exist
        dst_labeled_images_dir = os.path.join(curr_dst_dir, 'labeled')
        if not os.path.exists(dst_labeled_images_dir):
            os.makedirs(dst_labeled_images_dir)

        labeled_images_dir = '{}_labeled'.format(src_vid_dir)
        labeled_images = glob.glob(os.path.join(labeled_images_dir, '*.png'))
        for im in labeled_images:
            im_fname = os.path.split(im)[1]
            shutil.copy(im, os.path.join(dst_labeled_images_dir, im_fname))




#%%

label_files = glob.glob(os.path.join(basedir, '*', 'labeled-data', '*', 'CollectedData*.h5'))

for i, lf in enumerate(label_files):

    curr_model = os.path.split(lf.split('/labeled-data')[0])[1]

    video_dir, fname = os.path.split(lf)
    video_name = os.path.split(video_dir)[1]    
    #print(video_name)

    # get curr dst dir
    model_type_name = [k for k, v in model_types.items() if curr_model in v][0]
    print('{}: {}'.format(model_type_name, i))
    dst_dir = os.path.join(dst_dir_base, model_type_name)

    # copy all subdirectories within labeled-data dir to dest dir:
    src_dir = os.path.join(video_dir, 'labeled-data')

    shutil.copytree(src_dir, dst_dir)

#%%
    curr_dst_dir = os.path.join(dst_dir, video_name)

    # copy label file
    new_fpath = os.path.join(curr_dst_dir, fname)
    #shutil.copy(lf, new_fpath)

    # copy labeled imgs if exist
    dst_labeled_images_dir = os.path.join(curr_dst_dir, 'labeled')

    labeled_images_dir = '{}_labeled'.format(video_dir)
    labeled_images = glob.glob(os.path.join(labeled_images_dir, '*_individual.png'))
    for im in labeled_images:
        im_fname = os.path.split(im)[1]
        print(im_fname)
        #shutil.copy(im, os.path.join(dst_labeled_images_dir, im_fname))


# %%
# %%

