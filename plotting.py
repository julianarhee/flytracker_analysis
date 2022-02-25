#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   plotting.py
@Time    :   2022/02/20 17:19:12
@Author  :   julianarhee 
@Contact :   juliana.rhee@gmail.com
'''

import pylab as pl
from matplotlib import rc

# ------------------------------------------------------------------------------
# General
# ------------------------------------------------------------------------------
def label_figure(fig, data_identifier):
    fig.text(0, 1,data_identifier, ha='left', va='top', fontsize=8)

    
def set_plot_params(light=False, default_sizes=True, lw_axes=0.25, axis_labelsize=7, tick_labelsize=6, color='k', dpi=100):
    if light:
        color=tuple([0.8]*3)
        if default_sizes:
            lw_axes=1.0
            axis_labelsize=12
            tick_labelsize=10
        dpi=300
    else:
        if default_sizes:
            lw_axes=0.25
            axis_labelsize=7
            tick_labelsize=6
        color='k'
        dpi=100
        
    #### Plot params
    pl.rcParams['font.size'] = tick_labelsize
    #pl.rcParams['text.usetex'] = True

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('text', usetex=True)

    pl.rcParams["axes.titlesize"] = axis_labelsize+2
  
    pl.rcParams["axes.labelsize"] = axis_labelsize
    pl.rcParams["axes.linewidth"] = lw_axes
    pl.rcParams["xtick.labelsize"] = tick_labelsize
    pl.rcParams["ytick.labelsize"] = tick_labelsize
    pl.rcParams['xtick.major.width'] = lw_axes
    pl.rcParams['xtick.minor.width'] = lw_axes
    pl.rcParams['ytick.major.width'] = lw_axes
    pl.rcParams['ytick.minor.width'] = lw_axes
    
    pl.rcParams['legend.fontsize'] = tick_labelsize
    pl.rcParams['legend.title_fontsize'] = axis_labelsize #labelsize+2
 
    pl.rcParams['figure.figsize'] = (5, 4)
    pl.rcParams['figure.dpi'] = dpi
    pl.rcParams['savefig.dpi'] = dpi
    pl.rcParams['svg.fonttype'] = 'none' #: path
            
    for param in ['xtick.color', 'ytick.color', 'axes.labelcolor', 'axes.edgecolor']:
        pl.rcParams[param] = color

# ------------------------------------------------------------------------------
# FlyTracker
# ------------------------------------------------------------------------------
def plot_wing_extensions(trk, start_frame=0, end_frame=None, ax=None, figsize=(20,3),
                         c1='lightblue', c2='steelblue', l1='var1', l2='var2', xaxis='sec'):
    if ax is None:
        fig, ax = pl.subplots(figsize=figsize)
    if end_frame is None:
        end_frame = int(trk.index.tolist()[-1])
    bout_dur_sec = (end_frame-start_frame)/fps
    df_ = trk.loc[start_frame:end_frame]
    ax.plot(df_[xaxis], np.rad2deg(df_['wing_r_ang']), color=c1, label=l1)
    ax.plot(df_[xaxis], np.rad2deg(df_['wing_l_ang']), color=c2, label=l2)

    return ax