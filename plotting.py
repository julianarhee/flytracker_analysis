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
