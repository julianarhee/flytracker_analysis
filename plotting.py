#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : plotting.py
Created        : 2023/03/27 18:57:30
Project        : /Users/julianarhee/Repositories/plume-tracking
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
import numpy as np
import pandas as pd
import scipy.stats as spstats
import matplotlib as mpl
import pylab as pl
import seaborn as sns

import utils as util

## generic
# ----------------------------------------------------------------------
# Visualization 
# ----------------------------------------------------------------------
def label_figure(fig, fig_id, x=0.01, y=0.98):
    fig.text(x, y, fig_id, fontsize=8)

def set_sns_style(style='dark', min_fontsize=6):
    font_styles = {
                    'axes.labelsize': min_fontsize+1, # x and y labels
                    'axes.titlesize': min_fontsize+1, # axis title size
                    'figure.titlesize': min_fontsize+4,
                    'xtick.labelsize': min_fontsize, # fontsize of tick labels
                    'ytick.labelsize': min_fontsize,  
                    'legend.fontsize': min_fontsize,
                    'legend.title_fontsize': min_fontsize+1
        }
    for k, v in font_styles.items():
        pl.rcParams[k] = v

    pl.rcParams['axes.linewidth'] = 0.5

    if style=='dark':
        custom_style = {
                    'axes.labelcolor': 'white',
                    'axes.edgecolor': 'white',
                    'grid.color': 'gray',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'text.color': 'white',
                    'axes.facecolor': 'black',
                    'axes.grid': False,
                    'figure.facecolor': 'black'}
        custom_style.update(font_styles)

#        pl.rcParams['figure.facecolor'] = 'black'
#        pl.rcParams['axes.facecolor'] = 'black'
        sns.set_style("dark", rc=custom_style)
    elif style == 'white':
        custom_style = {
                    'axes.labelcolor': 'black',
                    'axes.edgecolor': 'black',
                    'grid.color': 'gray',
                    'xtick.color': 'black',
                    'ytick.color': 'black',
                    'text.color': 'black',
                    'axes.facecolor': 'white',
                    'axes.grid': False,
                    'figure.facecolor': 'white'}
        custom_style.update(font_styles)
        sns.set_style('white', rc=custom_style)

    pl.rcParams['savefig.dpi'] = 400
    pl.rcParams['figure.figsize'] = [6,4]

    pl.rcParams['svg.fonttype'] = 'none'


# ticks
def set_outward_spines(ax):
    ax.tick_params(which='both', axis='both', length=2, width=0.5, color='w',
               direction='out', left=True)

def remove_spines(ax, axes=['right', 'top']):
    for pos in axes:
       ax.spines[pos].set_visible(False)


def vertical_scalebar(ax, leg_xpos=0, leg_ypos=0, leg_scale=100, color='w', lw=1,fontsize=6):
    #leg_xpos=0; leg_ypos=round(df0.loc[odor_ix]['ft_posy']); leg_scale=100
    ax.plot([leg_xpos, leg_xpos], [leg_ypos, leg_ypos+leg_scale], color, lw=lw)
    
    ax.text(leg_xpos-5, leg_ypos+(leg_scale/2), '{} mm'.format(leg_scale), fontsize=fontsize, horizontalalignment='right')
    #ax.axis('off')


def custom_legend(labels, colors, use_line=True, lw=4, markersize=10):
    '''
    Returns legend handles

    Arguments:
        labels -- _description_
        colors -- _description_

    Keyword Arguments:
        use_line -- _description_ (default: {True})
        lw -- _description_ (default: {4})
        markersize -- _description_ (default: {10})

    Returns:
        _description_
    '''
    if use_line:
        legh = [mpl.lines.Line2D([0], [0], color=c, label=l, lw=lw) for c, l in zip(colors, labels)]
    else:
        legh = [mpl.lines.Line2D([0], [0], marker='o', color='w', label=l, lw=0,
                    markerfacecolor=c, markersize=markersize) for c, l in zip(colors, labels)]

    return legh
##

def plot_array_of_trajectories(trajdf, sorted_eff=[], nr=5, nc=7, 
                            aspect_ratio=0.5, sharey=True,
                            bool_colors=['r'], bool_vars=['instrip'], title='filename',
                            notable=[]):

    if len(sorted_eff)==0:
        sorted_eff = sorted(trajdf['filename'].unique(), key=util.natsort)


    maxy = trajdf['ft_posy'].max() if not sharey else 1600
     
    fig, axn = pl.subplots(nr, nc, figsize=(15,8), sharex=True)
    for fi, fn in enumerate(sorted_eff): #(fn, df_) in enumerate(etdf.groupby('filename')):
        if fi >= nr*nc:
            break
        ax=axn.flat[fi]
        df_ = trajdf[trajdf['filename']==fn].copy()
        #eff_ix = float(mean_tortdf[mean_tortdf['filename']==fn]['efficiency_ix'].unique())
        # PLOT
        plot_zeroed_trajectory(df_, ax=ax, traj_lw=1.5, odor_lw=1.0,
                                     strip_width=50, #params[fn]['strip_width'],
                                     strip_sep=1000, #) #params[fn]['strip_sep'])
                                bool_colors=bool_colors,
                                bool_vars=bool_vars)
        # legend
        ax.axis('off')
        if fi==0:
            leg_xpos=-150; leg_ypos=0; leg_scale=100
            vertical_scalebar(ax, leg_xpos=leg_xpos, leg_ypos=leg_ypos)
        #ax.set_box_aspect(3)
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, maxy])
        ax.axis('off')
        ax.set_aspect(aspect_ratio)
        if title=='filename':
            ax_title = fn
        else:
            ax_title = fn.split('_')[0] # use datetime str 
        if fn in notable:
            ax.set_title('{}:\n*{}'.format(fi, ax_title), fontsize=6, loc='left')
        else:
            ax.set_title('{}:\n{}'.format(fi, ax_title), fontsize=6, loc='left')

    for ax in axn.flat[fi:]:
        ax.axis('off')
        #ax.set_aspect(0.5)
        
    pl.tight_layout()
    pl.subplots_adjust(top=0.85, hspace=0.4, wspace=0.5) #left=0.1, right=0.9, wspace=1, hspace=1, bottom=0.1, top=0.8)
    return fig


# circular stuff

def add_colorwheel(fig, cmap='hsv', axes=[0.8, 0.8, 0.1, 0.1], 
                   theta_range=[-np.pi, np.pi], deg2plot=None):

    '''
    Assumes values go from 0-->180, -180-->0. (radians).
    ''' 
    display_axes = fig.add_axes(axes, projection='polar')
    #display_axes._direction = max(theta_range) #2*np.pi ## This is a nasty hack - using the hidden field to 
                                      ## multiply the values such that 1 become 2*pi
                                      ## this field is supposed to take values 1 or -1 only!!
    #norm = mpl.colors.Normalize(0.0, 2*np.pi)
    norm = mpl.colors.Normalize(theta_range[0], theta_range[1])

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(display_axes, cmap=mpl.cm.get_cmap(cmap, quant_steps),
                                       norm=norm, orientation='horizontal')
    # aesthetics - get rid of border and axis labels                                   
    cb.outline.set_visible(False)                                 
    #display_axes.set_axis_off()
    #display_axes.set_rlim([-1,1])
    if deg2plot is not None:
        display_axes.plot([0, deg2plot], [0, 1], 'k')
    
    display_axes.set_theta_zero_location('N')
    display_axes.set_theta_direction(-1)  # theta increasing clockwise

    return display_axes

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, 
                    edgecolor='w', facecolor=[0.7]*3, alpha=0.7, lw=0.5):
    """
    Produce a circular histogram of angles on ax.
    From: https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    # x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        #bins = np.linspace(-np.pi, np.pi, num=bins+1)
        bins = np.linspace(0, 2*np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, width=widths, #align='edge', 
                     edgecolor=edgecolor, fill=True, linewidth=lw, facecolor=facecolor,
                    alpha=alpha)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
        #ax.tick_params(which='both', axis='both', size=0)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # theta increasing clockwise
     
    return n, bins, patches



def colorbar_from_mappable(ax, norm, cmap, hue_title='', axes=[0.85, 0.3, 0.01, 0.4],
                            fontsize=7): #pad=0.05):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = ax.figure
    #ax.legend_.remove()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=pad)
    cax = fig.add_axes(axes) 
    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax) #ax=ax)
    cbar.ax.set_ylabel(hue_title, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    #pl.colorbar(im, cax=cax)

def plot_vector_path(ax, x, y, c, scale=1.5, width=0.005, headwidth=5, pivot='tail', 
                    colormap=mpl.cm.plasma, vmin=None, vmax=None, hue_title='',
                    axes=[0.8, 0.3, 0.01, 0.4]):
    if vmin is None:
        #vmin, vmax = b_[hue_param].min(), b_[hue_param].max()
        vmin, vmax = c.min(), c.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #colors=b_[hue_param]

#     uu = b_['ft_posx'].shift(periods=-1) - b_['ft_posx']
#     vv = b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
#     ax.quiver(b_['ft_posx'].values, b_['ft_posy'].values, uu, vv, color=colormap(norm(colors)), 
#               angles='xy', scale_units='xy', scale=1.5)
    uu = np.roll(x, -1) - x # b_['ft_posx']
    vv = np.roll(y, -1) - y #b_['ft_posy'].shift(periods=-1) - b_['ft_posy']
    uu[-1]=np.nan
    vv[-1]=np.nan
    ax.quiver(x, y, uu, vv, color=colormap(norm(c)), 
              angles='xy', scale_units='xy', scale=scale, pivot=pivot,
              width=width, headwidth=headwidth)
    colorbar_from_mappable(ax, norm, cmap=colormap, hue_title=hue_title, axes=axes)
    return ax

# DEEPLABCUT plotting
def plot_bodyparts(df, frame_ix, bodyparts2plot, ax=None, pcutoff=0.9, 
                    color='r', alpha=1.0, markersize=6):
    if ax is None:
        fig, ax = pl.subplots()
    for bpindex, bp in enumerate(bodyparts2plot):
        prob = df.xs(
            (bp, "likelihood"), level=(-2, -1), axis=1
        ).values.squeeze()
        mask = prob < pcutoff # confident predictions have likelihood > pcutoff
        temp_x = np.ma.array(
            df.xs((bp, "x"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        temp_y = np.ma.array(
            df.xs((bp, "y"), level=(-2, -1), axis=1).values.squeeze(),
            mask=mask)
        ax.plot(temp_x[frame_ix], temp_y[frame_ix], ".", color=color, alpha=alpha, markersize=markersize)


def plot_skeleton(coords, inds=None, ax=None, color='k', alpha=1, lw=1):
    if ax is None:
        fig, ax = pl.subplots()
    # abdomen lines
    segs = coords[tuple(zip(*tuple(inds))), :].swapaxes(0, 1) if inds else []
    coll = mpl.collections.LineCollection(segs, colors=color, alpha=alpha, lw=lw)
#         # og skeleton
#         segs0 = coords[tuple(zip(*tuple(inds_og))), :].swapaxes(0, 1) if inds_og else []
#         col0 = mpl.collections.LineCollection(segs0, colors=skeleton_color, alpha=alphavalue, lw=skeleton_lw)
    # plot
    ax.add_collection(coll)
#         ax.add_collection(col0)


def plot_ethogram(positions, ev, plot_behaviors=[], behavior_colors=[], nonbinary_behavs=[],
                bg_color=[0.7]*3, lw=0.05):
    import matplotlib.gridspec as gridspec

    if len(behavior_colors)==0:
        behavior_colors = sns.color_palette('cubehelix', 
                            n_colors=len(positions)+1, desat=1)[1:]
    if len(plot_behaviors)==0:
        plot_behaviors = np.arange(0, len(positions))
 
    fig = pl.figure(figsize=(8, len(nonbinary_behavs)*1.5)) #, constrained_layout=True)
    fig.patch.set_facecolor('k') #fig.patch.set_alpha(0)
    spec = gridspec.GridSpec(ncols=1, nrows=len(nonbinary_behavs)+2, figure=fig)
    ax0 = fig.add_subplot(spec[0:2, 0]) # video

    ax0.eventplot(positions, orientation='horizontal', 
                  colors=behavior_colors, linewidths=lw, lineoffsets=1.25)
    xlim = ax0.get_xlim()[-1]
    for ai, (label, color) in enumerate(zip(plot_behaviors, behavior_colors)):
        print(ai, label)
        ax0.text(xlim, ai*1.25, label, color=color, va='center')

    for ai, behav in enumerate(nonbinary_behavs):
        #print(ai, behav)
        ax_ = fig.add_subplot(spec[ai+2, 0]) #, sharex=ax0)  # 2p
        ax_.plot(ev['Time Vector (s)'], ev[behav], color=bg_color, lw=0.5)
        ax_.set_ylabel(behav, fontsize=8)
        if behav=='leg_dist':
            ax_.set_ylim([0, 18])
        elif behav=='vel':
            ax_.set_ylim([0, 75])
        elif 'angle' in behav:
            ax_.set_ylim([0, 3.2])
    for ai, ax in enumerate(fig.axes):
        if ai == 3:
            ax.set_xlabel('time (s)')
            ax.set_xticklabels(ax_.get_xticks()[0:])
        else: 
            ax.set_xlabel('')
            ax.set_xticklabels([])
    pl.subplots_adjust(left=0.1, right=0.8, top=0.9)
    return fig

