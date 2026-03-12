#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File           : Untitled-1
Created        : 2025/04/21 19:32:49
Project        : <<projectpath>>
Author         : jyr
Email          : juliana.rhee@gmail.com
Last Modified  : 
'''
#%%
import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import libs.utils as util
import libs.plotting as putil

import importlib

# %%
# Set plotting
plot_style='dark'
putil.set_sns_style(plot_style, min_fontsize=18)
bg_color = [0.7]*3 if plot_style=='dark' else 'k'

#%%
assay = '38mm_dyad' 
experiment = 'MF'
#%
minerva_base = '/Volumes/Juliana'

# Specify local dirs
local_basedir = '/Users/julianarhee/Dropbox @RU Dropbox/Juliana Rhee/free_behavior' 
localdir = os.path.join(local_basedir, assay, experiment, 'FlyTracker')
out_fpath_local = os.path.join(localdir, 'relative_metrics.pkl')

#% Load aggregate relative metrics 
print("Loading processed data from:", out_fpath_local)
df0 = pd.read_pickle(out_fpath_local)
# Print summary of data
print(df0[['species', 'acquisition']].drop_duplicates().groupby('species').count())
# %%
# Pick 1
mel = df0[df0['species']=='Dmel']['acquisition'].unique()

curracq = mel[0]
currdf = df0[df0['acquisition']==curracq].copy()

# %%
xvars = ['body_area', 'targ_rel_pos_x', 'targ_rel_pos_y', 
         'targ_ang_size', 'theta_error', 'theta_error_dt', 
         'angle_between', 'facing_angle', 'dist_to_other', 
         'max_wing_ang', 'rel_vel', 'heading']
yvars = ['vel', 'ang_vel']

all_vars = xvars + yvars

df = currdf[all_vars].copy()
#%% Look at corrs
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), cbar=False, annot=True, fmt=".1f",
            cmap='viridis')

# %%
# Define the number of rows and columns you want
n_rows=3
n_cols=4

# Create the subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
fig.set_size_inches(10, 5)
for i, column in enumerate(df.iloc[:, :-2].columns):
    sns.histplot(df[column], ax=axes[i//n_cols, i % n_cols], kde=True)
plt.tight_layout()
#%%
# Create the subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
fig.set_size_inches(10, 5)
for i, column in enumerate(df.iloc[:,:-2].columns):
    sns.regplot(x = df[column], y = df[yvars[0]], ax=axes[i//n_cols,i%n_cols],
                scatter_kws={"color": bg_color, "s": 1, 'alpha': 0.5}, 
                line_kws={"color": "red"},
                ci=None, truncate=False)
plt.tight_layout()
# %%
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
fig.set_size_inches(10, 5)
for i, column in enumerate(df.iloc[:,:-2].columns):
    sns.regplot(x = df[column], y = df[yvars[1]], ax=axes[i//n_cols,i%n_cols],
                scatter_kws={"color": bg_color, "s": 1, 'alpha': 0.5}, 
                line_kws={"color": "red"},
                ci=None, truncate=False)
plt.tight_layout()
#%%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#%%
def print_predictions(y_true,y_pred, count):
  print("Predictions:")
  print(y_true.assign(
      Y1_pred = y_pred[:,0],
      Y2_pred = y_pred[:,1]   
  ).head(count).to_markdown(index = False))

def show_results(y_true, y_pred, count = 5):
  print("R2 score: ", r2_score(y_true,y_pred))
  print("Mean squared error: ", mean_squared_error(y_true,y_pred))
  print("Mean absolute error: ", mean_absolute_error(y_true,y_pred))
  print_predictions(y_true,y_pred, count)

#%%

# fill NaN in df by interpolating
df = df.interpolate(method='linear', limit_direction='both')

#%%
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:-2], df.iloc[:,-2:], test_size = 0.2, random_state = 42)
print(X_train.shape,X_test.shape)
print(y_train.shape, y_test.shape)
#%%

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X_train,y_train)
show_results(y_test,linear.predict(X_test))

#%%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

svm_multi = MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
svm_multi.fit(X_train,y_train)
show_results(y_test,svm_multi.predict(X_test))

# https://www.geeksforgeeks.org/multioutput-regression-in-machine-learning/
# %%
