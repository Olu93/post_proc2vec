# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Assignment 2, due 2021-03-12 at 17:00 CET
# %% [markdown]
# In this assignment, you will demonstrate your ability to use several regression models to predict the case remaining time. In addition, you will also show that you can evaluate their performance and discuss the results in a report.
#
# To be able to build models, you will employ the regression algorithms.
#
# %% [markdown]
# ## Part 1: Exploring the two data sets
#
#
#
# ### Data set 1: Sepsis
#
# Import the file *sepsis.csv* to load the Sepsis data set. This real-life event log contains events of sepsis cases from a hospital. Sepsis is a life threatening condition typically caused by an infection. One case represents the pathway through the hospital. The events were recorded by the ERP (Enterprise Resource Planning) system of the hospital. The original data set contains about 1000 cases with in total 15,000 events that were recorded for 16 different activities. Moreover, 39 data attributes are recorded, e.g., the group responsible for the activity, the results of tests and information from checklists.
#
# Additional information about the data can be found :
# https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639
#
# http://ceur-ws.org/Vol-1859/bpmds-08-paper.pdf
#
# ### Data set 2: Application of Financial Loan
#
# Import the file *bpic2012a.csv* to load the BPIC12 data set. This is a real-life event log taken from a Dutch Financial Institute. The event log records the cases of an application process for a personal loan or overdraft within a global financing organization.
#
# The original log contains some 262.200 events in 13.087 cases. The version used in this assignment is a preprocess event log. Apart from some anonymization, the log contains all data as it came from the financial institute.
#
# The amount requested by the customer is indicated in the case attribute AMOUNT_REQ, which is global, i.e. every case contains this attribute. The original event log is a merger of three intertwined sub processes. The event log used in this assignment is the sub process regarding Application.
#
# Additional information about the data can be found : https://www.win.tue.nl/bpi/doku.php?id=2012:challenge
#
#
#
#

# %%
import io
import itertools as it
import json
import operator
import pickle
import random as r

import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import scipy as sc
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing as kprep
import tensorflow.keras.preprocessing.sequence as kseq
import tensorflow.keras.utils as utils
from category_encoders import (BinaryEncoder, CatBoostEncoder, CountEncoder, HashingEncoder, LeaveOneOutEncoder,
                               OneHotEncoder, OrdinalEncoder)
from category_encoders.target_encoder import TargetEncoder
# visualize model structure
from IPython.display import SVG
from mpl_toolkits.mplot3d import \
    Axes3D  # <--- This is important for 3d plotting
from pandas.core.arrays.categorical import Categorical
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (LinearRegression, SGDRegressor, TweedieRegressor)
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization
from tqdm.notebook import tqdm

from helpers import constants as c

keras.utils.vis_utils.pydot = pydot

# %%
all_results = pd.read_csv(io.open(c.file_all_results, "rb"), index_col=0)
hparam_mapper = json.load(io.open(c.file_hparams_json, "r"))
hparams = "p1 p2 wsize esize bsize epochs".split()
cparams = "iscbow iseps isactonly".split()
significane_level = 0.05

formula_all = "test_R2 ~ p1 + wsize*esize*bsize + iscbow*iseps*isactonly"
formula_hparams = "test_R2 ~ p1 + wsize*esize*bsize"
formula_cparams = "test_R2 ~ p1 + iscbow*iseps*isactonly"
# random_effects_type = data_2vec_categorical["type"]
# random_effects_type =
all_results

# %%

results = all_results
group_indexer = ["model_name", "data_set"]
final_results = results.sort_values("test_R2", ascending=True)

best_models = final_results.groupby(group_indexer).apply(
    lambda df: df.drop(group_indexer, axis=1).tail(1)).reset_index()
best_models.sort_values("test_R2")

# %%

# %%
ds_tmp = results.copy()
parts = ds_tmp.data_set.str.split(c.SEP, expand=True)
parts.columns = "type wsize esize bsize epochs iscbow iseps isactonly loss val-loss".split()
ds_tmp = pd.merge(ds_tmp, parts, left_index=True, right_index=True).copy()

# %%
data_normal = ds_tmp[~ds_tmp.type.isin(["proc2vec", "act2vec"])].drop(ds_tmp.columns[7:], axis=1).drop("data_set",
                                                                                                       axis=1)
data_normal

# %%
data_2vec = ds_tmp.loc[ds_tmp.type.isin(["proc2vec", "act2vec"]), :].copy()
data_2vec.type = "[" + ds_tmp.type.astype(str) + "]"
data_2vec["test_R2_cat"] = pd.cut(data_2vec.test_R2, 10)
# data_2vec.wsize = data_2vec.wsize.str.split(c.CMB, expand=True)[1]
# data_2vec.esize = data_2vec.esize.str.split(c.CMB, expand=True)[1]
data_2vec.bsize = data_2vec.bsize.str.split(c.CMB, expand=True)[1]
# data_2vec.epochs = data_2vec.epochs.str.split(c.CMB, expand=True)[1]
data_2vec.iscbow = data_2vec.iscbow.str.split(c.CMB, expand=True)[2]
data_2vec.iseps = data_2vec.iseps.str.split(c.CMB, expand=True)[1]
data_2vec.isactonly = data_2vec.isactonly.str.split(c.CMB, expand=True)[1]
data_2vec.loss = data_2vec.loss.str.split(c.CMB, expand=True)[1].astype(float)
data_2vec["val-loss"] = data_2vec["val-loss"].str.split(c.CMB, expand=True)[1].astype(float)
data_2vec = data_2vec.drop("data_set", axis=1)
data_2vec["p1_square"] = np.square(data_2vec.p1)
data_2vec.p2 = MinMaxScaler().fit_transform(data_2vec.p2.values.reshape(-1, 1))
data_2vec

# %%
data_2vec_categorical = data_2vec.copy()
data_2vec_categorical["groups"] = data_2vec_categorical[parts.columns[:-2]].astype("str").sum(axis=1)
data_2vec_categorical["plot_groups"] = data_2vec_categorical[parts.columns[[0, 1, 2]]].astype("str").sum(axis=1)
data_2vec_categorical.to_csv(c.file_all_results_cat)
data_2vec_categorical
# %%
data_2vec_proc2vec = data_2vec_categorical[data_2vec_categorical.type == "[proc2vec]"].copy()
cmb_col = data_2vec_proc2vec.columns[[6, 7, 8]]
cmb_col_no_bsize = data_2vec_proc2vec.columns[[6, 7]]
cmb_col_w_p1 = data_2vec_proc2vec.columns[[1, 6, 7, 8]]
cmb_col_w_p1_n_bs = data_2vec_proc2vec.columns[[1, 6, 7]]
data_2vec_proc2vec["p2v_groups"] = data_2vec_proc2vec[cmb_col].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_no_bsize"] = data_2vec_proc2vec[cmb_col_no_bsize].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_with_p1"] = data_2vec_proc2vec[cmb_col_w_p1].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_with_p1_no_bsize"] = data_2vec_proc2vec[cmb_col_w_p1_n_bs].astype(str).sum(axis=1).copy()
data_2vec_proc2vec.iscbow = data_2vec_proc2vec.iscbow.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.iseps = data_2vec_proc2vec.iseps.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.isactonly = data_2vec_proc2vec.isactonly.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.to_csv(c.file_all_results_cat_p2v)
data_2vec_proc2vec
# %%
data_2vec_act2vec = data_2vec_categorical[data_2vec_categorical.type == "[act2vec]"].copy()
data_2vec_proc2vec.to_csv(c.file_all_results_cat_a2v)
data_2vec_act2vec
# %%
all_results_numerical = pd.read_csv(io.open(c.file_all_results_random, "rb"), index_col=0)
results = all_results_numerical
group_indexer = ["model_name", "data_set"]

ds_tmp = results.copy()
parts = ds_tmp.data_set.str.split(c.SEP, expand=True)
parts.columns = "type wsize esize bsize epochs iscbow iseps isactonly loss val-loss".split()
run_parts = parts.type.str.split(":", expand=True)
run_parts.columns = "run type".split()
ds_tmp = pd.merge(ds_tmp, parts.drop("type", axis=1), left_index=True, right_index=True).copy()
ds_tmp = pd.merge(ds_tmp, run_parts, left_index=True, right_index=True).copy()
ds_tmp
# %%

data_normal = ds_tmp[~ds_tmp.type.isin(["proc2vec", "act2vec"])].drop(ds_tmp.columns[7:], axis=1).drop("data_set",
                                                                                                       axis=1)
data_normal

# %%
data_2vec = ds_tmp.loc[ds_tmp.type.isin(["proc2vec", "act2vec"]), :].copy()
data_2vec.type = "[" + ds_tmp.type.astype(str) + "]"
data_2vec["test_R2_cat"] = pd.cut(data_2vec.test_R2, 10)
data_2vec.wsize = data_2vec.wsize.str.split(c.CMB, expand=True)[1]
data_2vec.esize = data_2vec.esize.str.split(c.CMB, expand=True)[1]
data_2vec.bsize = data_2vec.bsize.str.split(c.CMB, expand=True)[1]
# data_2vec.epochs = data_2vec.epochs.str.split(c.CMB, expand=True)[1]
data_2vec.iscbow = data_2vec.iscbow.str.split(c.CMB, expand=True)[2]
data_2vec.iseps = data_2vec.iseps.str.split(c.CMB, expand=True)[1]
data_2vec.isactonly = data_2vec.isactonly.str.split(c.CMB, expand=True)[1]
data_2vec.loss = data_2vec.loss.str.split(c.CMB, expand=True)[1].astype(float)
data_2vec["val-loss"] = data_2vec["val-loss"].str.split(c.CMB, expand=True)[1].astype(float)
data_2vec = data_2vec.drop("data_set", axis=1)
data_2vec["p1_square"] = np.square(data_2vec.p1)
data_2vec.p2 = MinMaxScaler().fit_transform(data_2vec.p2.values.reshape(-1, 1))
data_2vec
# %%
data_2vec_numerical = data_2vec.copy()
data_2vec_numerical["groups"] = data_2vec_numerical[parts.columns[:-2]].astype("str").sum(axis=1)
data_2vec_numerical["plot_groups"] = data_2vec_numerical[parts.columns[[0, 1, 2]]].astype("str").sum(axis=1)
data_2vec_numerical.to_csv(c.file_all_results_num)
data_2vec_numerical

# %%
data_2vec_proc2vec = data_2vec_numerical[data_2vec_numerical.type.isin(["[proc2vec]"])].copy()
cmb_col = data_2vec_proc2vec.columns[[6, 7, 8]]
cmb_col_no_bsize = data_2vec_proc2vec.columns[[6, 7]]
cmb_col_w_p1 = data_2vec_proc2vec.columns[[1, 6, 7, 8]]
cmb_col_w_p1_n_bs = data_2vec_proc2vec.columns[[1, 6, 7]]
data_2vec_proc2vec["p2v_groups"] = data_2vec_proc2vec[cmb_col].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_no_bsize"] = data_2vec_proc2vec[cmb_col_no_bsize].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_with_p1"] = data_2vec_proc2vec[cmb_col_w_p1].astype(str).sum(axis=1).copy()
data_2vec_proc2vec["p2v_groups_with_p1_no_bsize"] = data_2vec_proc2vec[cmb_col_w_p1_n_bs].astype(str).sum(axis=1).copy()
data_2vec_proc2vec.iscbow = data_2vec_proc2vec.iscbow.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.iseps = data_2vec_proc2vec.iseps.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.isactonly = data_2vec_proc2vec.isactonly.replace("False", False).replace("True", True).astype(bool)
data_2vec_proc2vec.to_csv(c.file_all_results_num_p2v)
data_2vec_proc2vec
# %%
data_2vec_act2vec = data_2vec_numerical[data_2vec_numerical.type.isin(["[act2vec]"])].copy()
data_2vec_proc2vec.to_csv(c.file_all_results_num_a2v)
data_2vec_act2vec