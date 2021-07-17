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
data_2vec_categorical = pd.read_csv(io.open(c.file_all_results_cat, "rb"), index_col=0)
data_2vec_cat_proc2vec = pd.read_csv(io.open(c.file_all_results_cat_p2v, "rb"), index_col=0)
data_2vec_cat_act2vec = pd.read_csv(io.open(c.file_all_results_cat_a2v, "rb"), index_col=0)
data_2vec_numerical = pd.read_csv(io.open(c.file_all_results_num, "rb"), index_col=0)
data_2vec_num_proc2vec = pd.read_csv(io.open(c.file_all_results_num_p2v, "rb"), index_col=0)
data_2vec_num_act2vec = pd.read_csv(io.open(c.file_all_results_num_a2v, "rb"), index_col=0)
significane_level = 0.05
# %%
drop_vars = ["train_R2", "test_R2", "loss", "p1_square"]
# target_var = "test_R2_cat"
target_var = "plot_groups"


def plot_numericals(data2vec, row, col, sample_size=1000):
    data2vec_numericals = data2vec
    data2vec_numericals = data2vec_numericals.sample(sample_size)
    sns.catplot(x="p1", y="test_R2", row=row, col=col, aspect=1, kind="box", data=data2vec_numericals)
    plt.show()


plot_numericals(data_2vec_cat_proc2vec, "epochs", "bsize", 20000)

# %%
plot_numericals(data_2vec_cat_proc2vec, "wsize", "bsize", 20000)

# %%
# Basic Test
md = smf.mixedlm(
    "test_R2 ~ p1 + p1_square + p2",
    data_2vec_categorical,
    groups=data_2vec_categorical.groups,
    # re_formula="~ p1 + p2",
)
mdf = md.fit()
print(mdf.summary())
# %%
# HParam interactions Test



# %%
# https://github.com/junpenglao/GLMM-in-Python/blob/master/GLMM_in_python.ipynb
def plot_dataset_by_group(data_2vec_categorical, grouper_str, figsize=(15, 15), title=""):
    plt_groups = list(data_2vec_categorical.groupby(grouper_str))
    fig, axes = plt.subplots(len(plt_groups), 2, figsize=figsize, sharey=True)
    for (idx, grp), (ax1, ax2) in zip(plt_groups, axes):
        sns.lineplot(x="p1", y="test_R2", hue="plot_groups", data=grp[grp.type == "[proc2vec]"], ax=ax1)
        ax1.set_xlabel(f" p2vec -- {grouper_str} = {idx}")
        sns.lineplot(x="p1", y="test_R2", hue="plot_groups", data=grp[grp.type == "[act2vec]"], ax=ax2)
        ax2.set_xlabel(f" a2vec -- {grouper_str} = {idx}")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# %%
def extract_restuls(significane_level, mdf):
    print("\nSignificant Parameters:\n")
    return pd.DataFrame([mdf.params[mdf.pvalues < significane_level], mdf.pvalues[mdf.pvalues < significane_level]],
                        index="coeff pval".split()).T


# %%
plot_dataset_by_group(data_2vec_categorical, "wsize")
# %%
plot_dataset_by_group(data_2vec_categorical, "esize")
# %%
plot_dataset_by_group(data_2vec_categorical, "bsize", (10, 20))
# %%
plot_dataset_by_group(data_2vec_categorical, "type", (10, 10))
# %%
# Proc2Vec interactions Test
md = smf.mixedlm(
    "test_R2 ~ wsize * esize * bsize",
    data_2vec_categorical,
    groups=data_2vec_categorical.groups,
    re_formula="~ p1 + p1_square + p2",
)
mdf = md.fit()
print(mdf.summary())

print(extract_restuls(significane_level, mdf))
# %%
# Proc2Vec interactions Test
md = smf.mixedlm("test_R2 ~ iscbow * isactonly * iseps",
                 data_2vec_cat_proc2vec,
                 re_formula="p1 + np.square(p1) + p2",
                 groups=data_2vec_cat_proc2vec.p2v_groups)
mdf = md.fit()
print(mdf.summary())
print(extract_restuls(significane_level, mdf))
# %%
plot_numericals(data_2vec_numerical, "type", "epochs", 20000)
# %%
tmp_data = data_2vec_numerical.groupby(["type","wsize", "esize"]).mean()[["train_R2", "test_R2"]].reset_index()
sns.lineplot(data=tmp_data, x="esize", y="test_R2", hue="type")
# %%
