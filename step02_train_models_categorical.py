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
from helpers.utils import compute
import io
import itertools as it
import operator
import pickle
from IPython.core.display import display

import json
import random as r

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
from category_encoders import (BinaryEncoder, CatBoostEncoder, CountEncoder, HashingEncoder, LeaveOneOutEncoder,
                               OneHotEncoder, OrdinalEncoder)
from category_encoders.target_encoder import TargetEncoder
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler)
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
# visualize model structure
from IPython.display import SVG
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow.keras.preprocessing.sequence as kseq
import tensorflow.keras.preprocessing as kprep
import pydot
import keras
from helpers import constants as c
import time

keras.utils.vis_utils.pydot = pydot

# %%
# data = pd.read_csv("./bpic2012a.csv", sep=";")
all_datasets = pickle.load(open(c.file_all_datasets, "rb"))

# %%
for k, v in list(all_datasets.items())[:10]:
    print("===========================")
    print(k)
    display(v["data"])
# %%
# all_datasets = new_items
# %%
TEST_SIZE = 0.8

DEBUG = 0
if DEBUG:
    TEST_SIZE = 0.8

all_datasets_splits = {}
for key, data2process in all_datasets.items():
    X_data, Y_data = data2process["data"].drop("remtime", axis=1), data2process["data"]["remtime"],
    all_datasets_splits[key] = train_test_split(X_data, Y_data, test_size=TEST_SIZE, shuffle=True)

# %%
all_results = []
hparam_mapper = {}

final_pbar = tqdm(total=len(all_datasets))
final_pbar.refresh()
for dataset_name, data_set in all_datasets_splits.items():
    final_pbar.set_description("Running dataset all")
    X_train, X_test, y_train, y_test = data_set

    param_set_1 = np.arange(2, 15, 2)
    param_set_2 = np.linspace(100, 1000, 11, dtype=int)
    pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"DT - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train, y_train, DecisionTreeRegressor, max_depth=p1, max_leaf_nodes=p2)
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.extend(result.to_dict(orient="records"))
            pbar_sm.update(1)
        pbar_sm.refresh()

    final_pbar.update(1)
    final_pbar.refresh()

    current_results = pd.DataFrame(all_results)
    current_results.to_csv(io.open(c.file_all_results, 'w'), line_terminator='\n')
    json.dump(hparam_mapper, io.open(c.file_hparams_json, 'w'))
# %%
current_results
# %%
