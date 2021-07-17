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
from helpers.viz import clean_up, save_to_pdf
import io
import itertools as it
import operator
import pickle
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
from tqdm.notebook import tqdm
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
import json
from helpers import constants as c
keras.utils.vis_utils.pydot = pydot

# %%
# data = pd.read_csv("./bpic2012a.csv", sep=";")
all_results = pd.read_csv(io.open(c.file_all_results, "rb"), index_col=0)
hparam_mapper = json.load(io.open(c.file_hparams_json, "r"))
all_results

# %%
def extract_results(results):
    col_model, param_1, param_2, train_metric_name, val_metric_name, dataset_name = list(results.columns)
    results_2D_1 = results.drop(param_2, axis=1)
    results_2D_2 = results.drop(param_1, axis=1)
    results_2D_1 = results_2D_1.groupby([col_model, dataset_name, param_1]).mean().reset_index()
    results_2D_2 = results_2D_2.groupby([col_model, dataset_name, param_2]).mean().reset_index()
    results_3D = results.groupby([col_model, dataset_name, param_1, param_2]).mean().reset_index()
    return results, results_2D_1, results_2D_2, results_3D, train_metric_name, val_metric_name  #, best_test_MAPE, best_model


def plot_3D(results_2D_1, results_2D_2, results_3D_3, train_metric_name, val_metric_name, hparam_mapper={}):
    col_model, col_dset, param_1, param_2, train_metric_name, val_metric_name = list(results_3D_3.columns)
    configs = np.unique(list(results_3D_3[[col_model, col_dset]].to_records(index=False)))
    size = 20
    for (m_name, ds_name) in configs:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches((size, size / 4))
        fig.suptitle(f"Plot for model {m_name} on dataset {ds_name}")
        fig.set_facecolor('white')
        param_1_name, param_2_name = hparam_mapper.get(m_name)
        subset_results_2D_1 = results_2D_1.loc[results_2D_1[col_model] == m_name].loc[results_2D_1[col_dset] == ds_name]
        subset_results_2D_2 = results_2D_2.loc[results_2D_2[col_model] == m_name].loc[results_2D_2[col_dset] == ds_name]
        subset_results_3D_3 = results_3D_3.loc[results_3D_3[col_model] == m_name].loc[results_3D_3[col_dset] == ds_name]

        ax = ax1
        ax.set_xlabel(param_1_name.replace("_", " ").title())
        ax.set_ylabel(val_metric_name.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[train_metric_name], color="blue")
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[val_metric_name], color="red")

        ax = ax2
        ax.set_xlabel(param_2_name.replace("_", " ").title())
        ax.set_ylabel(val_metric_name.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.plot(subset_results_2D_2[param_2], subset_results_2D_2[train_metric_name], color="blue")
        ax.plot(subset_results_2D_2[param_2], subset_results_2D_2[val_metric_name], color="red")

        ax = fig.add_subplot(1, 4, 3, projection='3d')
        ax3.remove()
        ax3 = ax  # projection="3d"
        ax.set_xlabel(param_1_name.replace("_", " ").title())
        ax.set_ylabel(param_2_name.replace("_", " ").title())
        ax.view_init(25, 75)
        ax.plot_trisurf(subset_results_3D_3[param_1],
                        subset_results_3D_3[param_2],
                        subset_results_3D_3[train_metric_name],
                        cmap="Blues")
        ax.plot_trisurf(subset_results_3D_3[param_1],
                        subset_results_3D_3[param_2],
                        subset_results_3D_3[val_metric_name],
                        cmap="Reds")

        ax = ax4
        ax.set_xlabel(param_1_name.replace("_", " ").title())
        ax.set_ylabel(param_2_name.replace("_", " ").title())
        pivoted_data = pd.pivot_table(subset_results_3D_3, values=val_metric_name, index=param_1, columns=param_2)
        sns.heatmap(pivoted_data,
                    xticklabels=pivoted_data.columns.values.round(5),
                    yticklabels=pivoted_data.index.values.round(5),
                    cmap="RdYlGn",
                    cbar_kws=dict(orientation="horizontal"),
                    ax=ax)

        fig.tight_layout()
        fig.savefig(c.path_evaluations / f"{m_name}_{ds_name}.png", transparent=False)
        plt.show()
    return fig


extract = extract_results(all_results)
results, rest = extract[0], extract[1:]
plot_3D(*rest, hparam_mapper=hparam_mapper)
# plt.savefig("./regression_results-test.png")
# plt.show()

# %%
results
# %%
all_images_evaluations = list(c.path_evaluations.glob('*.png'))
save_to_pdf(all_images_evaluations, c.file_evaluations_pdf)
clean_up(all_images_evaluations)