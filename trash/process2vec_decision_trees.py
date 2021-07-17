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
import itertools as it
import operator
import pickle
from utils import construct_name, create_act2vec_embeddings_cbow, create_act2vec_embeddings_skipgram, create_proc2vec_embeddings, encode_emb, pp_cat_dummy_encoding, pp_cat_label_encoding, pp_cat_target_encoding, pp_multi_index, pp_num_minmax
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
keras.utils.vis_utils.pydot = pydot

# %%
# data = pd.read_csv("./bpic2012a.csv", sep=";")
all_datasets = pickle.load(open("all_datasets.pkl", "rb"))
column_CaseID = 'Case ID'
column_Activity = 'Activity'
column_Timestamps = 'Complete Timestamp'
column_Remtime = 'remtime'
all_important_cols = [
    column_CaseID,
    column_Activity,
    column_Timestamps,
]
all_time_columns = ["month", "weekday", "hour"]
all_remaining_cols = ["elapsed", "Resource", "AMOUNT_REQ", "open_cases"]

column_prefix_index = [column_CaseID, "idx"]

# %%
for k,v in list(all_datasets.items())[:10]:
    print("===========================")
    print(k)
    display(v)
# %%
# all_datasets = new_items
# %%
import time
NUM_REPEATS = 5

TEST_SIZE = 0.2

DEBUG = False
if DEBUG:
    NUM_REPEATS = 2
    TEST_SIZE = 0.8

all_datasets_splits = {}
for key, data2process in all_datasets.items():
    X_data, Y_data = data2process.drop("remtime", axis=1), data2process["remtime"],
    all_datasets_splits[key] = train_test_split(X_data, Y_data, test_size=TEST_SIZE, shuffle=True)

# %%
all_results = []
hparam_mapper = {}


def compute(X_train, y_train, Regressor, **kwargs):
    start = time.time()
    repeats = NUM_REPEATS
    k_fold = KFold(n_splits=repeats, shuffle=True)
    kw_vals = tuple(kwargs.values())
    collector = []
    for (train_indices, test_indices) in k_fold.split(X_train):
        x = np.array(X_train)[train_indices]
        y = np.array(y_train)[train_indices]
        x_ = np.array(X_train)[test_indices]
        y_ = np.array(y_train)[test_indices]
        clf = Regressor(**kwargs).fit(x, y)
        preds = clf.predict(x)
        preds_ = clf.predict(x_)
        train_MAE = r2_score(y, preds)
        test_MAE = r2_score(y_, preds_)

        vals = (Regressor.__name__, ) + kw_vals[:2] + (train_MAE, test_MAE, clf)
        keys = "model_name, p1, p2, train_R2, test_R2, clf".split(", ")
        result = {key: val for key, val in zip(keys, vals)}
        collector.append(result)
    end_results = pd.DataFrame(collector).to_dict(orient="records")[0]
    # prep_kwarg_for_print = " - ".join(
    #     [f"{key} = {'{:.5f}'.format(val) if type(val) == float else val}" for key, val in kwargs.items()])
    # if DEBUG:
    #     performance_report_string = f"Train-MAE: {end_results['train_MAE']:.4f} - Val-MAE: {end_results['test_MAE']:.4f}"
    #     print(f"{time.time()-start:.2f}s - {prep_kwarg_for_print} - {performance_report_string}")

    return end_results, {Regressor.__name__: list(kwargs.keys())[:2]}


final_pbar = tqdm(total=len(all_datasets))
final_pbar.refresh()
for dataset_name, data_set in all_datasets_splits.items():
    final_pbar.set_description(f"Running dataset all")
    X_train, X_test, y_train, y_test = data_set
    # pca = PCA(n_components=len(X_train.columns)).fit(X_train)
    # enough_explained_variance = pca.explained_variance_ratio_.cumsum() < .95
    # X_train_tmp = pd.DataFrame(pca.fit_transform(X_train)[:, enough_explained_variance])
    # X_train_reduced, y_train_reduced = (X_train, y_train) if "bpic" not in dataset_name else (X_train_tmp, y_train)
    # X_train_subset, y_train_subset = (X_train, y_train) if "bpic" not in dataset_name else (X_train, y_train)

    param_set_1 = np.arange(2, 15, 2)
    param_set_2 = np.linspace(50, 1000, 11, dtype=int)
    pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"DT - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train, y_train, DecisionTreeRegressor, max_depth=p1, max_leaf_nodes=p2)
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.append(result)
            pbar_sm.update(1)
        pbar_sm.refresh()

    final_pbar.update(1)
    final_pbar.refresh()

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
        fig.set_size_inches((size, size/4))
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

        ax = fig.add_subplot(1,4,3, projection='3d')
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
        fig.savefig(f"./figures/{m_name}_{ds_name}.png", transparent=False)
        plt.show()
    return fig


current_results = pd.DataFrame(all_results)
extract = extract_results(current_results)
results, rest = extract[0], extract[1:]
plot_3D(*rest, hparam_mapper=hparam_mapper)
# plt.savefig("./regression_results-test.png")
# plt.show()
# %%
group_indexer = ["model_name", "data_set"]
final_results = results.sort_values("test_R2", ascending=True)

best_models = final_results.groupby(group_indexer).apply(
    lambda df: df.drop(group_indexer, axis=1).tail(1)).reset_index()
best_models[best_models.model_name == "DecisionTreeRegressor"].sort_values("test_R2")

# %%
ds_tmp = best_models[best_models.model_name == "DecisionTreeRegressor"].groupby(group_indexer[1])
num_data_sets = len(ds_tmp)
num_cols = 2
num_rows = (num_data_sets // num_cols)
fig, axes = plt.subplots(num_rows if (num_data_sets % num_cols) == 0 else num_rows + 1, num_cols, sharey=True)
fig.set_size_inches((15, 3 * num_data_sets))

flat_axes = axes.flatten()
for i, (index, group) in enumerate(ds_tmp):
    ax = flat_axes[i]
    group.plot.bar(x="model_name", y=["test_MAPE", "train_MAPE"], title=index, sharey=True, subplots=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')

fig.tight_layout()
plt.show()


# %%
