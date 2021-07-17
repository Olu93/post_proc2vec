# %%
import io
import itertools as it
from helpers.utils import pp_multi_index, construct_emb_datasets, pp_cat_dummy_encoding, pp_cat_label_encoding, pp_cat_target_encoding, pp_num_minmax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
from tqdm.auto import tqdm
import pydot
import keras
from multiprocessing import Pool
from pathlib import Path
from helpers import constants as c
import random as r

data = pd.read_csv(c.file_initial_dataset, sep=";")
dataset_file = c.file_all_datasets_random

column_prefix_index = [c.column_CaseID, "idx"]
data_BPIC = data.replace("missing", np.nan).dropna()
data_BPIC[c.column_Timestamps] = pd.to_datetime(data_BPIC[c.column_Timestamps])
data_BPIC = data_BPIC.sort_values([c.column_CaseID, c.column_Timestamps])
data_BPIC["weekday"] = data_BPIC[c.column_Timestamps].dt.dayofweek
data_BPIC["hour"] = data_BPIC[c.column_Timestamps].dt.hour
data_BPIC["month"] = data_BPIC[c.column_Timestamps].dt.month
data_BPIC = data_BPIC.drop(c.column_Timestamps, axis=1)
data_BPIC.shape

all_results = []
hparam_mapper = {}

window_sizes = range(1, 4)
embed_sizes = range(2, 9)
batch_sizes = reversed([96, 128, 160, 192, 224, 256])
epochss = range(1, 10)
if False:
    window_sizes = [2]
    embed_sizes = [2, 3]
    batch_sizes = [128]
    epochss = [2]
data = [data_BPIC.copy()]
col_sets = [[c.column_CaseID, c.column_Activity, c.column_Remtime, c.all_time_columns, c.all_remaining_cols]]
all_param_configs = list(it.product(data, col_sets, window_sizes, embed_sizes, batch_sizes, epochss))
len(all_param_configs)
# %%
all_configs = r.choices(all_param_configs, k=100)
len(all_configs)
# %%
all_datasets = {}
for idx, params in tqdm(enumerate(all_configs)):
    print(params)
    partial = construct_emb_datasets(params)
    partial = {f"run{idx}:{key}": val for key, val in partial.items()}
    all_datasets.update(partial)
    pickle.dump(all_datasets, io.open(dataset_file, "wb"))

# %%
tmp_data = data_BPIC.copy()
tmp_data = pp_multi_index(tmp_data, c.column_CaseID)
tmp_data, _ = pp_cat_target_encoding(tmp_data, [c.column_Activity], c.column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, c.all_time_columns + ["Resource"], c.column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=c.all_remaining_cols + c.all_time_columns)
all_datasets[
    "normal_target_encoding--window_NA--emb_NA--batch_NA--num-epochs_NA--is_cbow_NA--seperated_NA--actonly_NA--loss_NA--val-loss_NA"] = {
        "data": tmp_data,
        "emb_dict": None,
    }

tmp_data = data_BPIC.copy()
tmp_data = pp_multi_index(tmp_data, c.column_CaseID)
tmp_data, _ = pp_cat_label_encoding(tmp_data, [c.column_Activity], c.column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, c.all_time_columns + ["Resource"], c.column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=c.all_remaining_cols + c.all_time_columns + [c.column_Activity])
all_datasets[
    "normal_label_encoding--window_NA--emb_NA--batch_NA--num-epochs_NA--is_cbow_NA--seperated_NA--actonly_NA--loss_NA--val-loss_NA"] = {
        "data": tmp_data,
        "emb_dict": None,
    }

tmp_data = data_BPIC.copy()
tmp_data = pp_multi_index(tmp_data, c.column_CaseID)
tmp_data, _ = pp_cat_dummy_encoding(tmp_data, [c.column_Activity], c.column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, c.all_time_columns + ["Resource"], c.column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=c.all_remaining_cols + c.all_time_columns)
all_datasets[
    "normal_dummy_encoding--window_NA--emb_NA--batch_NA--num-epochs_NA--is_cbow_NA--seperated_NA--actonly_NA--loss_NA--val-loss_NA"] = {
        "data": tmp_data,
        "emb_dict": None,
    }
pickle.dump(all_datasets, io.open(dataset_file, "wb"))

# %%
