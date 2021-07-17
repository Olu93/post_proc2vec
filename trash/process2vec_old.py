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
import random as r

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
from category_encoders import (BinaryEncoder, CatBoostEncoder, CountEncoder, HashingEncoder, LeaveOneOutEncoder,
                               OneHotEncoder)
from category_encoders.target_encoder import TargetEncoder
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, TweedieRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (mean_absolute_error, mean_squared_error)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler, StandardScaler)
from sklearn.decomposition import PCA
from tqdm import tqdm, tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
# visualize model structure
from IPython.display import SVG
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
# %%
data = pd.read_csv("./bpic2012a.csv", sep=";")
column_CaseID = 'Case ID'
column_Activity = 'Activity'
column_Timestamps = 'Complete Timestamp'
column_label = 'remtime'
all_important_cols = [
    column_CaseID,
    column_Activity,
    column_Timestamps,
]
all_time_columns = ["month", "weekday", "hour"]
column_prefix_index = [column_CaseID, "idx"]
# %% [markdown]
#
# ### 1.1 Exploratory data analysis
#
# For each data set, create 2-3 figures and tables that help you understand the data
#
# **For both data sets, use the column "remtime" as the response variable for regression**
#
#
# #### Tips: ---------------
#
# Make sure to at least check the data type of each variable and to understand the distribution of each variable, especially the response variable.
#
# While exploring the data, you may also think about questions such as:
# - Should the variables be normalized or not?
# - Are the variables informative?
# - Are any of the potential predictor variables highly correlated?
# - Any relevant, useful preprocessing steps that may be taken?
# - Can you create figures or tables that answers these questions?
#
# Note that some of these variables are categorical variables. How would you preprocess these variables

# %%
data.describe()

# %%
# def preprocess_data(data,
#                     case_col="case:concept:name",
#                     time_stamp_col="time:timestamp",
#                     sort_col="@@index",
#                     drop_cols=["start_timestamp", "@@index"]):
#     data = data.sort_values(sort_col)
#     data[time_stamp_col] = pd.to_datetime(data[time_stamp_col])
#     data["day"] = data[time_stamp_col].dt.dayofweek
#     data["hour"] = data[time_stamp_col].dt.hour
#     data["month"] = data[time_stamp_col].dt.month
#     data = data.drop(time_stamp_col, axis=1)

#     data = data.replace("True", 1)
#     data = data.replace("False", 0)
#     data = data.loc[~(log_BPIC12[case_col] == "missing_caseid")]
#     data = data.replace("missing", np.nan).dropna()
#     data = data.drop(drop_cols, axis=1, errors="ignore")
#     return data

# log = preprocess_data(data_BPIC, "Case ID")

data_BPIC = data.replace("missing", np.nan).dropna()
data_BPIC[column_Timestamps] = pd.to_datetime(data_BPIC[column_Timestamps])
data_BPIC = data_BPIC.sort_values([column_CaseID, column_Timestamps])
data_BPIC["weekday"] = data_BPIC[column_Timestamps].dt.dayofweek
data_BPIC["hour"] = data_BPIC[column_Timestamps].dt.hour
data_BPIC["month"] = data_BPIC[column_Timestamps].dt.month
data_BPIC = data_BPIC.drop(column_Timestamps, axis=1)
data_BPIC.shape

# %%
fig = plt.figure(figsize=(15, 7))

ax = fig.add_subplot()
data_BPIC.drop([column_CaseID], axis=1).boxplot(ax=ax)
ax.set_title('')
ax.set_yscale('log')
fig.subplots_adjust(top=0.87)

fig.tight_layout()
plt.show()

# %%
# https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
fig, ax = plt.subplots(figsize=(20, 8))
data_BPIC.drop(["month", "weekday", "hour"], axis=1).hist(ax=ax, bins=17)
fig.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(20, 8))
data_BPIC[["month", "weekday", "hour"]].hist(ax=ax, bins=23)
fig.tight_layout()
plt.show()

# %%
remtime = data_BPIC["remtime"]
categoricals = data_BPIC.drop([column_CaseID], axis=1).select_dtypes(include=['object'])
data_BPIC_categorical = categoricals.join(remtime)
data_BPIC_categorical = data_BPIC_categorical.rename(columns={0: "remtime"})

ax = plt.subplot()
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
g = sns.barplot(x="Activity", y="remtime", data=data_BPIC, ax=ax)
g.set_xticklabels(g.get_xticklabels())

plt.tight_layout()
plt.show()

# %%

drop_cols = [column_CaseID, column_label, column_Activity] + all_time_columns
num_plots = len(data_BPIC.columns) - len(drop_cols)
fig, axes = plt.subplots(num_plots // 2, num_plots // 2, figsize=(15, 10))
# data_BPIC_melted = data_BPIC.drop([column_CaseID], axis=1).melt(id_vars=["remtime"], )
for idx, col in enumerate(data_BPIC.columns.difference(drop_cols)):
    ax = axes.flatten()[idx]
    im = ax.hist2d(data_BPIC[column_label], data_BPIC[col], bins=20, norm=mpl.colors.LogNorm())[3]
    ax.set_xlabel(col)
    ax.set_ylabel(column_label)
    # ax.set_yscale('log')
    # ax.set_xscale('log')

fig.tight_layout()
fig.colorbar(im, ax=axes)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(15, 10))
data_BPIC_prep = pd.DataFrame(data_BPIC)
data_BPIC_prep = data_BPIC_prep.join(pd.get_dummies(data_BPIC_prep[column_Activity])).drop(column_Activity, axis=1)
data_BPIC_prep = pd.DataFrame(MinMaxScaler().fit_transform(data_BPIC_prep),
                              columns=data_BPIC_prep.columns,
                              index=data_BPIC_prep.index)
sns.heatmap(data_BPIC_prep.drop(column_CaseID, axis=1).corr(), ax=ax, annot=True, cmap="RdBu")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Part 2: Preprocessing and Trace Encoding
#
#


# %%
# Standard Preprocessing
def preprocess_cat_target_encoding(data, xcols, ycol):
    X, y = data[xcols].astype('category'), data[ycol]
    encoder = TargetEncoder().fit(X, y)
    data[xcols] = encoder.transform(X)
    return data, encoder


def preprocess_cat_dummy_encoding(data, xcols, ycol):
    data = data.copy()
    X, y = data[xcols].astype('category'), data[ycol]
    encoder = OneHotEncoder().fit(X, y)
    dummies = encoder.transform(X)
    data[dummies.columns] = dummies
    data = data.drop(xcols, axis=1)
    return data, encoder


def preprocess_num_standardization(data, xcols):
    X = data[xcols]
    encoder = StandardScaler().fit(X)
    data[xcols] = encoder.transform(X)
    return data, encoder


def preprocess_num_minmax(data, xcols):
    X = data[xcols]
    encoder = MinMaxScaler().fit(X)
    data[xcols] = encoder.transform(X)
    return data, encoder


#%%

# Activity2Vec
data = data_BPIC.copy()
le = LabelEncoder().fit(data[column_Activity])
data[column_Activity] = le.fit_transform(data[column_Activity])
acitvity_encodings = data[[column_CaseID,
                           column_Activity]].groupby(column_CaseID).apply(lambda df: df[column_Activity].values)
pad = len(max(acitvity_encodings, key=len))
np_activity_encodings = np.array([list(i) + [0] * (pad - len(i)) for i in acitvity_encodings.values])

# # %%
# def f(d):
#     return pd.MultiIndex.from_product(d.index.levels, names=d.index.names)

# def g(d):
#     return d.reindex(f(d), fill_value=0)

# data_BPIC_act2vec.groupby(column_CaseID).head(8).set_index(column_CaseID, append=True).pipe(g).reset_index(column_CaseID)
# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(np_activity_encodings), 3))
input_array = np_activity_encodings
model.compile('rmsprop', 'mse')
embeddings = model.predict(input_array)
print(embeddings.shape)

# %%
# Act2Vec


def create_group(df):
    df["idx"] = range(df.shape[0])
    return df


def encode_act2vec(data, embeddings, column_CaseID, column_Activity):
    num_cases, num_events, dim_emb = embeddings.shape
    data_new = data.drop(column_Activity, axis=1).groupby(column_CaseID).apply(create_group)
    # print(data_new.shape)
    unique_cases = data_new.groupby(column_CaseID).count().index
    assert len(unique_cases) == len(embeddings)
    cases_padded = [(case, i) for case in unique_cases for i in range(num_events)]
    new_shape = (np.product(embeddings.shape[:2]), -1)
    embeddings_new = embeddings.reshape(new_shape)
    # print(len(cases_padded))
    # print(embeddings_new.shape)
    df_cases_padded = pd.DataFrame(cases_padded, columns=column_prefix_index)
    df_embeddings = pd.DataFrame(embeddings_new, columns=[f"dim_{column_Activity}_" + str(i) for i in range(dim_emb)])
    df_embedding = pd.concat([df_cases_padded, df_embeddings], axis=1)
    df_embedding = df_embedding.set_index(column_prefix_index)
    data_new = data_new.set_index(column_prefix_index)
    # print(df_embedding.head(10))
    data_BPIC_XXX_final = data_new.join(df_embedding)
    return data_BPIC_XXX_final


data_act2vec_encoded = encode_act2vec(data_BPIC, embeddings, column_CaseID, column_Activity)

# tf_input_layer = layers.Input((None, max_sent_len, len_vec))

# embedding_size = 3
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(input_dim=12, output_dim=embedding_size, input_length=1, name="embedding"))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(50, activation="relu"))
# model.add(tf.keras.layers.Dense(15, activation="relu"))
# model.add(tf.keras.layers.Dense(1))
# model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.fit(x=data_small_df['mnth'].as_matrix(), y=data_small_df['cnt_Scaled'].as_matrix(), epochs=50, batch_size=4)

# %%

data_act2vec, enc_target = preprocess_cat_target_encoding(data_act2vec_encoded, ["Resource"] + all_time_columns,
                                                          "remtime")
data_act2vec, enc_standardization = preprocess_num_standardization(data_act2vec, ["AMOUNT_REQ", "open_cases"])
data_act2vec, enc_minmax = preprocess_num_minmax(data_act2vec, ["elapsed", "Resource"] + all_time_columns)

data_normal, enc_target = preprocess_cat_dummy_encoding(data_BPIC, [column_Activity], "remtime")
data_normal, enc_target = preprocess_cat_target_encoding(data_normal, ["Resource"] + all_time_columns, "remtime")
data_normal, enc_minmax = preprocess_num_minmax(data_normal,
                                                ["elapsed", "Resource", "AMOUNT_REQ", "open_cases"] + all_time_columns)

data_proc2vec, enc_target = preprocess_cat_dummy_encoding(data_BPIC, [column_Activity], "remtime")
data_proc2vec, enc_target = preprocess_cat_target_encoding(data_proc2vec, ["Resource"] + all_time_columns, "remtime")
data_proc2vec, enc_minmax = preprocess_num_minmax(
    data_normal, ["elapsed", "Resource", "AMOUNT_REQ", "open_cases", "remtime"] + all_time_columns)
data_proc2vec = data_proc2vec.drop(column_label, axis=1)

# %%
# Proc2Vec

data = data_BPIC.copy()
max_sent_len = data_BPIC.groupby([column_CaseID])[column_Activity].count().max()
len_vec = data_BPIC.shape[1]


def sequence_generator_batch(data, column_CaseID, max_sent_len, window_size=2, batch_size=5):
    len_data = window_size * 2
    batch_X = np.zeros([batch_size, len_data, data.shape[1] - 1])
    batch_y = np.zeros([batch_size, data.shape[1] - 1])
    for idx, (df_id, group) in enumerate(data.groupby(column_CaseID)):
        for i in range(len(group)):
            if i > max_sent_len:
                break
            target = group.iloc[i, :]
            bckward_shift = pd.DataFrame([group.shift(sh).iloc[i, :] for sh in range(window_size, 0, -1)])
            forward_shift = pd.DataFrame([group.shift(-sh).iloc[i, :] for sh in range(1, window_size + 1)])

            X = pd.concat([bckward_shift, forward_shift]).drop(column_CaseID, axis=1).fillna(0)
            y = target.drop(column_CaseID)
            batch_X[i % len_data, :, :] = X.values
            batch_y[i % len_data, :] = y.values
            if ((i + 1) % len_data) == 0:
                result = (batch_X, batch_y)
                yield result
                batch_X[:, :, :] = 0
                batch_y[:, :] = 0


def sequence_generator(data, column_CaseID, max_sent_len, window_size=2):
    for idx, (df_id, group) in enumerate(data.groupby(column_CaseID)):
        for i in range(len(group)):
            if i > max_sent_len:
                break
            target = group.iloc[i, :]
            bckward_shift = pd.DataFrame([group.shift(sh).iloc[i, :] for sh in range(window_size, 0, -1)])
            forward_shift = pd.DataFrame([group.shift(-sh).iloc[i, :] for sh in range(1, window_size + 1)])

            X = np.vstack([bckward_shift, forward_shift])
            y = target.values
            yield X, y


generator = sequence_generator_batch(data_proc2vec, column_CaseID, max_sent_len)
print("======================")
batch = next(generator)
print(batch[0][0, :, 8:])
print("")
print(batch[0][1, :, 8:])
print("")
print(batch[0][2, :, 8:])
print("")
print(batch[0][3, :, 8:])
print("======================")
batch = next(generator)
print(batch[0][0, :, 8:])
print("")
print(batch[0][1, :, 8:])
print("")
print(batch[0][2, :, 8:])
print("")
print(batch[0][3, :, 8:])
print("======================")
batch = next(generator)
print(batch[0][0, :, 8:])
print("")
print(batch[0][1, :, 8:])
print("")
print(batch[0][2, :, 8:])
print("")
print(batch[0][3, :, 8:])
# print(batch[3][0])
# print(batch[4][0])

# %%
batch_size = 10
num_epochs = 2
window_size = 2
split_val = int(len(data_proc2vec) * .8)
data_train, data_val = data_proc2vec[:split_val], data_proc2vec[split_val:]
len_data = window_size * 2
len_data_dim = data_train.shape[1] - 1
steps_per_epoch = len(data_proc2vec) // batch_size
vec_dim = 5

inputs = layers.Input((len_data, len_data_dim))
x = layers.Flatten()(inputs)
# x = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs)
encoded = layers.Dense(vec_dim, activation='relu')(x)
decoded = layers.Dense(len_data_dim, activation='linear')(encoded)
cbow = models.Model(inputs=inputs, outputs=decoded, name="cbow_model")

encoded_input = layers.Input((vec_dim, ))
decoder_layer = cbow.layers[-1]

# encoder = models.Model(x, encoded)
# decoder = models.Model(encoded_input, decoder_layer(encoded_input))

loss = losses.MeanSquaredError()
cbow.compile(loss=loss, optimizer='adam', metrics=["mae"])

print(cbow.summary())
utils.plot_model(cbow, show_shapes=True, show_layer_names=False, rankdir='TB')
# %%
for e in range(num_epochs):
    generator_train = sequence_generator_batch(data_train, column_CaseID, max_sent_len, batch_size=batch_size)
    generator_val = sequence_generator_batch(data_val, column_CaseID, max_sent_len, batch_size=batch_size)
    cbow.fit(
        generator_train,
        steps_per_epoch=steps_per_epoch,
        # epochs=1,
        verbose=1,
        shuffle=True,
        validation_data=generator_val,
        validation_steps=len(data_val))
# %%

import time
NUM_REPEATS = 5

TEST_SIZE = 0.2

DEBUG = True
if DEBUG:
    NUM_REPEATS = 2
    TEST_SIZE = 0.8

data_act2vec_split = train_test_split(data_act2vec.drop("remtime", axis=1),
                                      data_act2vec["remtime"],
                                      test_size=TEST_SIZE,
                                      shuffle=True,
                                      random_state=42)

data_normal_split = train_test_split(data_normal.drop("remtime", axis=1),
                                     data_normal["remtime"],
                                     test_size=TEST_SIZE,
                                     shuffle=True,
                                     random_state=42)

# %%
all_datasets = [data_act2vec_split, data_normal_split]
all_results = []
dataset_names = "act2vec, normal".split(", ")
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
        train_MAE = mean_absolute_error(y, preds)
        test_MAE = mean_absolute_error(y_, preds_)

        vals = (Regressor.__name__, ) + kw_vals[:2] + (train_MAE, test_MAE, clf)
        keys = "model_name, p1, p2, train_MAE, test_MAE, clf".split(", ")
        result = {key: val for key, val in zip(keys, vals)}
        collector.append(result)
    end_results = pd.DataFrame(collector).groupby(keys[:2]).mean().reset_index().to_dict(orient="records")[0]
    prep_kwarg_for_print = " - ".join(
        [f"{key} = {'{:.5f}'.format(val) if type(val) == float else val}" for key, val in kwargs.items()])
    # if DEBUG:
    #     performance_report_string = f"Train-MAE: {end_results['train_MAE']:.4f} - Val-MAE: {end_results['test_MAE']:.4f}"
    #     print(f"{time.time()-start:.2f}s - {prep_kwarg_for_print} - {performance_report_string}")

    return end_results, {Regressor.__name__: list(kwargs.keys())[:2]}


final_pbar = tqdm_notebook(total=len(all_datasets))
final_pbar.refresh()
for dataset_name, data_set in zip(dataset_names, all_datasets):
    final_pbar.set_description(f"Running dataset {dataset_name}")
    X_train, X_test, y_train, y_test = data_set
    pca = PCA(n_components=len(X_train.columns)).fit(X_train)
    enough_explained_variance = pca.explained_variance_ratio_.cumsum() < .95
    X_train_tmp = pd.DataFrame(pca.fit_transform(X_train)[:, enough_explained_variance])
    X_train_reduced, y_train_reduced = (X_train, y_train) if "bpic" not in dataset_name else (X_train_tmp, y_train)
    X_train_subset, y_train_subset = (X_train, y_train) if "bpic" not in dataset_name else (X_train, y_train)

    param_set_1 = np.arange(2, 15, 2)
    param_set_2 = np.arange(2, 15, 1)
    pbar_sm = tqdm_notebook(total=len(param_set_1) * len(param_set_2), desc=f"DT - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train, y_train, DecisionTreeRegressor, max_depth=p1, max_leaf_nodes=p2)
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.append(result)
            pbar_sm.update(1)
        pbar_sm.refresh()

    param_set_1 = np.linspace(0.0001, 0.001, 9)
    param_set_2 = np.linspace(0.1, 0.001, 9)
    pbar_sm = tqdm_notebook(total=len(param_set_1) * len(param_set_2), desc=f"SGD - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train,
                                      y_train,
                                      SGDRegressor,
                                      eta0=p1,
                                      alpha=p2,
                                      loss="huber",
                                      learning_rate="adaptive")
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.append(result)
            pbar_sm.update(1)
        pbar_sm.refresh()

    param_set_1 = np.arange(2, 10, 1)
    param_set_2 = np.linspace(0.001, 1, 10)
    pbar_sm = tqdm_notebook(total=len(param_set_1) * len(param_set_2), desc=f"XGBoost - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train,
                                      y_train,
                                      GradientBoostingRegressor,
                                      max_depth=p1,
                                      learning_rate=p2,
                                      n_estimators=10)
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
    num_rows = len(configs)
    fig, axes = plt.subplots(num_rows, 4)
    fig.set_size_inches((25, 5 * num_rows))
    for r, (m_name, ds_name) in zip(range(num_rows), configs):
        param_1_name, param_2_name = hparam_mapper.get(m_name)
        subset_results_2D_1 = results_2D_1.loc[results_2D_1[col_model] == m_name].loc[results_2D_1[col_dset] == ds_name]
        subset_results_2D_2 = results_2D_2.loc[results_2D_2[col_model] == m_name].loc[results_2D_2[col_dset] == ds_name]
        subset_results_3D_3 = results_3D_3.loc[results_3D_3[col_model] == m_name].loc[results_3D_3[col_dset] == ds_name]

        ax = axes[r, 0]
        ax.set_title(f"Plot for model {m_name} on dataset {ds_name}")
        ax.set_xlabel(param_1_name.replace("_", " ").title())
        ax.set_ylabel(val_metric_name.replace("_", " ").title())
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[train_metric_name], color="blue")
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[val_metric_name], color="red")

        ax = axes[r, 1]
        ax.set_xlabel(param_2_name.replace("_", " ").title())
        ax.set_ylabel(val_metric_name.replace("_", " ").title())
        ax.plot(subset_results_2D_2[param_2], subset_results_2D_2[train_metric_name], color="blue")
        ax.plot(subset_results_2D_2[param_2], subset_results_2D_2[val_metric_name], color="red")

        ax = fig.add_subplot(num_rows, 4, (r * 4) + 3, projection='3d')
        axes[r, 2].remove()
        axes[r, 2] = ax  # projection="3d"
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

        ax = axes[r, 3]
        ax.set_xlabel(param_1_name.replace("_", " ").title())
        ax.set_ylabel(param_2_name.replace("_", " ").title())
        pivoted_data = pd.pivot_table(subset_results_3D_3, values=val_metric_name, index=param_1, columns=param_2)
        sns.heatmap(pivoted_data,
                    xticklabels=pivoted_data.columns.values.round(5),
                    yticklabels=pivoted_data.index.values.round(5),
                    cmap="RdYlGn",
                    cbar_kws=dict(orientation="horizontal"),
                    ax=ax)

    # plt.savefig(f"./{m_name}_{ds_name}.png")
    fig.tight_layout()
    return fig


current_results = pd.DataFrame(all_results)
extract = extract_results(current_results)
results, rest = extract[0], extract[1:]
plot_3D(*rest, hparam_mapper=hparam_mapper)
plt.savefig("./regression_results-5.png")
plt.show()
# %%
group_indexer = ["model_name", "data_set"]
final_results = results.sort_values("test_MAE", ascending=False)

best_models = final_results.groupby(group_indexer).apply(
    lambda df: df.drop(group_indexer, axis=1).head(1)).reset_index()
best_models.sort_values("test_MAE")

# %%
fig = plt.figure(figsize=(20, 15))
# best_models.plot.bar(ax=ax)

# plt.show()
for i, (index, group) in enumerate(best_models.groupby(group_indexer[1])):
    ax = fig.add_subplot(2, 3, i + 1)
    group.plot.bar(x="model_name", y=["test_MAE", "train_MAE"], logy=True, title=index, subplots=False, ax=ax)
fig.tight_layout()
plt.show()

# %%

#
# from keras.layers import Merge
# from keras.layers.core import Dense, Reshape
# from keras.layers.embeddings import Embedding
# from keras.models import Sequential

# # build skip-gram architecture
# word_model = Sequential()
# word_model.add(Embedding(vocab_size, embed_size,
#                          embeddings_initializer="glorot_uniform",
#                          input_length=1))
# word_model.add(Reshape((embed_size, )))

# context_model = Sequential()
# context_model.add(Embedding(vocab_size, embed_size,
#                   embeddings_initializer="glorot_uniform",
#                   input_length=1))
# context_model.add(Reshape((embed_size,)))

# model = Sequential()
# model.add(Merge([word_model, context_model], mode="dot"))
# model.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")

# # view model summary
# print(model.summary())

# # visualize model structure
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(model, show_shapes=True, show_layer_names=False,
#                  rankdir='TB').create(prog='dot', format='svg'))