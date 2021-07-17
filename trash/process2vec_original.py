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
from utils import create_act2vec_embeddings_cbow, create_act2vec_embeddings_skipgram, create_proc2vec_embeddings, encode_emb, pp_cat_dummy_encoding, pp_cat_label_encoding, pp_cat_target_encoding, pp_multi_index, pp_num_minmax
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
data = pd.read_csv("./bpic2012a.csv", sep=";")
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
remtime = data_BPIC["remtime"]
categoricals = data_BPIC.drop([column_CaseID], axis=1).select_dtypes(include=['object'])
data_BPIC_categorical = categoricals.join(remtime)
data_BPIC_categorical = data_BPIC_categorical.rename(columns={0: "remtime"})

# %%

drop_cols = [column_CaseID, column_Remtime, column_Activity] + all_time_columns
num_plots = len(data_BPIC.columns) - len(drop_cols)

# %%
data_BPIC_prep = pd.DataFrame(data_BPIC)
data_BPIC_prep = data_BPIC_prep.join(pd.get_dummies(data_BPIC_prep[column_Activity])).drop(column_Activity, axis=1)
data_BPIC_prep = pd.DataFrame(MinMaxScaler().fit_transform(data_BPIC_prep),
                              columns=data_BPIC_prep.columns,
                              index=data_BPIC_prep.index)


# %%
# Standard Preprocessing
def create_group(df):
    df["idx"] = range(df.shape[0])
    return df


def pp_cat_target_encoding(data, xcols, ycol):
    X, y = data[xcols].astype('category'), data[ycol]
    encoder = TargetEncoder().fit(X, y)
    data[xcols] = encoder.transform(X)
    return data, encoder


def pp_cat_label_encoding(data, xcols, ycol):
    X, y = data[xcols].astype('category'), data[ycol]
    encoder = OrdinalEncoder().fit(X, y)
    data[xcols] = encoder.transform(X)
    return data, encoder


def pp_cat_dummy_encoding(data, xcols, ycol):
    data = data.copy()
    X, y = data[xcols].astype('category'), data[ycol]
    encoder = OneHotEncoder().fit(X, y)
    dummies = encoder.transform(X)
    data[dummies.columns] = dummies
    data = data.drop(xcols, axis=1)
    return data, encoder


def pp_num_standardization(data, xcols):
    X = data[xcols]
    encoder = StandardScaler().fit(X)
    data[xcols] = encoder.transform(X)
    return data, encoder


def pp_num_minmax(data, xcols):
    X = data[xcols]
    encoder = MinMaxScaler().fit(X)
    data[xcols] = encoder.transform(X)
    return data, encoder


def pp_multi_index(data):
    data = data.groupby(column_CaseID).apply(create_group).set_index(column_prefix_index)
    return data


def p2v_seq_limiter(data, window_size=2):
    fluff = pd.DataFrame(np.full((window_size * 2, data.shape[1]), np.nan), columns=data.columns)
    data = data.copy().groupby(column_CaseID).apply(lambda x: x.append(fluff, ignore_index=True))
    data = data.shift(window_size)
    return data


def p2v_rolling_batches(df, window_size=2, batch_size=None):
    num_rows, num_cols = df.shape
    pkg_size = (2 * window_size) + 1
    num_rows_post = num_rows - (pkg_size - 1)

    result = np.lib.stride_tricks.sliding_window_view(df, (pkg_size, num_cols))
    result_batched = result.reshape((-1, pkg_size, num_cols))
    if batch_size:
        num_full_pkgs = num_rows_post // batch_size
        data_limit = num_full_pkgs * batch_size
        result_batched = result[:data_limit].reshape((-1, batch_size, pkg_size, num_cols))

    return result_batched


def plot_model_losses(all_models):
    num_model_configs = len(all_models)
    fig, axes = plt.subplots(num_model_configs, 2, figsize=(30, 60))
    for axs, (cnf, mdl) in zip(axes, all_models.items()):
        ax, ax2 = axs
        hist = mdl["losses"].history
        df_loss = pd.DataFrame(zip(hist["loss"], hist["val_loss"]), columns=["train", "val"])
        ax.plot(df_loss)
        ax.set_title(str(cnf))
        ax2 = plot_embeddings(mdl["emb"], ax=ax2)


def plot_one_model_loss(mdl, losses, title=""):
    fig, ax = plt.subplots(figsize=(5, 5))
    hist = losses.history
    t_loss = hist["loss"]
    cols = ["train"]
    input_ = t_loss
    if "val_loss" in hist:
        v_loss = hist["val_loss"]
        cols.append(t_loss)
        input_ = zip(hist["loss"], v_loss)

    df_loss = pd.DataFrame(input_, columns=cols)
    ax.plot(df_loss)
    ax.set_title(title)


def plot_embeddings(embs, dim=2, ax=None):
    fig = plt.figure(figsize=(12, 12))

    if dim == 3 and embs.shape[1] >= 3:
        reduced = pd.DataFrame(PCA(dim).fit_transform(embs), index=embs.index)
        ax = fig.add_subplot(111, projection='3d')
        for i, (a, b, c) in reduced.iterrows():
            ax.scatter(a, b, c)
            ax.text(a, b, c, i)

    if dim == 2 and embs.shape[1] >= 2:
        reduced = pd.DataFrame(PCA(dim).fit_transform(embs), index=embs.index)
        ax = fig.add_subplot(111) if not ax else ax
        for i, (a, b) in reduced.iterrows():
            ax.scatter(a, b)
            ax.text(a, b, i, rotation=15)
    return ax


# %%
# Act2Vec
def encode_skipgram(data, column_CaseID, column_Activity, window_size):
    data = data.copy()
    single_token = data[column_Activity]
    encoding_selector = [column_CaseID, column_Activity]
    _ = data[encoding_selector].groupby(column_CaseID).apply(lambda df: df[column_Activity].values)
    le = LabelEncoder().fit(single_token)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data[column_Activity] = le.transform(data[column_Activity])
    vocab_size = len(le_name_mapping) + 1
    process_activities_encoded = data[encoding_selector].groupby(column_CaseID).apply(
        lambda df: df[column_Activity].values)
    corpus = process_activities_encoded.values
    skip_grams = [kseq.skipgrams(sent, vocabulary_size=vocab_size, window_size=window_size) for sent in corpus]
    X_, Y_ = zip(*skip_grams)
    X_w_, X_c_ = zip(*[list(zip(*x)) for x in X_])
    X_w = kseq.pad_sequences(X_w_)
    X_c = kseq.pad_sequences(X_c_)
    X = np.stack([X_w, X_c], axis=1)
    Y = kseq.pad_sequences(np.array(Y_))
    return le_name_mapping, vocab_size, X, Y


def encode_cbow(data, column_CaseID, column_Activity, window_size):
    data = data.copy()
    context_width = (window_size * 2) + 1
    single_token = data[column_Activity]
    encoding_selector = [column_CaseID, column_Activity]
    _ = data[encoding_selector].groupby(column_CaseID).apply(lambda df: df[column_Activity].values)
    le = LabelEncoder().fit(single_token)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data[column_Activity] = le.transform(data[column_Activity])
    vocab_size = len(le_name_mapping) + 1
    process_activities_encoded = data[encoding_selector].groupby(column_CaseID).apply(
        lambda df: df[column_Activity].values)
    corpus = process_activities_encoded.values
    corpus_padded = kseq.pad_sequences(corpus, value=-1) + 1
    tmp = np.lib.stride_tricks.sliding_window_view(corpus_padded, (1, context_width))
    tmp = np.squeeze(tmp)
    X_c_1, X_c_2, Y_ = tmp[:, :, :window_size], tmp[:, :, window_size + 1:], tmp[:, :, window_size]
    X, Y = np.dstack([X_c_1, X_c_2]), tf.keras.utils.to_categorical(Y_)
    return le_name_mapping, vocab_size, X, Y


def layers_skipgram(embed_size, vocab_size, X):
    _, num_pad = X[0].shape
    input_ = layers.Input(X.shape[1:])
    shared_embedding_layer = layers.Embedding(vocab_size, embed_size, name="Word_Embeddings")
    word_input, context_input = shared_embedding_layer(input_[:, 0]), shared_embedding_layer(input_[:, 1])
    comb_input = layers.Multiply()([word_input, context_input])
    comb_input = tf.reduce_sum(comb_input, axis=-1, name="Cosine_similarity")
    comb_input = layers.Dense(num_pad, activation="sigmoid")(comb_input)
    return input_, comb_input


def layers_cbow(embed_size, vocab_size, X):
    context_in = layers.Input(X.shape[1:])
    context_input = layers.Reshape((-1, X.shape[-1]))(context_in)
    context_input = layers.Embedding(vocab_size, embed_size, name="Word_Embeddings")(context_input)
    context_input = tf.reduce_mean(context_input, axis=-2)
    context_input = layers.Dense(vocab_size, activation='softmax', name="Predict_Target")(context_input)
    return context_in, context_input


def create_act2vec_embeddings_skipgram(
        data,
        column_CaseID,
        column_Activity,
        embed_size=4,
        window_size=2,
        epochs=5,
        batch_size=10,
):
    # https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html
    le_name_mapping, vocab_size, X, Y = encode_skipgram(data, column_CaseID, column_Activity, window_size)

    # build skip-gram architecture
    X_in, X_out = layers_skipgram(embed_size, vocab_size, X)
    model = models.Model(inputs=X_in, outputs=X_out, name="Skip-Gram")
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    l = model.fit(
        X,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )

    word_embed_layer = model.get_layer("Word_Embeddings")
    weights = word_embed_layer.get_weights()[0]

    embeddings = pd.DataFrame(weights[1:], index=le_name_mapping.keys())

    return model, l, embeddings


def create_act2vec_embeddings_cbow(
        data,
        column_CaseID,
        column_Activity,
        embed_size=4,
        window_size=2,
        epochs=5,
        batch_size=10,
):
    le_name_mapping, vocab_size, X, Y = encode_cbow(data, column_CaseID, column_Activity, window_size)
    X_in, X_out = layers_cbow(embed_size, vocab_size, X)
    model = models.Model(inputs=X_in, outputs=X_out, name="CBOW")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    l = model.fit(
        X,
        Y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    word_embed_layer = model.get_layer("Word_Embeddings")
    weights = word_embed_layer.get_weights()[0]
    embeddings = pd.DataFrame(weights[1:], index=le_name_mapping.keys())
    return model, l, embeddings


window_size = 3
embed_size = 2
batch_size = 32
epochs = 20
data = data_BPIC.copy()
params = {
    "data": data,
    "column_CaseID": column_CaseID,
    "column_Activity": column_Activity,
    "window_size": window_size,
    "embed_size": embed_size,
    "batch_size": batch_size,
    "epochs": epochs,
}
a2v_skipgram_model, a2v_hist, a2v_emb_cbow = create_act2vec_embeddings_cbow(**params)
print(a2v_emb_cbow)
plot_embeddings(a2v_emb_cbow)
plt.show()

a2v_skipgram_model, a2v_hist, a2v_emb_skip = create_act2vec_embeddings_skipgram(**params)
print(a2v_emb_skip)
plot_embeddings(a2v_emb_skip)
plt.show()

# plot_embeddings(a2v_emb_cbow, dim=3)
# plt.show()
# plot_embeddings(a2v_emb_skip, dim=3)
# plt.show()


# %%
def encode_emb(data, embeddings, column_CaseID, column_Activity):
    data_new, data_activities = data.drop(column_Activity, axis=1), data[column_Activity]
    _, dim_emb = embeddings.shape
    col_embs = [f"dim_{column_Activity}_" + str(i) for i in range(dim_emb)]
    col_data = list(data_new.columns)
    embeddings = embeddings.loc[data_activities]

    df_data = pd.concat([data_new.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    df_data.columns = col_data + col_embs
    return df_data


encode_emb(data, a2v_emb_cbow, column_CaseID, column_Activity)

# %%
from numpy.random import seed
num_seed = 35

# for num_seed in range(20, 100):
#     print("===========================")
#     print(num_seed)
#     print("===========================")
seed(num_seed)
tf.random.set_seed(num_seed + 1)

IS_CBOW = False
IS_SEPERATED = False
IS_ACT_ONLY = False


def create_proc2vec_embeddings(
        data,
        column_CaseID,
        column_Activity,
        embed_size=4,
        window_size=2,
        epochs=5,
        batch_size=10,
        ctx_emb_dim=2,
        act_emb_dim=2,
        IS_CBOW=False,
        IS_SEPERATED=False,
        IS_ACT_ONLY=False,
):
    data = data.copy()
    encoding_selector = [column_CaseID, column_Activity]
    if IS_CBOW:
        le_name_mapping, vocab_size, X, Y = encode_cbow(data, column_CaseID, column_Activity, window_size)
        X_in, X_out = layers_cbow(embed_size, vocab_size, X)
    if not IS_CBOW:
        le_name_mapping, vocab_size, X, Y = encode_skipgram(data, column_CaseID, column_Activity, window_size)
        X_in, X_out = layers_skipgram(embed_size, vocab_size, X)
    X_raw = data.groupby(column_CaseID).agg(["mean", "max", "min", "sum"], )
    X_normed = MinMaxScaler().fit_transform(X_raw)
    num_acts = len(le_name_mapping) + 1

    X_context_in = layers.Input(X_raw.shape[1:], name="CTX_INPUT")
    if IS_SEPERATED:
        # x_ctx = layers.RepeatVector(window_size * 2 if IS_CBOW else X_out.shape[1])(X_context_in)
        x_ctx = layers.RepeatVector(X_out.shape[1])(X_context_in)
        x_ctx = layers.Dense(ctx_emb_dim, activation='relu', name="ctx_Weight_Seperation")(x_ctx)
        x_act = layers.Dense(act_emb_dim, activation='relu', name="act_Weight_Seperation")(X_out)
        if not IS_CBOW:
            x_act = tf.expand_dims(X_out, -1)

        x_proc = layers.Concatenate(axis=-1, name="Embedding")([x_ctx, x_act])

    if not IS_SEPERATED:
        # x_ctx = layers.RepeatVector(window_size * 2 if IS_CBOW else X_out.shape[1])(X_context_in)
        x_ctx = layers.RepeatVector(X_out.shape[1])(X_context_in)
        x_act = X_out if IS_CBOW else tf.expand_dims(X_out, -1)
        x_proc = layers.Concatenate(axis=-1, name="Unified_weights")([x_ctx, x_act])
        x_proc = layers.Dense(ctx_emb_dim + act_emb_dim, name="Embedding")(x_proc)

    if IS_ACT_ONLY:
        fn_activation = "softmax" if IS_CBOW else "sigmoid"
        x_proc = layers.Dense(num_acts if IS_CBOW else 1, name="Predict", activation=fn_activation)(x_proc)
        y_proc = Y if IS_CBOW else np.expand_dims(Y, -1)
        loss_fn = "categorical_crossentropy" if IS_CBOW else "mean_squared_error"
    if not IS_ACT_ONLY:
        Y_raw_cbow = MinMaxScaler().fit_transform(
            data.groupby(column_CaseID).shift(-1).fillna(-1).drop(column_Activity, axis=1))
        Y_normed_cbow = np.repeat(np.expand_dims(Y_raw_cbow, 1), X_out.shape[1], axis=1) if IS_CBOW else np.repeat(
            np.expand_dims(Y_raw_cbow, 1), Y.shape[1], axis=1)
        y_proc = Y_normed_cbow if IS_CBOW else np.dstack(
            [np.repeat(np.expand_dims(X_normed, 1), Y.shape[1], axis=1),
             np.expand_dims(Y, -1)])
        out_dim = Y_raw_cbow.shape[-1] if IS_CBOW else X_normed.shape[-1] + 1
        x_proc = layers.Dense(out_dim, name="Predict", activation="sigmoid")(x_proc)
        loss_fn = "mean_squared_error"

    model = models.Model(inputs=[X_in, X_context_in], outputs=x_proc, name="cbow_model")
    model.compile(loss=loss_fn, optimizer='adam', run_eagerly=True)
    utils.plot_model(
        model,
        to_file=f"./model_imgs/IS_CBOW-{IS_CBOW}&IS_SEPERATED-{IS_SEPERATED}&IS_ACT_ONLY-{IS_ACT_ONLY}_model.png",
        show_shapes=True,
        rankdir='TB')

    l = model.fit([X, X_normed],
                  y_proc,
                  verbose=1,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  shuffle=True)
    word_embed_layer = model.get_layer("Word_Embeddings")
    weights = word_embed_layer.get_weights()[0]
    embeddings = pd.DataFrame(weights[1:], index=le_name_mapping.keys())
    return model, l, embeddings


params["epochs"] = 3
params["batch_size"] = 16
params["embed_size"] = 2
params["window_size"] = 1
p2v_skipgram_model, p2v_hist, p2v_emb_skip = create_proc2vec_embeddings(IS_CBOW=True,
                                                                        IS_SEPERATED=False,
                                                                        IS_ACT_ONLY=False,
                                                                        **params)

plot_embeddings(p2v_emb_skip, dim=2)
plt.show()
plot_embeddings(p2v_emb_skip, dim=3)
plt.show()


# %%
def create_all_proc2vec_embeddings(params):
    act_emb_dim = 4
    ctx_emb_dim = 3
    variants = list(it.product(*[[True, False]] * 3))
    results = {}
    for IS_CBOW, IS_SEPERATED, IS_ACT_ONLY in variants:
        print(f"{IS_CBOW}-{IS_SEPERATED}-{IS_ACT_ONLY}")
        model, loss, embeddings = create_proc2vec_embeddings(
            ctx_emb_dim=ctx_emb_dim,
            act_emb_dim=act_emb_dim,
            IS_CBOW=IS_CBOW,
            IS_SEPERATED=IS_SEPERATED,
            IS_ACT_ONLY=IS_ACT_ONLY,
            **params,
        )
        results[(IS_CBOW, IS_SEPERATED, IS_ACT_ONLY)] = {"model": model, "emb": embeddings, "losses": loss}
    return results


params = {
    "data": data,
    "column_CaseID": column_CaseID,
    "column_Activity": column_Activity,
    "window_size": 3,
    "embed_size": 2,
    "batch_size": 128,
    "epochs": 2,
}
all_models = create_all_proc2vec_embeddings(params)
plot_model_losses(all_models)
plt.show()

# %%
window_size = 3
embed_size = 2
batch_size = 32
epochs = 20
data = data_BPIC.copy()
params = {
    "data": data,
    "column_CaseID": column_CaseID,
    "column_Activity": column_Activity,
    "window_size": window_size,
    "embed_size": embed_size,
    "batch_size": batch_size,
    "epochs": epochs,
}
all_models = create_all_proc2vec_embeddings(params)
plot_model_losses(all_models)
plt.show()

a2v_cbow_model, a2v_hist_cbow, a2v_emb_cbow = create_act2vec_embeddings_cbow(**params)
plot_embeddings(a2v_emb_cbow)
plt.show()

a2v_skipgram_model, a2v_hist_skip, a2v_emb_skip = create_act2vec_embeddings_skipgram(**params)
plot_embeddings(a2v_emb_skip)
plt.show()

# %%
embeddings = all_models[(False, False, False)]["emb"]
data_proc2vec_w_embeddings = encode_emb(data, embeddings, column_CaseID, column_Activity)
data_proc2vec_w_embeddings

# %%
all_datasets = {}
for key, val in all_models.items():
    first, second, third = key
    tmp_data = data.copy()
    tmp_data = encode_emb(tmp_data, val["emb"], column_CaseID, column_Activity)
    tmp_data = pp_multi_index(tmp_data)
    tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
    tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
    all_datasets[f"proc2vec_{first}_{second}_{third}"] = tmp_data

tmp_data = data.copy()
tmp_data = encode_emb(tmp_data, a2v_emb_cbow, column_CaseID, column_Activity)
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["act2vec_cbow"] = tmp_data

tmp_data = data.copy()
tmp_data = encode_emb(tmp_data, a2v_emb_skip, column_CaseID, column_Activity)
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["act2vec_skipgram"] = tmp_data

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_target_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["normal_target_encoding"] = tmp_data

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_label_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns + [column_Activity])
all_datasets["normal_label_encoding"] = tmp_data

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_dummy_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["normal_dummy_encoding"] = tmp_data

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
    all_datasets_splits[key] = train_test_split(X_data, Y_data, test_size=TEST_SIZE, shuffle=True, random_state=42)

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
    end_results = pd.DataFrame(collector).groupby(keys[:2]).mean().reset_index().to_dict(orient="records")[0]
    # prep_kwarg_for_print = " - ".join(
    #     [f"{key} = {'{:.5f}'.format(val) if type(val) == float else val}" for key, val in kwargs.items()])
    # if DEBUG:
    #     performance_report_string = f"Train-MAE: {end_results['train_MAE']:.4f} - Val-MAE: {end_results['test_MAE']:.4f}"
    #     print(f"{time.time()-start:.2f}s - {prep_kwarg_for_print} - {performance_report_string}")

    return end_results, {Regressor.__name__: list(kwargs.keys())[:2]}


final_pbar = tqdm(total=len(all_datasets))
final_pbar.refresh()
for dataset_name, data_set in all_datasets_splits.items():
    final_pbar.set_description(f"Running dataset {dataset_name}")
    X_train, X_test, y_train, y_test = data_set
    pca = PCA(n_components=len(X_train.columns)).fit(X_train)
    enough_explained_variance = pca.explained_variance_ratio_.cumsum() < .95
    X_train_tmp = pd.DataFrame(pca.fit_transform(X_train)[:, enough_explained_variance])
    X_train_reduced, y_train_reduced = (X_train, y_train) if "bpic" not in dataset_name else (X_train_tmp, y_train)
    X_train_subset, y_train_subset = (X_train, y_train) if "bpic" not in dataset_name else (X_train, y_train)

    # param_set_1 = np.linspace(0.001, 1, 10)
    # param_set_2 = np.linspace(0.001, 0.99, 10)
    # pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"MLP - {dataset_name}")
    # for p1 in param_set_1:
    #     for p2 in param_set_2:
    #         result, hparams = compute(X_train,
    #                                   y_train,
    #                                   MLPRegressor,
    #                                   alpha=p1,
    #                                   learning_rate_init=p2,
    #                                   learning_rate="adaptive",
    #                                   hidden_layer_sizes=(3, ))
    #         result["data_set"] = dataset_name
    #         hparam_mapper.update(hparams)
    #         all_results.append(result)
    #         pbar_sm.update(1)
    #     pbar_sm.refresh()

    # param_set_1 = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    # param_set_2 = np.linspace(0.001, 1.1, 15)
    # pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"SVR - {dataset_name}")
    # for p1 in param_set_1:
    #     for p2 in param_set_2:
    #         result, hparams = compute(X_train_reduced, y_train_reduced, SVR, C=p1, gamma=p2, cache_size=512)
    #         result["data_set"] = dataset_name
    #         hparam_mapper.update(hparams)
    #         all_results.append(result)
    #         pbar_sm.update(1)
    #     pbar_sm.refresh()

    param_set_1 = np.arange(2, 15, 2)
    param_set_2 = np.linspace(2, 1000, 11, dtype=int)
    pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"DT - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train, y_train, DecisionTreeRegressor, max_depth=p1, max_leaf_nodes=p2)
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.append(result)
            pbar_sm.update(1)
        pbar_sm.refresh()

    # param_set_1 = np.arange(1, 13, 2)
    # param_set_2 = np.linspace(1, 3, 6)
    # pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"KNN - {dataset_name}")
    # for p1 in param_set_1:
    #     for p2 in param_set_2:
    #         result, hparams = compute(X_train_subset,
    #                                   y_train_subset,
    #                                   KNeighborsRegressor,
    #                                   n_neighbors=p1,
    #                                   p=p2,
    #                                   n_jobs=-1)
    #         result["data_set"] = dataset_name
    #         hparam_mapper.update(hparams)
    #         all_results.append(result)
    #         pbar_sm.update(1)
    #     pbar_sm.refresh()

    # param_set_1 = np.arange(2, 10, 1)
    # param_set_2 = np.linspace(0.001, 1, 10)
    # pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"XGBoost - {dataset_name}")
    # for p1 in param_set_1:
    #     for p2 in param_set_2:
    #         result, hparams = compute(X_train,
    #                                   y_train,
    #                                   GradientBoostingRegressor,
    #                                   max_depth=p1,
    #                                   learning_rate=p2,
    #                                   n_estimators=10)
    #         result["data_set"] = dataset_name
    #         hparam_mapper.update(hparams)
    #         all_results.append(result)
    #         pbar_sm.update(1)
    #     pbar_sm.refresh()

    # final_pbar.update(1)
    # final_pbar.refresh()


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
        # ax.set_ylim(0, 100)
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[train_metric_name], color="blue")
        ax.plot(subset_results_2D_1[param_1], subset_results_2D_1[val_metric_name], color="red")

        ax = axes[r, 1]
        ax.set_xlabel(param_2_name.replace("_", " ").title())
        ax.set_ylabel(val_metric_name.replace("_", " ").title())
        # ax.set_ylim(0, 100)
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
final_results = results.sort_values("test_R2", ascending=True)

best_models = final_results.groupby(group_indexer).apply(
    lambda df: df.drop(group_indexer, axis=1).tail(1)).reset_index()
best_models[best_models.model_name == "DecisionTreeRegressor"].sort_values("test_R2")

# %%
ds_tmp = best_models[best_models.model_name == "GradientBoostingRegressor"].groupby(group_indexer[1])
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
window_sizes = [1, 2, 3]
embed_sizes = [2, 3, 4]
batch_sizes = [16, 32, 64]
epochss = [1, 5, 10, 30]
data = data_BPIC.copy()
all_param_configs = list(it.product(window_sizes, embed_sizes, batch_sizes, epochss))
len(all_param_configs)
# %%

all_datasets = {}
col_set = [column_CaseID, column_Activity, column_Remtime, all_time_columns, all_remaining_cols]




from multiprocessing import Pool
if __name__ ==  '__main__': 
    num_processors = 3
    p=Pool(processes = num_processors)
    output = p.starmap()
    print(output)

for window_size, embed_size, batch_size, epochs in tqdm(all_param_configs):
    construct_emb_datasets(data, col_set, window_size, embed_size, batch_size, epochs)

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_target_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["normal_target_encoding"] = tmp_data

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_label_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns + [column_Activity])
all_datasets["normal_label_encoding"] = tmp_data

tmp_data = data.copy()
tmp_data = pp_multi_index(tmp_data)
tmp_data, _ = pp_cat_dummy_encoding(tmp_data, [column_Activity], column_Remtime)
tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
all_datasets["normal_dummy_encoding"] = tmp_data

# %%
NUM_REPEATS = 5

TEST_SIZE = 0.2

DEBUG = False
if DEBUG:
    NUM_REPEATS = 2
    TEST_SIZE = 0.8

all_datasets_splits = {}
for key, data2process in all_datasets.items():
    X_data, Y_data = data2process.drop("remtime", axis=1), data2process["remtime"],
    all_datasets_splits[key] = train_test_split(X_data, Y_data, test_size=TEST_SIZE, shuffle=True, random_state=42)

all_results = []
hparam_mapper = {}

final_pbar = tqdm(total=len(all_datasets))
final_pbar.refresh()
for dataset_name, data_set in all_datasets_splits.items():
    final_pbar.set_description(f"Running dataset {dataset_name}")
    X_train, X_test, y_train, y_test = data_set

    param_set_1 = np.arange(2, 15, 2)
    param_set_2 = np.linspace(2, 1000, 11, dtype=int)
    pbar_sm = tqdm(total=len(param_set_1) * len(param_set_2), desc=f"DT - {dataset_name}")
    for p1 in param_set_1:
        for p2 in param_set_2:
            result, hparams = compute(X_train, y_train, DecisionTreeRegressor, max_depth=p1, max_leaf_nodes=p2)
            result["data_set"] = dataset_name
            hparam_mapper.update(hparams)
            all_results.append(result)
            pbar_sm.update(1)
        pbar_sm.refresh()

current_results = pd.DataFrame(all_results)
current_results
# %%
extract = extract_results(current_results)
results, rest = extract[0], extract[1:]
plot_3D(*rest, hparam_mapper=hparam_mapper)
plt.savefig("./regression_results-5.png")
plt.show()
# %%
group_indexer = ["model_name", "data_set"]
final_results = results.sort_values("test_R2", ascending=True)

best_models = final_results.groupby(group_indexer).apply(
    lambda df: df.drop(group_indexer, axis=1).tail(1)).reset_index()
best_models[best_models.model_name == "DecisionTreeRegressor"].sort_values("test_R2")
# %%
