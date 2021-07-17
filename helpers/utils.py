from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import (OneHotEncoder, OrdinalEncoder, TargetEncoder)
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, StandardScaler)
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.models as models
import tensorflow.keras.preprocessing.sequence as kseq
import tensorflow.keras.preprocessing as kprep
import itertools as it
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
from helpers import constants as c
from helpers.constants import SEP, CMB
import time


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


def pp_multi_index(data, column_CaseID):
    column_prefix_index = [column_CaseID, "idx"]
    data = data.groupby(column_CaseID).apply(create_group).set_index(column_prefix_index)
    return data


def p2v_seq_limiter(data, column_CaseID, window_size=2):
    fluff = pd.DataFrame(np.full((window_size * 2, data.shape[1]), np.nan), columns=data.columns)
    data = data.copy().groupby(column_CaseID).apply(lambda x: x.append(fluff, ignore_index=True))
    data = data.shift(window_size)
    return data


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
    Y = kseq.pad_sequences(Y_)
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


def encode_emb(data, embeddings, column_CaseID, column_Activity):
    data_new, data_activities = data.drop(column_Activity, axis=1), data[column_Activity]
    _, dim_emb = embeddings.shape
    col_embs = [f"dim_{column_Activity}_" + str(i) for i in range(dim_emb)]
    col_data = list(data_new.columns)
    embeddings = embeddings.loc[data_activities]

    df_data = pd.concat([data_new.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)
    df_data.columns = col_data + col_embs
    return df_data


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
    utils.plot_model(model,
                     to_file=c.path_models / f"IS_CBOW-{IS_CBOW}&IS_SEPERATED-{IS_SEPERATED}&IS_ACT_ONLY-{IS_ACT_ONLY}_model.png",
                     show_shapes=True,
                     rankdir='TB')

    loss_dict = model.fit([X, X_normed],
                          y_proc,
                          verbose=1,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=0.2,
                          shuffle=True)
    word_embed_layer = model.get_layer("Word_Embeddings")
    weights = word_embed_layer.get_weights()[0]
    embeddings = pd.DataFrame(weights[1:], index=le_name_mapping.keys())
    return model, loss_dict, embeddings


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


def construct_name(ws,
                   es,
                   bs,
                   epochs,
                   mtype,
                   is_cbow="NA",
                   is_seperated="NA",
                   is_activity="NA",
                   loss="NA",
                   val_loss="NA"):
    prefix = f"window{CMB}{ws}{SEP}emb{CMB}{es}{SEP}batch{CMB}{bs}{SEP}num-epochs{CMB}{epochs}"
    suffix = f"is{CMB}cbow{CMB}{is_cbow}{SEP}seperated{CMB}{is_seperated}{SEP}actonly{CMB}{is_activity}{SEP}loss{CMB}{loss}{SEP}val-loss{CMB}{val_loss}"
    new_key = f"{mtype}{SEP}{prefix}{SEP}{suffix}"
    print(new_key)
    return new_key


def construct_emb_datasets(params):
    partial_datasets = {}
    data, col_set, window_size, embed_size, batch_size, epochs = params
    column_CaseID, column_Activity, column_Remtime, all_time_columns, all_remaining_cols = col_set
    params = {
        "data": data.copy(),
        "column_CaseID": column_CaseID,
        "column_Activity": column_Activity,
        "window_size": window_size,
        "embed_size": embed_size,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    _, a2v_losses_cbow, a2v_emb_cbow = create_act2vec_embeddings_cbow(**params)
    _, a2v_losses_skip, a2v_emb_skip = create_act2vec_embeddings_skipgram(**params)
    all_models = create_all_proc2vec_embeddings(params)
    for key, val in all_models.items():
        first, second, third = key
        tmp_data = data.copy()
        tmp_data = encode_emb(tmp_data, val["emb"], column_CaseID, column_Activity)
        tmp_data = pp_multi_index(tmp_data, column_CaseID)
        tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
        tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
        partial_name = construct_name(
            window_size,
            embed_size,
            batch_size,
            epochs,
            "proc2vec",
            first,
            second,
            third,
            val["losses"].history["loss"][-1],
            val["losses"].history["val_loss"][-1],
        )
        partial_datasets[partial_name] = {"data": tmp_data, "emb_dict": val["emb"]}
    tmp_data = data.copy()
    tmp_data = encode_emb(tmp_data, a2v_emb_cbow, column_CaseID, column_Activity)
    tmp_data = pp_multi_index(tmp_data, column_CaseID)
    tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
    tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
    partial_name = construct_name(
        window_size,
        embed_size,
        batch_size,
        epochs,
        "act2vec",
        "True",
        None,
        None,
        a2v_losses_cbow.history["loss"][-1],
        a2v_losses_cbow.history["val_loss"][-1],
    )
    partial_datasets[partial_name] = {"data": tmp_data, "emb_dict": val["emb"]}
    tmp_data = data.copy()
    tmp_data = encode_emb(tmp_data, a2v_emb_skip, column_CaseID, column_Activity)
    tmp_data = pp_multi_index(tmp_data, column_CaseID)
    tmp_data, _ = pp_cat_target_encoding(tmp_data, all_time_columns + ["Resource"], column_Remtime)
    tmp_data, _ = pp_num_minmax(tmp_data, xcols=all_remaining_cols + all_time_columns)
    partial_name = construct_name(
        window_size,
        embed_size,
        batch_size,
        epochs,
        "act2vec",
        "False",
        None,
        None,
        a2v_losses_skip.history["loss"][-1],
        a2v_losses_skip.history["val_loss"][-1],
    )
    partial_datasets[partial_name] = {"data": tmp_data, "emb_dict": val["emb"]}
    return partial_datasets


def compute(X_train, y_train, Regressor, **kwargs):
    repeats = c.CV_REPEATS
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

        vals = (Regressor.__name__, ) + kw_vals[:2] + (train_MAE, test_MAE)
        keys = "model_name, p1, p2, train_R2, test_R2".split(", ")
        result = {key: val for key, val in zip(keys, vals)}
        collector.append(result)
    end_results = pd.DataFrame(collector)
    return end_results, {Regressor.__name__: list(kwargs.keys())[:2]}


def extract_results(results):
    col_model, param_1, param_2, train_metric_name, val_metric_name, dataset_name = list(
        results.groupby(["p1", "p2"]).mean().reset_index().columns)
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