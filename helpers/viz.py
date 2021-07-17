import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
from helpers import constants as c
from pprint import pprint
import os


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
    if embs is None:
        return None

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


def process_image(filename):
    img = Image.open(filename)
    img_rgb = img.convert('RGB')
    return img_rgb


def save_to_pdf(image_glob, target):
    image_list = list(map(process_image, image_glob))
    assert len(image_glob) > 0, "No images to combine!"
    result = image_list[0].save(target, save_all=True, append_images=image_list[1:])
    return result


def clean_up(image_glob):
    pprint(image_glob)
    assert len(image_glob) > 0, "No images to clean!"
    return [os.remove(img) for img in image_glob]