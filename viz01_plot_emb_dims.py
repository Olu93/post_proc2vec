# %%
from helpers.viz import clean_up, plot_embeddings, save_to_pdf
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
from IPython.display import display
# %%
all_datasets = pickle.load(open(c.file_all_datasets, "rb"))

for k, v in list(all_datasets.items())[:10]:
    print("===========================")
    print(k)
    display(v["data"])

# %%

for k, v in tqdm(list(all_datasets.items())):
    fig = plt.figure(figsize=(10, 10), facecolor="w")
    ax = plt.gca()
    print("===========================")
    ax = plot_embeddings(v["emb_dict"], 2, ax)
    if not ax:
        continue
    ax.set_xlabel(k)
    fig.tight_layout()
    fig.savefig(c.path_embeddings / f"tmp_{k}.png", transparent=False)
    fig.close()
    # plt.show()

# %%
all_images_embeddings = list(c.path_embeddings.glob('*.png'))
save_to_pdf(all_images_embeddings, c.file_embeddings_pdf)
clean_up(all_images_embeddings)
# %%
