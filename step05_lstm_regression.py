# %%
from keras.engine.input_layer import Input
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from helpers import constants as c
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
from pandas.core.arrays.categorical import Categorical
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
import pickle
from IPython.display import display
# %%
all_datasets = pickle.load(open(c.file_all_datasets, "rb"))
for k, v in list(all_datasets.items())[:10]:
    print("===========================")
    print(k)
    display(v["data"])

# %%
tmp = list(all_datasets.items())[0][1]['data']
tmp_reset = tmp.reset_index()
indices = tmp_reset[tmp_reset[c.column_CaseID] == 173703].index
# cnt = 0
# for idx, grp in tmp.groupby(c.column_CaseID):
#     cnt += 1
#     if cnt > 3:
#         break
#     tf.keras.preprocessing.timeseries_dataset_from_array(
#         grp,
#         targets,
#         sequence_length,
#     )

# tmp_seq = [sequence.pad_sequences(grp, maxlen=7) for idx, grp in tmp.groupby(c.column_CaseID)]
# tmp_seq

# %%
embedding_vector_length = 32
model = Sequential()
model.add(Embedding())
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)