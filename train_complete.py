# Tools import
import glob
import os
from typing import Tuple
import tensorflow as tf
import numpy as np
from itertools import product
from generate_data import save_np_arr_with_channel

# Layers and Models import
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras import Model

# Call back import
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from build_model import build_full_lstm
from data_processing import build_ds_with_split

timestep = 2
channel_num = 3
ct_input_shape = (timestep,512, 512, 3, 1)
raw_input_shape = (timestep, 2)
base_input_shape = (timestep,4)
csv_file_path = './train.csv'
ct_dir_path = './ct_interpolated_{channel_num}_dir.npy'

nn_feature_size = 10
ff_feature_size = 10

train_ds, test_ds, val_ds = build_ds_with_split(csv_file_path, ct_dir_path, timestep)

model = build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                    nn_feature_size, ff_feature_size)

