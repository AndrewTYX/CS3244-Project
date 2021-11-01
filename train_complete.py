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

timestep = 2
ct_input_shape = (timestep,512, 512, 3, 1)
raw_input_shape = (timestep, 2)
base_input_shape = (timestep,4)

nn_feature_size = 10
ff_feature_size = 10
# Build ds for time series and ct scan

# Zip the ds with labels

# Concatenate raw input for timeseries and feature output layer from CNN
# Train the model

# Test the model
