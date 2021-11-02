# Tools import
import glob
import os
from typing import Tuple
import tensorflow as tf
import numpy as np
from itertools import product
from generate_ct_data import save_np_arr_with_channel

# Layers and Models import
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras import Model

# Call back import
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from build_model import build_full_lstm
from data_processing import build_ds_with_split

#TODO: adapt to hyper parameter tuning
timestep = 2
channel_num = 3
cnn_kernel_size = 3
cnn_pool_size = 2
cnn_drop_rate = 0.4
lstm_unit = 50


ct_input_shape = (timestep,512, 512, 3, 1)
raw_input_shape = (timestep, 2)
base_input_shape = (timestep,4)
csv_file_path = './train.csv'
ct_dir_path = './ct_interpolated_{channel_num}_dir.npy'

nn_feature_size = 10
ff_feature_size = 10

def generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num):
    return f'{nn_feature_size}_{ff_feature_size}_{timestep}_{channel_num}_model'

if not os.path.exists(ct_dir_path):
    print('CT dictionary not exists, craeting one...')
    save_np_arr_with_channel(channel_num=channel_num)

train_ds, test_ds, val_ds = build_ds_with_split(csv_file_path, ct_dir_path, timestep)

model = build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                    nn_feature_size, ff_feature_size)



# Set up training
model_name = generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num)

model_path = './model_checkpoints/' + model_name
csv_log_path = './train_logs/' + model_name

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError()])
    
# Setup callbacks
csv_logger_callback = CSVLogger(csv_log_path)
checkpoint_callback = ModelCheckpoint(filepath = model_path, 
                                    monitor='loss',
                                    save_best_only=True)
earlystop_callback = EarlyStopping(monitor='loss', min_delta=1, patience=200)

model.fit(train_ds, validation_data = val_ds, epochs=1000, callbacks=[csv_logger_callback, checkpoint_callback, earlystop_callback])
