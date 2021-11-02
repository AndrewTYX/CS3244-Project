# Tools import
import glob
import os
from typing import Tuple
from tensorboard.plugins.hparams.summary_v2 import hparams
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

# Hyper parameter tuning
from tensorboard.plugins.hparams import api as hp

#TODO: adapt to hyper parameter tuning
# timestep = 2
# channel_num = 3
# cnn_kernel_size = 3
# cnn_pool_size = 2
# cnn_drop_rate = 0.4
# lstm_unit = 50
# nn_feature_size = 10
# ff_feature_size = 10

# Setup hyperparameter
HP_TIME_STEP = hp.Hparam('timestep', hp.Discrete([2, 4, 6]))
HP_CHANNEL_NUM = hp.Hparam('channel_num', hp.Discrete([2, 5, 10]))
HP_KERNEL_SIZE = hp.HParam('cnn_kernel_size', hp.Discrete([3, 5, 9]))
HP_POOL_SIZE = hp.HParam('cnn_pool_size', hp.Discrete([2, 4, 6]))
HP_DROP_RATE = hp.HParam('cnn_drop_rate', hp.Discrete([0.2, 0.4, 0.6]))
HP_LSTM_UNIT = hp.HParam('lstm_unit', hp.Discrete([20, 40, 50]))
HP_NN_FEATURES_SIZE = hp.HParam('nn_feature_size', hp.Discrete([10, 20, 30]))
HP_FF_FEATURES_SIZE = hp.HParam('ff_feature_size', hp.Discrete([10, 20, 30]))

METRIC_LOSS = 'loss'



def generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num):
    return f'{nn_feature_size}_{ff_feature_size}_{timestep}_{channel_num}_model'


def train_test_model(hparams):
    timestep = hparams['timestep']
    channel_num = hparams['channel_num']
    cnn_kernel_size = hparams['cnn_kernel_size']
    cnn_pool_size = hparams['cnn_pool_size']
    cnn_drop_rate = hparams['cnn_drop_rate']
    lstm_unit = hparams['timestep']
    nn_feature_size = hparams['nn_feature_size']
    ff_feature_size = hparams['ff_feature_size']
    ct_input_shape = (timestep,512, 512, 3, 1)
    raw_input_shape = (timestep, 2)
    base_input_shape = (timestep,4)
    csv_file_path = './train.csv'
    ct_dir_path = './ct_interpolated_{channel_num}_dir.npy'

    if not os.path.exists(ct_dir_path):
        print('CT dictionary not exists, craeting one...')
        save_np_arr_with_channel(channel_num=channel_num)
        print('Create CT dictionary successfully!')

    train_ds, test_ds, val_ds = build_ds_with_split(csv_file_path, ct_dir_path, timestep)

    model = build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                        nn_feature_size, ff_feature_size, cnn_kernel_size, cnn_pool_size, cnn_drop_rate, lstm_unit)


    # Set up training
    model_name = generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num)

    model_path = './model_checkpoints/' + model_name
    csv_log_path = './train_logs/' + model_name

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError(),
                        tf.keras.metrics.MeanAbsoluteError()])
    
    logdir = './board_log'

    if not os.path.exits(logdir):
        print('Tensor board log dir not exist creating one...')
        os.mkdir('./board_log')
        print('Create tensorboard log dir successfully!')

    # Setup callbacks
    csv_logger_callback = CSVLogger(csv_log_path)
    checkpoint_callback = ModelCheckpoint(filepath = model_path, 
                                        monitor='loss',
                                        save_best_only=True)
    earlystop_callback = EarlyStopping(monitor='loss', min_delta=1, patience=200)
    board_metric_callback = tf.keras.callbacks.TensorBoard(logdir)
    hp_callback = hp.KerasCallback(logdir, hparams)

    model.fit(train_ds, validation_data = val_ds, epochs=1000, 
        callbacks=[csv_logger_callback, checkpoint_callback, earlystop_callback, board_metric_callback, hp_callback])