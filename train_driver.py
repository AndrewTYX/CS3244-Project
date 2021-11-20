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
HP_TIME_STEP = hp.HParam('timestep', hp.Discrete([2, 3, 4]))
HP_CHANNEL_NUM = hp.HParam('channel_num', hp.Discrete([3, 6, 10]))
HP_KERNEL_SIZE = hp.HParam('cnn_kernel_size', hp.Discrete([3]))
HP_POOL_SIZE = hp.HParam('cnn_pool_size', hp.Discrete([2, 4]))
HP_DROP_RATE = hp.HParam('cnn_drop_rate', hp.Discrete([0.2, 0.4, 0.6]))
HP_LSTM_UNIT = hp.HParam('lstm_unit', hp.Discrete([50, 100]))
HP_NN_FEATURES_SIZE = hp.HParam('nn_feature_size', hp.Discrete([10, 20, 30]))
HP_FF_FEATURES_SIZE = hp.HParam('ff_feature_size', hp.Discrete([5, 10, 15]))

METRIC_LOSS = 'loss'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_TIME_STEP, HP_CHANNEL_NUM, HP_KERNEL_SIZE, HP_POOL_SIZE, HP_DROP_RATE, 
    HP_LSTM_UNIT, HP_NN_FEATURES_SIZE, HP_FF_FEATURES_SIZE],
    metrics=[hp.Metric(METRIC_LOSS, display_name='loss')],
  )

def generate_hparams_list():
    '''
    Generate the hparams dictionary list for training sessions
    '''
    res = []
    for ele in product(HP_TIME_STEP.domain.values, HP_CHANNEL_NUM.domain.values,
        HP_KERNEL_SIZE.domain.values, HP_POOL_SIZE.domain.values, HP_DROP_RATE.domain.values, 
        HP_LSTM_UNIT.domain.values, HP_NN_FEATURES_SIZE.domain.values, HP_FF_FEATURES_SIZE.domain.values):
        
        hparams = {
            HP_TIME_STEP: ele[0],
            HP_CHANNEL_NUM: ele[1],
            HP_KERNEL_SIZE:ele[2],
            HP_POOL_SIZE:ele[3],
            HP_DROP_RATE:ele[4],
            HP_LSTM_UNIT:ele[5],
            HP_NN_FEATURES_SIZE:ele[6],
            HP_FF_FEATURES_SIZE:ele[7]
        }

        res.append(hparams)
    return res

def generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num):
    return f'{nn_feature_size}_{ff_feature_size}_{timestep}_{channel_num}_model'

def train_test_model(hparams):
    timestep = hparams[HP_TIME_STEP]
    channel_num = hparams[HP_CHANNEL_NUM]
    cnn_kernel_size = hparams[HP_KERNEL_SIZE]
    cnn_pool_size = hparams[HP_POOL_SIZE]
    cnn_drop_rate = hparams[HP_DROP_RATE]
    lstm_unit = hparams[HP_LSTM_UNIT]
    nn_feature_size = hparams[HP_NN_FEATURES_SIZE]
    ff_feature_size = hparams[HP_FF_FEATURES_SIZE]
    ct_input_shape = (512, 512, channel_num, 1)
    raw_input_shape = (timestep, 2)
    base_input_shape = (3,)
    csv_file_path = './train.csv'
    ct_dir_path = f'./ct_interpolated_{channel_num}_dir.npy'
    ds_buffer = './ds_buffer'

    if not os.path.exists(ct_dir_path):
        print('CT dictionary not exists, craeting one...')
        save_np_arr_with_channel(channel_num=channel_num)
        print('Create CT dictionary successfully!')

    model = build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                        nn_feature_size, ff_feature_size, cnn_kernel_size, cnn_pool_size, cnn_drop_rate, lstm_unit)
    model.summary()

    # Set up training
    print('[Driver] Generating the model..')
    model_name = generate_model_name(nn_feature_size, ff_feature_size, timestep, channel_num)
    print('[Driver] Generated model structure successfully!')
    
    train_ds, test_ds, val_ds = build_ds_with_split(csv_file_path, ct_dir_path, timestep)
    print('[Driver] Generated datasets successfully!')
    model_path = './model_checkpoints/' + model_name
    csv_log_path = './train_logs/' + model_name + ".csv"

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.MeanSquaredError(),
                        tf.keras.metrics.MeanAbsoluteError()])
    
    logdir = './board_log'

    if not os.path.exists(logdir):
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

    print(train_ds.batch(1).element_spec)
    model.fit(train_ds.batch(1), validation_data = val_ds.batch(1), epochs=2000, verbose=0,
        callbacks=[csv_logger_callback, checkpoint_callback, earlystop_callback, board_metric_callback, hp_callback])

    print('[Driver] Evaluating current model...')
    _, loss = model.evaluate(test_ds.batch(1))

    print('Current loss = %d' % loss)

    return loss

session_num = 0
mirrored_strategy = tf.distribute.MirroredStrategy()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:2"])
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["GPU:0","GPU:1","GPU:2"])
hparams_list = generate_hparams_list()

for params in hparams_list:
    run_name = "run-%d" % session_num
    print('--- Starting trial：%s' % run_name)
    print({h.name: params[h] for h in params})
    test_loss = train_test_model(params)
    session_num += 1