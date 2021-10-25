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

def build_cnn_model(input_shape):
    input_layer = Input(input_shape)
    conv_layer1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer1)
    pooling_layer1 = BatchNormalization()(pooling_layer1)  
    conv_layer2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer1)
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer2)
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer1)
    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer3)
    pooling_layer3 = BatchNormalization()(pooling_layer3)
    conv_layer4 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer3)
    pooling_layer4 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer4)
    pooling_layer4 = BatchNormalization()(pooling_layer4)
    conv_layer5 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer4)
    pooling_layer5 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer5)
    conv_layer6 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer5)
    pooling_layer6 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer6)
    conv_layer7 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(pooling_layer6)
    pooling_layer7 = MaxPool3D(pool_size=(2, 2, 2), padding='same')(conv_layer7)
    
    pooling_layer9 = BatchNormalization()(pooling_layer7)
    flatten_layer = Flatten()(pooling_layer9)
    
    dense_layer3 = Dense(units=512, activation='relu')(flatten_layer)
    dense_layer3 = Dropout(0.4)(dense_layer3)

    dense_layer4 = Dense(units=256, activation='relu')(dense_layer3)
    dense_layer4 = Dropout(0.4)(dense_layer3)
  
    output_layer = Dense(units=1, activation='linear')(dense_layer4)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def build_ds_from(input_path, label_path):
    print('Building dataset...')
    label = np.load(label_path)
    print(len(label))
    data = np.load(input_path)
    data = np.expand_dims(data, -1)
    
    return tf.data.Dataset.from_tensor_slices((data, label))

def generate_training_combination(input_path_list, label_path_list):
    return product(input_path_list, label_path_list)

def batch_dataset(ds, batch_size=1):
    return ds.batch(batch_size)

def split_dataset(dataset: tf.data.Dataset, 
                  dataset_size: int, 
                  train_ratio: float, 
                  validation_ratio: float, test_ratio:float) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    assert (train_ratio + validation_ratio) <= 1

    train_count = int(dataset_size * train_ratio)
    validation_count = int(dataset_size * validation_ratio)
    test_count = dataset_size - (train_count + validation_count)

    dataset = dataset.shuffle(dataset_size)

    train_dataset = dataset.take(train_count)
    validation_dataset = dataset.skip(train_count).take(validation_count)
    test_dataset = dataset.skip(validation_count + train_count).take(test_count)

    return batch_dataset(train_dataset), batch_dataset(validation_dataset), batch_dataset(test_dataset)
    
# Data Access
channel_size = [3, 5, 10, 15, 20, 30, 50, 70, 90]
label_path_list = ['./avg_change.npy']
dataset_size = 174

# tf.debugging.set_log_device_placement(True)
# Training parameter

for (channel_num, label_path) in generate_training_combination(channel_size, label_path_list):
    input_path = f'./ct_interpolated_{channel_num}.npy'
    if not os.path.exists(input_path):
        save_np_arr_with_channel(channel_num)
    
    ds = build_ds_from(input_path, label_path)
    input_shape = ds.element_spec[0].shape
    
    print(input_shape)
        
    input_name = os.path.basename(input_path).split('.')[0]
    label_name = os.path.basename(label_path).split('.')[0]

    model_path = f'./model_checkpoints/{input_name}_{label_name}'
    csv_log_path = f'./train_logs/{input_name}_{label_name}'
    
    if os.path.exists(model_path):
        continue
    
    # gpus = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # with strategy.scope():
    # Build and compile model
    model = build_cnn_model((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.MeanAbsoluteError()])
    
    # Setup callbacks
    csv_logger_callback = CSVLogger(f'./train_logs/{input_name}_{label_name}')
    checkpoint_callback = ModelCheckpoint(filepath = f'./model_checkpoints/{input_name}_{label_name}', 
                                        monitor='loss',
                                        save_best_only=True)
    earlystop_callback = EarlyStopping(monitor='loss', min_delta=1, patience=200)
    
    train_ds, val_ds, test_ds = split_dataset(ds, dataset_size, 0.8, 0.1, 0.1)
    
    model.fit(train_ds, validation_data = val_ds, epochs=1000, callbacks=[csv_logger_callback, checkpoint_callback, earlystop_callback])
    