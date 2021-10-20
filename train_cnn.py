# Tools import
import glob
import os
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
    label = np.load(label_path)
    data = np.load(input_path)
    data = np.expand_dims(data, -1)
    
    return tf.data.Dataset.from_tensor_slices((data, label))

def generate_training_combination(input_path_list, label_path_list):
    return product(input_path_list, label_path_list)

# Data Access
input_path_list = ['./ct_interpolated_3.npy', './ct_interpolated_5.npy','./ct_interpolated_10.npy']
label_path_list = ['./avg_change.npy']

# Training parameter
BATCH_SIZE = 8

for (input_path, label_path) in generate_training_combination(input_path_list, label_path_list):
    if not os.path.exists(input_path):
        basename = os.path.basename(input_path)
        number = basename.split('.')[0].split('_')[2]
        save_np_arr_with_channel(int(number))
    
    ds = build_ds_from(input_path, label_path)
    input_shape = ds.element_spec[0].shape
    
    print(input_shape)
        
    input_name = os.path.basename(input_path).split('.')[0]
    label_name = os.path.basename(label_path).split('.')[0]

    tf.debugging.set_log_device_placement(True)
    
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
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
        
        model.fit(ds.batch(BATCH_SIZE), epochs=1000, callbacks=[csv_logger_callback, checkpoint_callback, earlystop_callback])