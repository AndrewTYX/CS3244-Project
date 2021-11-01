from tensorflow.keras.layers import LSTM, Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout, Concatenate, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

def build_input_layer(ct_input_shape, raw_input_shape, base_input_shape):
    ct_input = Input(ct_input_shape)
    time_input = Input(raw_input_shape)
    base_input = Input(base_input_shape)
    return ct_input, time_input, base_input
    
def build_cnn_layers(feature_size):
    model = Sequential()
    model.add(TimeDistributed(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    model.add(TimeDistributed(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPool3D(pool_size=(2, 2, 2), padding='same')))
    
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Flatten()))
    
    model.add(TimeDistributed(Dense(units=512, activation='relu')))
    model.add(TimeDistributed(Dropout(0.4)))

    model.add(TimeDistributed(Dense(units=256, activation='relu')))
    model.add(TimeDistributed(Dropout(0.4)))
  
    model.add(TimeDistributed(Dense(units=feature_size, activation='relu')))
    
    return model

def build_ff_layers(feature_size):
    model = Sequential()
    model.add(TimeDistributed(Dense(units=16, activation='relu')))
    model.add(TimeDistributed(Dense(units=32, activation='relu')))
    model.add(TimeDistributed(Dense(units=feature_size, activation='softmax')))
    return model

def build_LSTM(nn_feature_size, ff_feature_size, time_feature_size):
    n_features = nn_feature_size + ff_feature_size + time_feature_size
    n_steps = 2
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences = True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation = 'relu'))
    model.add(Dense(1))
    
    return model
    
def build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                    nn_feature_size, ff_feature_size):
    '''
    ct shape should be (time_step)
    '''`
    ct_input, time_input, base_input = build_input_layer(ct_input_shape, raw_input_shape, base_input_shape)
    
    x1 = build_cnn_layers(ct_input_shape, nn_feature_size)(ct_input)
    x2 = time_input
    x3 = build_ff_layers(base_input_shape, ff_feature_size)(base_input)
    
    out = Concatenate()([x1, x2, x3])
    
    out = build_LSTM(nn_feature_size, ff_feature_size, 2)(out)
    
    model = Model([ct_input, time_input, base_input], out)
    
    return model
    
    
    
    
    
    