from tensorflow.keras.layers import LSTM, Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout, Concatenate, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

def build_input_layer(raw_input_shape,ct_input_shape=None, base_input_shape=None):
    assert ct_input_shape != None or base_input_shape != None

    if ct_input_shape == None:
        return Input(raw_input_shape), Input(base_input_shape)
    
    if base_input_shape == None:
        return Input(raw_input_shape), Input(ct_input_shape)

    return Input(raw_input_shape), Input(ct_input_shape), Input(base_input_shape)
    
def build_cnn_layers(feature_size, kernel_size, pool_size, dropout):
    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=64, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(kernel_size, kernel_size, kernel_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=64, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=128, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=256, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    model.add(Conv3D(filters=128, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    model.add(Conv3D(filters=32, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(pool_size, pool_size, pool_size), padding='same'))
    
    model.add(BatchNormalization())
    model.add(Flatten())
    
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(dropout))
  
    model.add(Dense(units=feature_size, activation='relu'))
    
    return model

def build_ff(time_feature_size, nn_feature_size, ff_feature_size, units):
    feature_size = time_feature_size + nn_feature_size + ff_feature_size
    model = Sequential()
    model.add(Dense(units=units[0], activation='relu'))
    model.add(Dense(units=units[1], activation='relu'))
    model.add(Dense(units=feature_size, activation='softmax'))
    return model

def build_LSTM(time_feature_size, unit):
    n_steps = 2

    model = Sequential()
    model.add(LSTM(unit, activation='relu', return_sequences = True, input_shape=(n_steps, time_feature_size)))
    model.add(LSTM(unit, activation = 'relu'))
    model.add(Dense(1))
    
    return model
    
def build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape,
                    nn_feature_size, ff_feature_size, cnn_kernel_size=3, cnn_pool_size=2, cnn_drop_out=0.4, lstm_unit=50, ff_units=[16,32]):
    '''
    ct shape should be (time_step)
    '''
    time_input, ct_input, base_input = build_input_layer(raw_input_shape = raw_input_shape, ct_input_shape = ct_input_shape, base_input_shape = base_input_shape)
    
    x1 = build_cnn_layers(nn_feature_size, cnn_kernel_size, cnn_pool_size, cnn_drop_out)(ct_input)
    x2 = build_LSTM(time_feature_size = 2, unit = lstm_unit)(time_input)
    x3 = base_input
    
    out = Concatenate()([x1, x2, x3])
    
    out = build_ff(time_feature_size = 2, nn_feature_size = nn_feature_size, ff_feature_size = ff_feature_size, units = ff_units)(out)
    
    model = Model([ct_input, time_input, base_input], out)
    
    return model
    
    
# def build_ct_lstm(ct_input_shape, raw_input_shape, nn_feature_size, cnn_kernel_size=3, cnn_pool_size=2, cnn_drop_out=0.4, lstm_unit=50):
#     time_input, ct_input = build_input_layer(raw_input_shape=raw_input_shape, ct_input_shape=ct_input_shape)
    
#     x1 = build_cnn_layers(nn_feature_size, cnn_kernel_size, cnn_pool_size, cnn_drop_out)(ct_input)
#     x2 = time_input

#     out = Concatenate()([x1, x2])

#     out = build_LSTM(time_feature_size=2, nn_feature_size=nn_feature_size, unit=lstm_unit)

#     model = Model([ct_input, time_input], out)

#     return model


# def build_base_lstm(base_input_shape, raw_input_shape, ff_feature_size, lstm_unit=50):
#     time_input, base_input = build_input_layer(raw_input_shape=raw_input_shape, base_input_shape=base_input_shape)
    
#     x1 = build_ff_layers(ff_feature_size)(base_input)
#     x2 = time_input

#     out = Concatenate()([x1, x2])

#     out = build_LSTM(time_feature_size=2, nn_feature_size=ff_feature_size, unit=lstm_unit)

#     model = Model([base_input, time_input], out)

#     return model

    