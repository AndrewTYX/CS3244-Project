
from tensorflow.keras.layers import LSTM, Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras import Model
from build_model import build_full_lstm

ct_input_shape = (512, 512, 3, 1)
raw_input_shape = (2, 2)
base_input_shape = (3,)
nn_feature_size = 10
ff_feature_size = 5
model = build_full_lstm(ct_input_shape, raw_input_shape, base_input_shape, nn_feature_size, ff_feature_size)
model.summary()