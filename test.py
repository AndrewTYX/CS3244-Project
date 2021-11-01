from build_model import build_input_layer
from build_model import build_cnn_layers
from build_model import build_ff_layers
from build_model import build_LSTM
from tensorflow.keras.layers import LSTM, Input, Conv3D, MaxPool3D, BatchNormalization, Flatten, Dense, Dropout, Concatenate
ct_input_shape = (2，512, 512, 3, 1)
raw_input_shape = 2
base_input_shape = (2，4)

nn_feature_size = 10
ff_feature_size = 10

ct_input, time_input, base_input = build_input_layer(ct_input_shape, raw_input_shape, base_input_shape)

x1 = build_cnn_layers(ct_input_shape, nn_feature_size)(ct_input)
x2 = time_input
x3 = build_ff_layers(base_input_shape, ff_feature_size)(base_input)

print(x1)
print(x2)
print(x3)

out = Concatenate()([x1, x2, x3])
    
out = build_LSTM(nn_feature_size, ff_feature_size, 2)(out)