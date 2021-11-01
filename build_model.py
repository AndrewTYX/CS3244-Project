def build_input_layer(ct_input_shape, raw_input_shape):
    ct_input = Input(ct_input_shape)
    time_input = Input(raw_input_shape)
    return ct_input, time_input
    
def build_cnn_layers(input_layer, feature_size):
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
  
    feature_layer = Dense(units=feature_size, activation='relu')(dense_layer4)
    
    return feature_layer
