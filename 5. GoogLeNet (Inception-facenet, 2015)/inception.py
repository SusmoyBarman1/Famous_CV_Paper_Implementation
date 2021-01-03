import tensorflow as tf
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense, Dropout

'''
Create Inception block
'''
def inception_block(X, filters):
    x7, x1, x5, x2, x6, x4 = filters

    tower1 = Conv2D(x1, (1, 1), strides=(1, 1), padding='same', activation='relu')(X)
    tower1 = Conv2D(x5, (3, 3), strides=(1, 1), padding='same', activation='relu')(tower1)

    tower2 = Conv2D(x2, (1, 1), strides=(1, 1), padding='same', activation='relu')(X)
    tower2 = Conv2D(x6, (5, 5), strides=(1, 1), padding='same', activation='relu')(tower2)

    tower3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(X)
    tower3 = Conv2D(x4, (1, 1), strides=(1, 1), padding='same', activation='relu')(tower3)

    tower4 = Conv2D(x7, (1, 1), strides=(1, 1), padding='same', activation='relu')(X)

    depthConcat = concatenate([tower1, tower2, tower3, tower4])

    return depthConcat


'''
Create extra outputs of the network
'''

def sideOutput(X):

    x1 = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(X)
    x1 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu')(x1)

    FC = Flatten()(x1)

    FC = Dropout(0.7)(FC)

    FC = Dense(1024)(FC)

    FC = Activation('relu')(FC)

    FC = Dense(1000)(FC)

    FC = Activation('softmax')(FC)

    return FC

'''
Creating main model
'''
def create_model(input_shape=(224, 224, 3)):
    x_input = Input(input_shape)

    L1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='L1_conv')(x_input)
    L1 = BatchNormalization(axis=1, epsilon=0.00001, name='L1_norm')(L1)
    L1 = Activation('relu', name='L1_activation')(L1)

    L2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='L2')(L1)

    L3 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='L3_conv')(L2)
    L3 = BatchNormalization(axis=1, epsilon=0.00001, name='L3_norm')(L3)
    L3 = Activation('relu', name='L3_activation')(L3)

    L4 = Conv2D(192, (3, 3), strides=(1, 1), padding='same', name='L4_conv')(L3)
    L4 = BatchNormalization(axis=1, epsilon=0.00001, name='L4_norm')(L4)
    L4 = Activation('relu', name='L4_activation')(L4)

    L5 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='L5')(L4)

    # Inception Block: 3a
    I_3a = inception_block(L5, [64, 96, 128, 16, 32, 32])

    # Inception Block: 3b
    I_3b = inception_block(I_3a, [128, 128, 192, 32, 96, 64])

    L6 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='L6')(I_3b)

    # Inception Block: 4a
    I_4a = inception_block(L6, [192, 96, 208, 16, 48, 64])

    # Inception Block: 4b
    I_4b = inception_block(I_4a, [160, 112, 224, 24, 64, 64])

    # Inception Block: 4c
    I_4c = inception_block(I_4b, [128, 128, 256, 24, 64, 64])

    # Inception Block: 4d
    I_4d = inception_block(I_4c, [112, 144, 288, 32, 64, 64])

    # Inception Block: 4e
    I_4e = inception_block(I_4d, [256, 160, 320, 32, 128, 128])

    L7 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='L7')(I_4e)

    # Inception Block: 5a
    I_5a = inception_block(L7, [256, 160, 320, 32, 128, 128])

    # Inception Block: 5b
    I_5b = inception_block(I_5a, [384, 192, 384, 48, 128, 128])

    L8 = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='L8')(I_5b)

    FC = Flatten()(L8)

    FC = Dropout(0.4)(FC)

    FC = Dense(1000)(FC)

    FC = Activation('softmax', name='FC_activation')(FC)

    output_2 = Model(inputs=x_input, outputs=FC, name='Inception-googlenet_output_2')

    output_0_layer = sideOutput(I_4a)
    output_0 = Model(inputs=x_input, outputs=output_0_layer, name='Inception-googlenet_output_0')

    output_1_layer = sideOutput(I_4d)
    output_1 = Model(inputs=x_input, outputs=output_1_layer, name='Inception-googlenet_output_1')

    return output_0, output_1, output_2


if __name__ == '__main__':
    
    model_0, model_1, model_2 = create_model()
    print('\n\n\n-----------------Model_0 Summary----------------------\n\n')
    print(model_0.summary())

    print('\n\n\n\n-----------------Model_1 Summary----------------------\n\n')
    print(model_1.summary())

    print('\n\n\n\n-----------------Model_2 Summary----------------------\n\n')
    print(model_2.summary())

    