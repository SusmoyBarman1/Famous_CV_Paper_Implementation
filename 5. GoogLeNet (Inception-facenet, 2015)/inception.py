import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.core import Lambda, Flatten, Dense

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

    model = Model(inputs=x_input, outputs=L4, name='Inception-googlenet')

    return model


if __name__ == '__main__':
    
    model = create_model()
    print('\n')
    print(model.summary())

    