'''
There are 16 layers which have trainable parameters (Conv2D, Dense). This architecture's names on it.

Conv2D: kernel=3x3, strides=1, padding=same
MaxPool: kernel=2x2, strides=2
'''

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def vgg16(imgSize = (224, 224, 3)):

    vgg = Sequential(name='VGG_16')
    
    # conv 64, 2 times
    vgg.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape = imgSize))
    vgg.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    vgg.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # conv 128, 2 times
    vgg.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    vgg.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # conv 256, 3 times
    vgg.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    vgg.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # conv 512, 3 times
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    vgg.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # conv 512, 3 times
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    vgg.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    
    vgg.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    vgg.add(Flatten())

    #FC layer
    vgg.add(Dense(4096, activation='relu'))
    vgg.add(Dense(4096, activation='relu'))

    # output layer
    vgg.add(Dense(1000, activation='softmax'))


    return vgg

if __name__ == '__main__':
    
    model = vgg16()
    print('\n')
    print(model.summary())