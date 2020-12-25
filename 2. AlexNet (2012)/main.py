from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def alexNet(imgSize = (227, 227, 3)):

    alex = Sequential(name='AlexNet')
    
    alex.add(Conv2D(filters=96, kernel_size=11, strides=4, activation='tanh', input_shape = imgSize))
    alex.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    
    alex.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='tanh'))
    alex.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    alex.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='tanh'))
    alex.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='tanh'))
    alex.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='tanh'))
    alex.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    

    alex.add(Flatten())

    alex.add(Dense(4096, activation='tanh'))
    alex.add(Dense(4096, activation='tanh'))
    alex.add(Dense(1000, activation='softmax'))

    return alex

if __name__ == '__main__':
    
    model = alexNet()
    print('\n')
    print(model.summary())