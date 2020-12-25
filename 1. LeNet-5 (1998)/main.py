'''
This paper was written to classify handwritten digit 0-9.
It has no padding
It used Avg Pooling instead of MaxPooling
'''

from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


def lenet_5(imgSize = (32, 32, 1)):

    lenet = Sequential(name='LeNet-5')
    
    lenet.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape = imgSize))
    lenet.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    
    lenet.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh'))
    lenet.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    lenet.add(Flatten())

    lenet.add(Dense(400, activation='tanh'))
    lenet.add(Dense(120, activation='tanh'))
    lenet.add(Dense(84, activation='tanh'))
    lenet.add(Dense(10, activation='softmax'))

    

    return lenet

if __name__ == '__main__':
    
    model = lenet_5()
    print('\n')
    print(model.summary())