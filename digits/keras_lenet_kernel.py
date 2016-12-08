import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D 
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

if __name__ == "__main__":


    train = pd.read_csv('./data/train.csv').values
    test  = pd.read_csv('./data/test.csv').values

    img_rows, img_cols = 28, 28
    
    X_train = train[:,1:].reshape(train.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype(float)
    X_train = X_train/255.0
   
    X_test = test.reshape(test.shape[0], img_rows, img_cols, 1) 
    X_test = X_test.astype(float)
    X_test = X_test/255.0

    y_train = np_utils.to_categorical(train[:,0])
    num_classes = y_train.shape[1]

    #cnn parameters
    num_filters_l1 = 32
    filter_size_l1 = 5
    num_filters_l2 = 64
    filter_size_l2 = 5

    cnn = Sequential()
    #CONV -> RELU -> MAXPOOL
    cnn.add(Convolution2D(num_filters_l1, filter_size_l1, filter_size_l1, input_shape=(img_rows, img_cols, 1), border_mode='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #CONV -> RELU -> MAXPOOL
    cnn.add(Convolution2D(num_filters_l2, filter_size_l2, filter_size_l2, border_mode='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #FC -> RELU
    cnn.add(Flatten())
    cnn.add(Dense(128))
    cnn.add(Activation('relu'))

    #Softmax Classifier
    cnn.add(Dense(num_classes))
    cnn.add(Activation('softmax'))  

    cnn.summary()
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    cnn.fit(X_train, y_train, batch_size=128, nb_epoch=1, verbose=1)

    y_pred = cnn.predict_classes(X_test) 

    #create submission 
    submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=28001, step=1), columns=['Label'])
    submission['Label'] = y_pred.reshape(-1,1)
    submission.index.name = "ImageId"
    submission.to_csv('./lenet_pred.csv', index=True, header=True)


  



