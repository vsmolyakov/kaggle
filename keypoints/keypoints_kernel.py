import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import model_from_json

FTRAIN = '/data/vision/fisher/data1/kaggle/keypoints/training.csv'
FTEST = '/data/vision/fisher/data1/kaggle/keypoints/test.csv'

def load(test=False, cols=None):
    
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))
    
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    
    if cols:
        df = df[list(cols)+['Image']]
    
    print df.count()
    df = df.dropna()
    
    X = np.vstack(df['Image'].values)/255
    X = X.astype(np.float32)
    
    if not test:
        y = df[df.columns[:-1]].values
        y = (y-48)/48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None
    
    return X, y
    
def load2d(test=False, cols=None):
    
    X, y = load(test, cols)
    X = X.reshape(-1,1,96,96)
    
    return X, y
    
def plot_sample(x, y, axis):
    
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48+48, y[1::2]*48+48, marker='x', s=10)

if __name__ == "__main__":
    
    X, y = load2d(test=False)
    
    print "X.shape", X.shape
    print "y.shape", y.shape        
    
    #neural net
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96)))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    

    model.add(Convolution2D(128, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dense(30))    
                
    sgd = SGD(lr='0.01', momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    hist = model.fit(X, y, nb_epoch=100, validation_split=0.2)

    #save trained model
    #json_string = model.to_json()
    #open('model1_arch.json', 'w').write(json_string)    
    #model.save_weights('model1_weights.h5')
    
    #loading saved model
    #model = model_from_json(open('model1_arch.json',).read())
    #model.load_weights('model1_weights.h5')
                
    #display loss 
    f=plt.figure()
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.show()
    f.savefig('./loss.png')
    
    #predict test images
    X_test, _ = load2d(test=True)
    y_test = model.predict(X_test)
    
    fig = plt.figure(figsize=(6,6))
    for i in range(16):
        axis = fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
        plot_sample(X_test[i], y_test[i], axis)
    plt.show()
    fig.savefig('./predicted.png')        
    
    #write out submission
    #submission = pd.DataFrame(index=pd.RangeIndex(start=1, stop=27124, step=1), columns=['Location'])
    #submission['Location'] = y_test.reshape(-1,1)
    #submission.index.name = 'RowId'
    #submission.to_csv('./keypoints_pred.csv', index=True, header=True)
    
    
    
    
    