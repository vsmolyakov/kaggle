
import numpy as np
from PIL import Image
import gzip
import os

from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import matplotlib.pyplot as plt

np.random.seed(0)

def get_data():

    files = os.listdir('./data/denoising/train')
    train_imgs = np.zeros((len(files),258,540))
    for idx, filename in enumerate(files):
        img = Image.open('./data/denoising/train/'+filename).convert('L')
        img = img.resize((540,258),Image.ANTIALIAS)
        train_imgs[idx,:] = np.asarray(img)

    files = os.listdir('./data/denoising/train_cleaned')
    train_denoised_imgs = np.zeros((len(files),258,540))
    for idx, filename in enumerate(files):
        img = Image.open('./data/denoising/train_cleaned/'+filename).convert('L')
        img = img.resize((540,258),Image.ANTIALIAS)
        train_denoised_imgs[idx,:] = np.asarray(img)

    files = os.listdir('./data/denoising/test')
    test_imgs = np.zeros((len(files),258,540)) 
    for idx, filename in enumerate(files):
        img = Image.open('./data/denoising/test/'+filename).convert('L')
        img = img.resize((540,258),Image.ANTIALIAS)
        test_imgs[idx,:] = np.asarray(img)

    return train_imgs, train_denoised_imgs, test_imgs

if __name__ == "__main__":


    train, train_denoised, test = get_data()
    
    #normalize
    train /= 255.0
    train_denoised /= 255.0
    test /= 255.0

    #reshape
    train = np.reshape(train, (len(train), 1, 258, 540))
    train_denoised = np.reshape(train, (len(train), 1, 258, 540))
    test = np.reshape(train, (len(train), 1, 258, 540))

    #autoencoder
    input_img = Input(shape=(1,258,540))
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
  
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(train, train_denoised, nb_epoch=1, batch_size=64, shuffle=True, verbose=2)    

    #denoise test data
    denoised_imgs = autoencoder.predict(test)

    denoised_imgs = denoised_imgs.reshape((-1, 258, 540))
    #denoised_imgs *= 255.0
    
    #im = Image.fromarray(denoised_imgs[1,:,:])
    #im = im.convert("L")
    #im.save('./denoised.png')

    #create submission
    submission = gzip.open("conv_autoenc.csv.gz","wt")
    submission.write("id,value\n") 
    
    files = os.listdir('./data/denoising/test')
    for idx, filename in enumerate(files):
        img_idx = int(filename[:-4])
        img = Image.open('./data/denoising/test/'+filename).convert('L')
        img_dim = img.size

        denoised_im = Image.fromarray(denoised_imgs[idx,:,:])
        denoised_im = denoised_im.convert("L")
        denoised_im = denoised_im.resize(img_dim,Image.ANTIALIAS)
        denoised_im = np.asarray(denoised_im)

        for j in range(denoised_im.shape[1]):
            for i in range(denoised_im.shape[0]):
                submission.write("{}_{}_{},{}\n".format(img_idx,i+1,j+1,denoised_im[i,j]))

    submission.close()






