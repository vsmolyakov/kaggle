import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import os, re, gc
from glob import glob
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

import librosa
import librosa.display
from tqdm import tqdm
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

sns.set_style("whitegrid")

SAVE_PATH = '/data/vision/fisher/data1/kaggle/speech/'
DATA_PATH = '/data/vision/fisher/data1/kaggle/speech/data/'

SAMPLE_LEN = 16000
NEW_SAMPLE_RATE = 8000
CLASS_LABELS = 'yes no up down left right on off stop go silence unknown'.split()

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N // 2)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, window='hann',
        nperseg=nperseg, noverlap=noverlap, detrend=False) 
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob(os.path.join(dirpath, r'*/*' + ext))
    pat = r'.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
        #end if
    #end for
    pat = r'.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
        #end if
    #end for
    return labels, fnames 

def pad_audio(samples):
    if len(samples) >= SAMPLE_LEN: return samples
    else: return np.pad(samples, pad_width=(SAMPLE_LEN - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in CLASS_LABELS:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
        #end if
    #end for
    return pd.get_dummies(pd.Series(nlabels))

def test_data_generator(batch=128):
    test_data_path = DATA_PATH + '/test/audio/'
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        #end if
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int((NEW_SAMPLE_RATE / float(rate)) * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=NEW_SAMPLE_RATE)
        imgs.append(specgram)
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = np.expand_dims(imgs, axis=-1)
            yield fnames, imgs
        #end if
    #end for
    if i < batch:
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis=-1)
        yield fnames, imgs
    #end if
    raise StopIteration()

def step_decay(epoch):
    lr_init = 0.001
    drop = 0.5
    epochs_drop = 4.0
    lr_new = lr_init * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr_new

class LR_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))


#load data
train_data_path = DATA_PATH + '/train/audio/'
labels, fnames = list_wavs_fname(train_data_path)

#visualize data
sample_file = '/yes/0a7c2a8d_nohash_0.wav'
sample_rate, samples = wavfile.read(train_data_path + sample_file)
freqs, times, spectrogram = log_specgram(samples, sample_rate)

S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
delta_mfcc = librosa.feature.delta(mfcc, order=1)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
mfcc_feat = np.vstack((mfcc, delta_mfcc, delta2_mfcc))

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False)
ax1.plot(np.linspace(0, sample_rate/float(len(samples)),sample_rate), samples)
ax1.set_title("raw wave of " + sample_file); ax1.set_ylabel("amplitude")
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16]); ax2.set_xticks(times[::16])
ax2.set_title('spectrogram of ' + sample_file)
ax2.set_ylabel('freq in Hz'); ax2.set_xlabel('seconds') 
plt.savefig('./figures/speech_features1.png')

plt.figure()
ax1 = plt.subplot(2,1,1)
librosa.display.specshow(mfcc)
plt.title("MFCC")
ax2 = plt.subplot(2,1,2, sharex=ax1)
librosa.display.specshow(delta_mfcc, x_axis='time')
plt.title("delta MFCC")
plt.savefig('./figures/speech_features2.png')

import pdb; pdb.set_trace()

#reduce training size
labels_fnames = zip(labels, fnames)
shuffle(labels_fnames)
NUM_TRAIN = np.int(0.1 * len(labels_fnames))
#NUM_TRAIN = -1 

print "loading training data..."
x_train, y_train = [], []
for label, fname in tqdm(labels_fnames[:NUM_TRAIN]):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > SAMPLE_LEN:
        n_samples = chop_audio(samples)
    else:
        n_samples = [samples]
    #end if

    for samples in n_samples:
        resampled = signal.resample(samples, int((NEW_SAMPLE_RATE / float(sample_rate)) * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=NEW_SAMPLE_RATE)
        y_train.append(label)
        x_train.append(specgram)
    #end for
#end for

x_train = np.array(x_train)
x_train = np.expand_dims(x_train, axis=-1) 

y_train = label_transform(y_train)
label_index = y_train.columns.values
num_classes = len(label_index)
y_train = y_train.values

#free up memory
del labels, fnames, labels_fnames
gc.collect()

#TODO: try without re-sampling (more data)
#TODO: try merging MFCC coefficients (multi-input model)
#TODO: add batch normalization
#TODO: check over-fitting on dev and add regularization
#TODO: better pre-processing of the input

#training params
batch_size = 128 
num_epochs = 16 

#model parameters
img_rows = 99
img_cols = 81
weight_decay = 1e-4

#CNN architecture
print "training CNN ..."
model = Sequential()
model.add(BatchNormalization(input_shape=(img_rows, img_cols, 1))) 
model.add(Conv2D(32, kernel_size = (3, 3), padding='same', activation='relu')) 
model.add(Conv2D(32, kernel_size = (3, 3), padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

#define callbacks
file_name = SAVE_PATH + 'speech-weights-checkpoint.h5'
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
hist_lr = LR_hist()
reduce_lr = LearningRateScheduler(step_decay) 
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=8, verbose=1)
callbacks_list = [checkpoint, tensor_board, hist_lr, reduce_lr, early_stopping]

#model training
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.2, shuffle=True, verbose=2)

model.save(SAVE_PATH + 'speech_final_model.h5', overwrite=True)
model.save_weights(SAVE_PATH + 'speech_final_weights.h5',overwrite=True)

#load saved model
#model = load_model(SAVE_PATH + 'speech_final_model.h5')

#model prediction
index, results = [], []
for fname, imgs in test_data_generator(batch=batch_size):
    preds = model.predict(imgs)
    preds_class = np.argmax(preds, axis=-1)
    preds_labels = [label_index[p] for p in preds_class]
    index.extend(fname) 
    results.extend(preds_labels)
#end for

#create a submission
submission_df = pd.DataFrame(columns=['fname', 'label'])
submission_df['fname'] = index
submission_df['label'] = results
submission_df.to_csv("./data/first_speech.csv", index=False)


#generate plots
plt.figure()
plt.plot(hist.history['loss'], c='b', lw=2.0, label='train')
plt.plot(hist.history['val_loss'], c='r', lw=2.0, label='val')
plt.title('TF speech model')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.savefig('./figures/speech_cnn_loss.png')

plt.figure()
plt.plot(hist.history['acc'], c='b', lw=2.0, label='train')
plt.plot(hist.history['val_acc'], c='r', lw=2.0, label='val')
plt.title('TF speech model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('./figures/speech_cnn_acc.png')

plt.figure()
plt.plot(hist_lr.lr, lw=2.0, label='learning rate')
plt.title('TF speech Model')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.savefig('./figures/speech_learning_rate.png')

plot_model(model, show_shapes=True, to_file='./figures/speech_model.png')


