import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import math
import os, re, gc
from glob import glob
import cPickle as pickle
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import concatenate
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM, Bidirectional, BatchNormalization

from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model

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
NEW_SAMPLE_RATE = 16000
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
            x_spec_cnn = []
            x_mfcc_cnn = []
            x_spec_lstm = []
            x_mfcc_lstm = []
            fnames = []
        #end if
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(samples, int((NEW_SAMPLE_RATE / float(rate)) * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=NEW_SAMPLE_RATE)
        
        S = librosa.feature.melspectrogram(resampled, sr=NEW_SAMPLE_RATE, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc_feat = np.vstack((mfcc, delta_mfcc, delta2_mfcc))

        x_spec_cnn.append(specgram)
        x_mfcc_cnn.append(mfcc_feat)
        x_spec_lstm.append(specgram.T)
        x_mfcc_lstm.append(mfcc_feat.T)
        fnames.append(path.split('/')[-1])
        if i == batch:
            i = 0
            x_spec_lstm = np.array(x_spec_lstm)
            x_mfcc_lstm = np.array(x_mfcc_lstm)

            x_spec_cnn = np.array(x_spec_cnn)
            x_spec_cnn = np.expand_dims(x_spec_cnn, axis=-1)
            x_mfcc_cnn = np.array(x_mfcc_cnn)
            x_mfcc_cnn = np.expand_dims(x_mfcc_cnn, axis=-1)

            yield fnames, x_spec_cnn, x_mfcc_cnn, x_spec_lstm, x_mfcc_lstm
        #end if
    #end for
    if i < batch:
        x_spec_lstm = np.array(x_spec_lstm)
        x_mfcc_lstm = np.array(x_mfcc_lstm)

        x_spec_cnn = np.array(x_spec_cnn)
        x_spec_cnn = np.expand_dims(x_spec_cnn, axis=-1)
        x_mfcc_cnn = np.array(x_mfcc_cnn)
        x_mfcc_cnn = np.expand_dims(x_mfcc_cnn, axis=-1)

        yield fnames, x_spec_cnn, x_mfcc_cnn, x_spec_lstm, x_mfcc_lstm
    #end if
    raise StopIteration()

def step_decay(epoch):
    lr_init = 0.001
    drop = 0.5
    epochs_drop = 8.0
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

print "number of training examples: ", len(fnames)
print "number of unique labels: ", len(np.unique(labels))

#reduce training size
labels_fnames = zip(labels, fnames)
shuffle(labels_fnames)
#NUM_TRAIN = np.int(0.1 * len(labels_fnames))
NUM_TRAIN = -1 

print "loading training data..."
y_train = []
x_spec_cnn, x_mfcc_cnn = [], []
x_spec_lstm, x_mfcc_lstm = [], []
for label, fname in tqdm(labels_fnames[:NUM_TRAIN]):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > SAMPLE_LEN:
        n_samples = chop_audio(samples)
    else:
        n_samples = [samples]
    #end if

    for samples in n_samples:
        """
        resampled = signal.resample(samples, int((NEW_SAMPLE_RATE / float(sample_rate)) * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=NEW_SAMPLE_RATE)

        S = librosa.feature.melspectrogram(resampled, sr=NEW_SAMPLE_RATE, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc, order=1)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        mfcc_feat = np.vstack((mfcc, delta_mfcc, delta2_mfcc))
        """

        y_train.append(label)
        """
        x_spec_lstm.append(specgram.T)
        x_mfcc_lstm.append(mfcc_feat.T)
        x_spec_cnn.append(specgram)
        x_mfcc_cnn.append(mfcc_feat)
        """
    #end for
#end for

"""
x_spec_lstm = np.array(x_spec_lstm)
x_mfcc_lstm = np.array(x_mfcc_lstm)

x_spec_cnn = np.array(x_spec_cnn)
x_spec_cnn = np.expand_dims(x_spec_cnn, axis=-1) 
x_mfcc_cnn = np.array(x_mfcc_cnn)
x_mfcc_cnn = np.expand_dims(x_mfcc_cnn, axis=-1) 
"""

y_train = label_transform(y_train)
label_index = y_train.columns.values
num_classes = len(label_index)
y_train = y_train.values

print "len(label_index): ", len(label_index)

#free up memory
del labels, fnames, labels_fnames
gc.collect()

#load saved model
print "loading ensemble..."
model_cnn = load_model(SAVE_PATH + '/trained_models/arch2/speech_final_model.h5')
model_lstm = load_model(SAVE_PATH + '/trained_models/arch3/speech_final_model.h5')

print "tuning ensemble weights..."
weight_cnn = 0.5
weight_lstm = 0.5

print "cnn weight: ", weight_cnn
print "lstm weight: ", weight_lstm

#model prediction
print "predicting on test data..."
batch_size = 128
index, results = [], []
for fname, x_spec_cnn, x_mfcc_cnn, x_spec_lstm, x_mfcc_lstm in tqdm(test_data_generator(batch=batch_size)):
    preds_cnn = model_cnn.predict([x_spec_cnn, x_mfcc_cnn])
    preds_lstm = model_lstm.predict([x_spec_lstm, x_mfcc_lstm])
    preds_ensemble = weight_cnn * preds_cnn + weight_lstm * preds_lstm

    preds_class = np.argmax(preds_ensemble, axis=-1)
    preds_labels = [label_index[p] for p in preds_class]
    index.extend(fname) 
    results.extend(preds_labels)
#end for

#create a submission
submission_df = pd.DataFrame(columns=['fname', 'label'])
submission_df['fname'] = index
submission_df['label'] = results
submission_df.to_csv("./data/speech_fourth_submission.csv", index=False)

plot_model(model_cnn, show_shapes=True, to_file='./figures/speech_model_cnn.png')
plot_model(model_lstm, show_shapes=True, to_file='./figures/speech_model_lstm.png')

