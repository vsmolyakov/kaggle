import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import re
import csv
import math
import codecs

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import LSTM, Embedding, Bidirectional

from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler 
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 

sns.set_style("whitegrid")

SAVE_PATH = '/data/vision/fisher/data1/kaggle/toxic/'
DATA_PATH = '/data/vision/fisher/data1/kaggle/toxic/data/'

MAX_NB_WORDS = 100000
#EMBEDDING_DIR = '/data/vision/fisher/data1/Glove/'
EMBEDDING_DIR = '/data/vision/fisher/data1/FastText/'

np.random.seed(0)

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

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

#load embeddings
print 'loading word embeddings...'
"""
embeddings_index = {}
f = codecs.open(os.path.join(EMBEDDING_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))
"""

embeddings_index = {}
f = codecs.open(os.path.join(EMBEDDING_DIR, 'wiki.en.vec'), encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

#load data
train_df = pd.read_csv(DATA_PATH + '/train.csv', sep=',', header=0)
test_df = pd.read_csv(DATA_PATH + '/test.csv', sep=',', header=0)
test_df = test_df.fillna('_NA_')

print "num train: ", train_df.shape[0]
print "num test: ", test_df.shape[0]

label_names = filter(lambda x: x not in ['id', 'comment_text'], train_df.columns.tolist())
y_train = train_df[label_names].values


#visualize word distribution
train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
sns.distplot(train_df['doc_len'], hist=True, kde=True, color='b', label='doc len')
plt.axvline(x=max_seq_len, color='k', linestyle='--', label='max len')
plt.title('comment length'); plt.legend()
plt.savefig('./figures/comment_length_hist.png')

raw_docs_train = train_df['comment_text'].tolist()
raw_docs_test = test_df['comment_text'].tolist() 
num_classes = len(label_names)

print "pre-processing train data..."
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))
#end for

processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))
#end for

print "tokenizing input data..."
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)
word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print "dictionary size: ", len(word_index) 

#pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

#training params
batch_size = 256 
num_epochs = 16 

#model parameters
hidden_size = 128 
embed_dim = 300
lstm_dropout = 0.2
dense_dropout = 0.5
weight_decay = 1e-3

#embedding matrix
print 'preparing embedding matrix...'
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print 'number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)

#LSTM architecture
print "training LSTM ..."
model = Sequential()
model.add(Embedding(nb_words, embed_dim,
          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
model.add(Bidirectional(LSTM(hidden_size, dropout=lstm_dropout, recurrent_dropout=lstm_dropout)))
model.add(Dense(hidden_size, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Dropout(dense_dropout))
model.add(Dense(hidden_size/4, kernel_regularizer=regularizers.l2(weight_decay), activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

#define callbacks
file_name = SAVE_PATH + 'lstm-weights-checkpoint.h5'
checkpoint = ModelCheckpoint(file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#tensor_board = TensorBoard(log_dir='./logs', write_graph=True)
hist_lr = LR_hist()
reduce_lr = LearningRateScheduler(step_decay) 
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=16, verbose=1)
callbacks_list = [checkpoint, hist_lr, reduce_lr, early_stopping]

#model training
hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)

model.save(SAVE_PATH + 'lstm_final_model.h5', overwrite=True)
model.save_weights(SAVE_PATH + 'lstm_final_weights.h5',overwrite=True)

#load saved model
#model = load_model(SAVE_PATH + 'final_model.h5')

y_test = model.predict(word_seq_test)

#create a submission
submission_df = pd.DataFrame(columns=['id'] + label_names)
submission_df['id'] = test_df['id'].values 
submission_df[label_names] = y_test 
submission_df.to_csv("./data/toxic_submission_first.csv", index=False)

#generate plots
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('LSTM sentiment')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.savefig('./figures/lstm_sentiment_loss.png')

plt.figure()
plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('LSTM sentiment')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('./figures/lstm_sentiment_acc.png')

plt.figure()
plt.plot(hist_lr.lr, lw=2.0, label='learning rate')
plt.title('LSTM sentiment')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.savefig('./figures/lstm_sentiment_learning_rate.png')

plot_model(model, show_shapes=True, to_file='./figures/lstm_sentiment_model.png')



