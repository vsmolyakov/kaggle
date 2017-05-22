import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import sys

np.random.seed(0)

BASE_DIR = './data/'
GLOVE_DIR = '/data/vision/fisher/data1/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.01

def text_to_wordlist(row, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    text = row['question']
    # Convert words to lower case and split them
    if type(text) is str:
        text = text.lower().split()
    else:
        return " "

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def main():
    
    #load embeddings
    print 'Indexing word vectors...'
    embeddings_index = {}
    f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    
    print 'Processing text dataset...' 
    train_df = pd.read_csv("./data/train.csv")
    test_df  = pd.read_csv("./data/test.csv")    
    
    q1df = train_df['question1'].reset_index()
    q2df = train_df['question2'].reset_index()
    q1df.columns = ['index', 'question']
    q2df.columns = ['index', 'question']
    texts_1 = q1df.apply(text_to_wordlist, axis=1, raw=True).tolist()
    texts_2 = q2df.apply(text_to_wordlist, axis=1, raw=True).tolist()
    labels = train_df['is_duplicate'].astype(int).tolist()
    print 'Found %s texts.' % len(texts_1)
    del q1df
    del q2df
        
    q1df = test_df['question1'].reset_index()
    q2df = test_df['question2'].reset_index()
    q1df.columns = ['index', 'question']
    q2df.columns = ['index', 'question']    
    test_texts_1 = q1df.apply(text_to_wordlist, axis=1, raw=True).tolist()
    test_texts_2 = q2df.apply(text_to_wordlist, axis=1, raw=True).tolist()
    test_labels = test_df['test_id'].astype(int).tolist()
    print 'Found %s texts.' % len(test_texts_1)
    del q1df
    del q2df
        
    '''
    texts_1 = [] 
    texts_2 = []
    labels = []  # list of label ids
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append(text_to_wordlist(values[3]))
            texts_2.append(text_to_wordlist(values[4]))
            labels.append(int(values[5]))
    print 'Found %s texts.' % len(texts_1)

    test_texts_1 = []
    test_texts_2 = []
    test_labels = []  # list of label ids
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append(text_to_wordlist(values[1]))
            test_texts_2.append(text_to_wordlist(values[2]))
            test_labels.append(values[0])
    print 'Found %s texts.' % len(test_texts_1)
    '''
    
    #tokenize, convert to sequences and pad
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    word_index = tokenizer.word_index
    print 'Found %s unique tokens.' % len(word_index)

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_labels = np.array(test_labels)
    del test_sequences_1
    del test_sequences_2
    del sequences_1
    del sequences_2
    
    #embedding matrix
    print 'Preparing embedding matrix...'
    nb_words = min(MAX_NB_WORDS, len(word_index))

    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print 'Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)
    
    #define the model
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = Conv1D(128, 3, activation='relu')(embedded_sequences_1)
    x1 = MaxPooling1D(10)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(64, activation='relu')(x1)
    x1 = Dropout(0.2)(x1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = Conv1D(128, 3, activation='relu')(embedded_sequences_2)
    y1 = MaxPooling1D(10)(y1)
    y1 = Flatten()(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(0.2)(y1)
    
    merged = merge([x1,y1], mode='concat')
    merged = BatchNormalization()(merged)
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    
    model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        
    model.fit([data_1,data_2], labels, validation_split=VALIDATION_SPLIT, nb_epoch=10, batch_size=1024, shuffle=True)
    preds = model.predict([test_data_1, test_data_2])

    keras_submission = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
    keras_submission.to_csv("keras_submission.csv", index=False)
                        

if __name__ == "__main__":
    main()
    