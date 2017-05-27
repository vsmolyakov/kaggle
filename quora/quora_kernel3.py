import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import re
import os
import codecs
import matplotlib.pyplot as plt
import datetime, time, json
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Merge, BatchNormalization, TimeDistributed
from keras.layers import Lambda, Activation, Flatten, Convolution1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import initializers
from keras import backend as K
from collections import defaultdict

GLOVE_DIR = '/data/vision/fisher/data1/Glove/'

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

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
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"usa", "America", text)
    text = re.sub(r"canada", "Canada", text)
    text = re.sub(r"japan", "Japan", text)
    text = re.sub(r"germany", "Germany", text)
    text = re.sub(r"burma", "Burma", text)
    text = re.sub(r"rohingya", "Rohingya", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"cambodia", "Cambodia", text)
    text = re.sub(r"zealand", "Zealand", text)
    text = re.sub(r"norway", "Norway", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"pakistan", "Pakistan", text)
    text = re.sub(r"britain", "Britain", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iphone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    text = ''.join([c for c in text if c not in punctuation])
    
    # Return a list of words
    return(text)


def process_questions(question_list, questions, question_list_name, dataframe):
    '''transform questions and display progress'''
    for question in questions:
        question_list.append(text_to_wordlist(question))
        if len(question_list) % 100000 == 0:
            progress = len(question_list)/len(dataframe) * 100
            print("{} is {}% complete.".format(question_list_name, round(progress, 1)))
            
def main():
    
    #load data
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    train = train.fillna('empty')
    test = test.fillna('empty')
    
    y_train = train.is_duplicate
    test_labels = test['test_id'].astype(int).tolist()
    
    print 'Processing text dataset...' 
    train_question1 = []
    process_questions(train_question1, train.question1, 'train_question1', train)
    
    train_question2 = []
    process_questions(train_question2, train.question2, 'train_question2', train)
    
    test_question1 = []
    process_questions(test_question1, test.question1, 'test_question1', test)
    
    test_question2 = []
    process_questions(test_question2, test.question2, 'test_question2', test)
    
    
    # Find the length of questions
    lengths = []
    for question in train_question1:
        lengths.append(len(question.split()))

    for question in train_question2:
        lengths.append(len(question.split()))

    lengths = pd.DataFrame(lengths, columns=['counts'])
    lengths.counts.describe
    print(np.percentile(lengths.counts, 99.5))


    print "fitting a tokenizer..."    
    num_words = 200000
    all_questions = train_question1 + train_question2 + test_question1 + test_question2
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(all_questions)
    
    print "converting to sequences..."    
    train_question1_word_sequences = tokenizer.texts_to_sequences(train_question1)
    train_question2_word_sequences = tokenizer.texts_to_sequences(train_question2)       
    test_question1_word_sequences = tokenizer.texts_to_sequences(test_question1)
    test_question2_word_sequences = tokenizer.texts_to_sequences(test_question2)
    
    word_index = tokenizer.word_index
    print "Words in index: %d" % len(word_index)
    
    print "padding sequences..."
    max_question_len = 36
    train_q1 = pad_sequences(train_question1_word_sequences, maxlen = max_question_len, padding = 'post', truncating = 'post')    
    train_q2 = pad_sequences(train_question2_word_sequences, maxlen = max_question_len, padding = 'post', truncating = 'post')
    test_q1 = pad_sequences(test_question1_word_sequences, maxlen = max_question_len, padding = 'post', truncating = 'post')
    test_q2 = pad_sequences(test_question2_word_sequences, maxlen = max_question_len, padding = 'post', truncating = 'post')
    
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
   
    embedding_dim = 300
    nb_words = len(word_index)
    word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            word_embedding_matrix[i] = embedding_vector

    print 'Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0)
    
    units = 128 # Number of nodes in the Dense layers
    dropout = 0.25 # Percentage of nodes to drop
    nb_filter = 32 # Number of filters to use in Convolution1D
    filter_length = 3 # Length of filter for Convolution1D

    # Initialize weights and biases for the Dense layers
    weights = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
    bias = 'zeros'

    model1 = Sequential()
    model1.add(Embedding(nb_words + 1,
                        embedding_dim,
                        weights = [word_embedding_matrix],
                        input_length = max_question_len,
                        trainable = False))

    model1.add(Convolution1D(filters = nb_filter, 
                            kernel_size = filter_length, 
                            padding = 'same'))
    model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    model1.add(Dropout(dropout))

    model1.add(Convolution1D(filters = nb_filter, 
                            kernel_size = filter_length, 
                            padding = 'same'))
    model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    model1.add(Dropout(dropout))
    model1.add(Flatten())


    model2 = Sequential()
    model2.add(Embedding(nb_words + 1,
                        embedding_dim,
                        weights = [word_embedding_matrix],
                        input_length = max_question_len,
                        trainable = False))

    model2.add(Convolution1D(filters = nb_filter, 
                            kernel_size = filter_length, 
                            padding = 'same'))
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    model2.add(Dropout(dropout))

    model2.add(Convolution1D(filters = nb_filter, 
                            kernel_size = filter_length, 
                            padding = 'same'))
    model2.add(BatchNormalization())
    model2.add(Activation('relu'))
    model2.add(Dropout(dropout))
    model2.add(Flatten())


    model3 = Sequential()
    model3.add(Embedding(nb_words + 1,
                        embedding_dim,
                        weights = [word_embedding_matrix],
                        input_length = max_question_len,
                        trainable = False))
    model3.add(TimeDistributed(Dense(embedding_dim)))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Dropout(dropout))
    model3.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))


    model4 = Sequential()
    model4.add(Embedding(nb_words + 1,
                        embedding_dim,
                        weights = [word_embedding_matrix],
                        input_length = max_question_len,
                        trainable = False))

    model4.add(TimeDistributed(Dense(embedding_dim)))
    model4.add(BatchNormalization())
    model4.add(Activation('relu'))
    model4.add(Dropout(dropout))
    model4.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim, )))
    

    modela = Sequential()
    modela.add(Merge([model1, model2], mode='concat'))
    modela.add(Dense(units*2, kernel_initializer=weights, bias_initializer=bias))
    modela.add(BatchNormalization())
    modela.add(Activation('relu'))
    modela.add(Dropout(dropout))

    modela.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
    modela.add(BatchNormalization())
    modela.add(Activation('relu'))
    modela.add(Dropout(dropout))

    modelb = Sequential()
    modelb.add(Merge([model3, model4], mode='concat'))
    modelb.add(Dense(units*2, kernel_initializer=weights, bias_initializer=bias))
    modelb.add(BatchNormalization())
    modelb.add(Activation('relu'))
    modelb.add(Dropout(dropout))

    modelb.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
    modelb.add(BatchNormalization())
    modelb.add(Activation('relu'))
    modelb.add(Dropout(dropout))


    model = Sequential()
    model.add(Merge([modela, modelb], mode='concat'))
    model.add(Dense(units*2, kernel_initializer=weights, bias_initializer=bias))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units, kernel_initializer=weights, bias_initializer=bias))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer=weights, bias_initializer=bias))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    # save the best weights for predicting the test question pairs
    save_best_weights = 'question_pairs_weights.h5'

    t0 = time.time()
    callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]

    history = model.fit([train_q1, train_q2, train_q1, train_q2],
                        y_train,
                        batch_size=256,
                        epochs=10,
                        validation_split=0.15,
                        verbose=True,
                        shuffle=True,
                        callbacks=callbacks)
    t1 = time.time()
    print "Minutes elapsed: %f" % ((t1 - t0) / 60.)
    
    
    summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                                'train_acc': history.history['acc'],
                                'valid_acc': history.history['val_acc'],
                                'train_loss': history.history['loss'],
                                'valid_loss': history.history['val_loss']})
                        
    plt.plot(summary_stats.train_loss)
    plt.plot(summary_stats.valid_loss)
    plt.show()
    
    
    model.load_weights(save_best_weights)
    predictions = model.predict([test_q1, test_q2, test_q1, test_q2], verbose = True)
    
    keras_submission = pd.DataFrame({"test_id":test_labels, "is_duplicate":predictions.ravel()})
    keras_submission.to_csv("keras_submission.csv", index=False)


if __name__ == "__main__":    
    main()