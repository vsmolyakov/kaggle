import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re

import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def review_to_words(raw_review):

    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]

    return " ".join(meaningful_words)

if __name__ == "__main__":

    train = pd.read_csv("./data/word2vec/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/word2vec/testData.tsv", header=0, delimiter="\t", quoting=3)

    num_reviews = train["review"].size
    sentiment_train = train["sentiment"].values

    clean_train_reviews = []
    print "cleaning training data..."
    for i in xrange(0, num_reviews):
        if ((i+1)%5000 == 0):
            print "review %d of %d" %(i+1, num_reviews)
        clean_train_reviews.append(review_to_words(train['review'][i]))
    
    vectorizer = TfidfVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()  
 
    #visualize training data
    print "fitting TSNE ..."
    tsne = TSNE(n_components=2, random_state=0)
    tsne_words = tsne.fit_transform(train_data_features[0:2000,:])

    f=plt.figure()
    plt.scatter(tsne_words[:,0], tsne_words[:,1], c=sentiment_train[0:2000])
    f.savefig('./tsne_words.png')

    print "training xgboost classifier..."
    xg_train = xgb.DMatrix(train_data_features, label=sentiment_train)
    
    num_round = 10 #number of rounds for boosting
    params = {'booster':'gbtree', 'max_depth': 10, 'objective':'binary:logistic', \
              'eval_metric': 'auc', 'silent': 1}
    
    bst = xgb.train(params, xg_train, num_round)  
 

    clean_test_reviews = []
    num_reviews = test["review"].size
    print "cleaning test data..."
    for i in xrange(0, num_reviews):
        if ((i+1)%5000 == 0):
            print "review %d of %d" %(i+1, num_reviews)
        clean_test_reviews.append(review_to_words(test['review'][i]))
    
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray() 
    xg_test = xgb.DMatrix(test_data_features)

    #create a submission
    result = bst.predict(xg_test)
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv("word2vec.csv", index=False, quoting=3) 
