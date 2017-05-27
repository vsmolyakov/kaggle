import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from nltk.corpus import stopwords
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split

import xgboost as xgb

np.random.seed(0)

def word_match_share(row):
    
    stops = set(stopwords.words('english'))
    
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/float(len(q1words) + len(q2words))
    return R

def tfidf_word_match_share(row):
    
    stops = set(stopwords.words('english'))
    
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / float(count + eps)

def jaccard(row):
    wic = set(str(row['question1'])).intersection(set(str(row['question2'])))
    uw = set(str(row['question1'])).union(str(row['question2']))
    if len(uw) == 0:
        uw = [1]
    return (len(wic) / len(uw))

def wc_diff(row):
    return abs(len(str(row['question1'])) - len(str(row['question2'])))
        
def q1_freq(row):
    return(len(q_dict[str(row['question1'])]))

def q2_freq(row):
    return(len(q_dict[str(row['question2'])]))
    
def q1_q2_intersect(row):
    return(len(set(q_dict[str(row['question1'])]).intersection(set(q_dict[str(row['question2'])]))))


#load data
train_df = pd.read_csv("./data/train.csv")
test_df  = pd.read_csv("./data/test.csv")

train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)

#global dictionaries
print "creating global weights dictionary..."
eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count, eps) for word, count in counts.items()}

print "creating global questions dictionary..."
ques = pd.concat([train_df[['question1', 'question2']], test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])

def main():
    
    #data analysis
    qids = pd.Series(train_df['qid1'].tolist() + train_df['qid2'].tolist())
    print "total number of question pairs of training: ", len(train_df)
    print "percent duplicate pairs: ", round(train_df['is_duplicate'].mean()*100.0,2)
    print "total number of questions in the training data: ", len(np.unique(qids))
    print "number of questions that appear multiple times: ", np.sum(qids.value_counts() > 1)

    plt.figure()
    plt.hist(qids.value_counts(), bins=50)
    plt.yscale('log')
    plt.title('Log-Histogram of question appearance counts')
    plt.xlabel('Number of occurences of question')
    plt.ylabel('Number of questions')
    plt.show()
    
    train_qs = pd.Series(train_df['question1'].tolist() + train_df['question2'].tolist()).astype(str)
    test_qs = pd.Series(test_df['question1'].tolist() + test_df['question2'].tolist()).astype(str)

    dist_train = train_qs.apply(len)
    dist_test = test_qs.apply(len)    

    plt.figure()
    plt.hist(dist_train, bins=200, range=[0,200], color='b', normed=True, label='train')
    plt.hist(dist_test, bins=200, range=[0,200], color='r', normed=True, alpha=0.5, label='test')
    plt.title('Normalized Histogram of Question Length')
    plt.xlabel('number of characters')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
    
    print "mean-train: %2.2f, std-train: %2.2f, mean-test: %2.2f, std-test: %2.2f" %(dist_train.mean(), dist_train.std(), dist_test.mean(), dist_test.std())
 
    dist_train = train_qs.apply(lambda x: len(x.split(' ')))
    dist_test = test_qs.apply(lambda x: len(x.split(' ')))
    
    plt.figure()
    plt.hist(dist_train, bins=50, range=[0,50], color='b', normed=True, label='train')
    plt.hist(dist_test, bins=50, range=[0,50], color='r', normed=True, alpha=0.5, label='test')
    plt.title('Normalized Histogram of Word Count')
    plt.xlabel('number of words')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
        
    #feature engineering
    train_word_match = train_df.apply(word_match_share, axis=1, raw=True)
    print 'word match AUC:', roc_auc_score(train_df['is_duplicate'], train_word_match)
    
    print 'Most common words and weights:'
    print sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10]
    print 'Least common words and weights: '
    print sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
    
    tfidf_train_word_match = train_df.apply(tfidf_word_match_share, axis=1, raw=True)
    tfidf_train_word_match = tfidf_train_word_match.fillna(0)
    print 'TFIDF AUC:', roc_auc_score(train_df['is_duplicate'], tfidf_train_word_match)

    train_jaccard = train_df.apply(jaccard, axis=1, raw=True)
    print 'jaccard AUC:', roc_auc_score(train_df['is_duplicate'], train_jaccard)            

    train_wc_diff = train_df.apply(wc_diff, axis=1, raw=True)    
    train_q1_q2_intersect = train_df.apply(q1_q2_intersect, axis=1, raw=True)       
                                                                                                                                                                    
    print "creating training data..."
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_train['word_match'] = train_word_match
    X_train['tfidf_word_match'] = tfidf_train_word_match    
    X_train['jaccard'] = train_jaccard
    X_train['wc_diff'] = train_wc_diff
    X_train['q1_q2_intersect'] = train_q1_q2_intersect

    print "creating test data..."
    X_test['word_match'] = test_df.apply(word_match_share, axis=1, raw=True)
    X_test['tfidf_word_match'] = test_df.apply(tfidf_word_match_share, axis=1, raw=True)
    X_test['jaccard'] = test_df.apply(jaccard, axis=1, raw=True)
    X_test['wc_diff'] = test_df.apply(wc_diff, axis=1, raw=True)
    X_test['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
    
    X_test = X_test.fillna(0)
    y_train = train_df['is_duplicate'].values
        
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4
    params['silent'] = 1
        
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    eval_list = [(dtrain,'train'), (dvalid, 'valid')]                                          
    
    print "running xgboost..."
    num_boost_round = 400
    xg = xgb.train(params, dtrain, num_boost_round, eval_list, early_stopping_rounds=50, verbose_eval=10)

    dtest = xgb.DMatrix(X_test)
    y_pred_xgb = xg.predict(dtest)
    
    #create submission
    test_idx = test_df['test_id']
    xgb_submission = pd.DataFrame({'test_id':test_idx,'is_duplicate':y_pred_xgb})
    xgb_submission.to_csv("xgb_submission.csv", index=False)
                        
                        
if __name__ == '__main__':
    
    main()
