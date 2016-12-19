import kagglegym
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

train = observation.train
median_values = train.median(axis=0)
train.fillna(median_values, inplace=True)

#feat = [name for name in observation.train.columns if name not in ['id', 'timestamp', 'y']]
feat = ['technical_20', 'fundamental_53', 'technical_30', 'technical_27', 'derived_0',\
        'fundamental_42', 'fundamental_48', 'technical_21', 'technical_24', 'fundamental_11',\
        'fundamental_44', 'technical_19', 'technical_13', 'technical_17', 'fundamental_9']

dtrain = xgb.DMatrix(train[feat].values, label = train['y'].values)

params = {}
params['booster']  = 'gbtree'
params['objective'] = 'reg:linear'
params['max_depth'] = 8
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8
params['silent'] = 1
params['eval_metric'] = 'rmse'
num_round = 300
eval_list  = [(dtrain,'train')]

print('training xgb model...')
bst = xgb.train(params, dtrain, num_round, eval_list)

print('training ridge regression...')
lr = Ridge()
lr.fit(train[feat].values, train['y'].values)

while True:
    observation.features.fillna(median_values, inplace=True)
    dtest = xgb.DMatrix(observation.features[feat].values)
    xgb_pred = bst.predict(dtest)
    lr_pred = lr.predict(observation.features[feat].values)  
    observation.target.y = 0.5*xgb_pred+0.5*lr_pred

    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break