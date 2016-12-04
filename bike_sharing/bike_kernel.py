import numpy as np
import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)

if __name__ == '__main__':
    
    #load data
    train_df = pd.read_csv('./data/train.csv', sep=',', header=0)
    test_df = pd.read_csv('./data/test.csv', sep=',', header=0)
    train_df.dropna()

    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    train_ts = train_df.set_index('datetime')
    test_ts = test_df.set_index('datetime')    
    train_ts['hour'] = train_ts.index.hour
    train_ts['day'] = train_ts.index.weekday_name
    test_ts['hour'] = test_ts.index.hour
    test_ts['day'] = test_ts.index.weekday_name

    #data exploration
    plt.figure()
    weekday_name = np.unique(train_ts['day'])
    for day in weekday_name:
        counts_mean, counts_std = [], []        
        train_day = train_ts.loc[train_ts['day']==day]
        for i in range(24):
            train_day_hour = train_day.loc[train_day['hour']==i]                
            counts_mean.append(train_day_hour['count'].mean())
            counts_std.append(train_day_hour['count'].std())
        #plt.errorbar(range(24), counts_mean, yerr=counts_std, label=day)
        plt.plot(counts_mean, label=day)
    plt.legend(loc=2, prop={'size':15})
    plt.xlabel('hours', fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.show()

    #feature ranking                                                                                                                                                                                                                                                                                                                                                        
    mapping = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
    train_ts['day'] = train_ts['day'].map(mapping).astype(int)
    test_ts['day'] = test_ts['day'].map(mapping).astype(int)    
    train_cols = [col for col in train_ts.columns if col not in ['count','registered','casual']]
    X_all = train_ts[train_cols].values
    y_all = train_ts['count'].values
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
    rf.fit(X_all, y_all)
    
    feature_ranks = rf.feature_importances_
    num_features = len(feature_ranks)
        
    plt.figure()
    width = 0.35
    plt.bar(range(num_features), feature_ranks, width)
    plt.xticks(np.arange(num_features)+width/2.0,tuple(train_cols),rotation='vertical',fontsize=16)
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    
    submission = pd.DataFrame(index=test_ts.index, columns=['count'])
    submission = submission.fillna(0)
    
    #use only past data for training and prediction
    for year in np.unique(test_ts.index.year):
        for month in np.unique(test_ts.index.month):            
            print "Predicting Year: %d, Month: %d" %(year, month)
            test_locs = np.logical_and(test_ts.index.year == year, test_ts.index.month == month)
            test_subset = test_ts[test_locs]
            train_locs = train_ts.index <= min(test_subset.index)
            train_subset = train_ts[train_locs]

            X_train = train_subset[train_cols].values
            y_train = train_subset['count'].values
            
            rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
            rf.fit(X_train, y_train)
            
            X_test = test_subset[train_cols].values                  
            counts = rf.predict(X_test)
            
            submission[test_locs]=np.round(counts.reshape(-1,1)).astype(int)
    

    #create a submission
    submission.to_csv('./bike_pred.csv', index=True, header=True)
            
    