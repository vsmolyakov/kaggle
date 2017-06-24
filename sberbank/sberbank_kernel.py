import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

np.random.seed(0)

def main():
    
    # load data
    macro_cols = ["balance_trade", "average_provision_of_build_contract",
                  "micex_rgbi_tr", "micex_cbi_tr", "mortgage_value", "mortgage_rate", "cpi", "ppi",
                  "income_per_cap", "rent_price_4+room_bus", "balance_trade_growth"]
        
    train_df = pd.read_csv("./data/train.csv", parse_dates = ['timestamp'])
    test_df  = pd.read_csv("./data/test.csv", parse_dates=['timestamp'])    
    macro_df = pd.read_csv("./data/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
            
    train_df['price_doc'] = np.log1p(train_df['price_doc'])

    #nan values
    train_na = (train_df.isnull().sum() / len(train_df)) * 100
    train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)

    #missing data visualization
    plt.figure()
    plt.xticks(rotation='90')
    sns.barplot(x=train_na.index,y=train_na)
    plt.title("percent missing data by feature")
    plt.ylabel("% missing")
    plt.tight_layout()
    plt.show()

    #data quality
    train_df.loc[train_df['state'] == 33, 'state'] = train_df['state'].mode().iloc[0]
    train_df.loc[train_df['build_year'] == 20052009, 'build_year'] = 2007
    train_df.loc[train_df['build_year'] == 0, 'build_year'] = np.nan
    train_df.loc[train_df['build_year'] == 1, 'build_year'] = np.nan  
    train_df.loc[train_df['build_year'] == 3, 'build_year'] = np.nan
    train_df.loc[train_df['build_year'] == 71, 'build_year'] = np.nan
    
    # truncate the extreme values in price_doc
    ulimit = np.percentile(train_df.price_doc.values, 99)
    llimit = np.percentile(train_df.price_doc.values, 1)
    train_df['price_doc'].loc[train_df['price_doc']>ulimit] = ulimit
    train_df['price_doc'].loc[train_df['price_doc']<llimit] = llimit
    
    #missing data imputation, merging, feature engineering and label encoding
    train_df['env'] = 'train'
    test_df['env'] = 'test'
    test_idx = test_df['id']
    
    train_df.drop(['id'], axis=1, inplace=True)    
    test_df.drop(['id'], axis=1, inplace=True)    
    test_df['price_doc'] = 0
        
    #train_df = train_df.dropna()  #drop training rows
    #macro_df = macro_df.dropna()
    tdf_med = test_df.median()    
    tdf_product_mode = stats.mode(test_df['product_type'])[0][0]            
    test_df = test_df.fillna(tdf_med)  #fill-in test rows
    test_df['product_type'] = test_df['product_type'].fillna(tdf_product_mode)    

    all_df = pd.concat([train_df, test_df])                                                                        
    all_df = pd.merge_ordered(all_df, macro_df, on='timestamp', how='left')
    adf_med = all_df.median()                
    all_df = all_df.fillna(adf_med)
    print "num nans: ", all_df.isnull().sum().sum()

    #add month and day of week    
    all_df["year"] = all_df.timestamp.dt.year
    all_df['month'] = all_df.timestamp.dt.month
    all_df['dow'] = all_df.timestamp.dt.dayofweek
        
    #add month-year
    month_year = (all_df.timestamp.dt.month + all_df.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    all_df['month_year_cnt'] = month_year.map(month_year_cnt_map)    
    
    #add week-year count
    week_year = (all_df.timestamp.dt.weekofyear + all_df.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    all_df['week_year_cnt'] = week_year.map(week_year_cnt_map)
            
    # num of floor from top
    all_df["floor_from_top"] = all_df["max_floor"] - all_df["floor"]

    # difference between full area and living area
    all_df["extra_sq"] = all_df["full_sq"] - all_df["life_sq"]
    
    # age of building
    all_df["age_of_building"] = all_df["build_year"] - all_df["year"]


    for f in all_df.columns:
        if all_df[f].dtype == 'object' and f is not 'env':
            lbl = LabelEncoder()
            lbl.fit(all_df[f])
            all_df[f] = lbl.transform(all_df[f])

    all_df.drop(['timestamp'], axis=1, inplace=True)
    y_train = all_df.loc[all_df['env']=='train', 'price_doc']    
    all_df.drop(['price_doc'], axis=1, inplace=True)        
    X_train = all_df[all_df['env']=='train']
    X_test = all_df[all_df['env']=='test']
    X_train.drop(['env'], axis=1, inplace=True)
    X_test.drop(['env'], axis=1, inplace=True)                

    print "X_train: ", np.shape(X_train)
    print "y_train: ", np.shape(y_train)    
    print "X_test: ", np.shape(X_test)        

    #data visualization
    idx = train_df[train_df['full_sq'] > 2000].index  #remove outliers
    plt.figure()
    plt.scatter(train_df['full_sq'].drop(idx), train_df['price_doc'].drop(idx), c='r', alpha=0.5)
    plt.title("Price vs Area")
    plt.xlabel("Total Area, square meters")
    plt.ylabel("Price")
    plt.show()
    
    #room counts
    plt.figure()
    sns.countplot(train_df['num_room'])
    plt.title("Number of Rooms")
    plt.xlabel("Rooms")
    plt.ylabel("Count")
    plt.show()
    
    train_df.groupby('product_type')['price_doc'].median()
    
    #build year histogram
    plt.figure()
    plt.xticks(rotation='90')
    idx = train_df[(train_df['build_year'] <= 1691) | (train_df['build_year'] >= 2018)].index
    by_df = train_df.drop(idx).sort_values(by=['build_year'])
    sns.countplot(by_df['build_year'])
    plt.title("Distribution of Build Year")
    plt.show()
    
    #mean price vs build year
    plt.figure()
    by_price = by_df.groupby('build_year')[['build_year', 'price_doc']].mean()
    sns.regplot(x='build_year', y='price_doc', data=by_price, scatter=False, order=3, truncate=True)
    plt.plot(by_price['build_year'], by_price['price_doc'], color='r')
    plt.title('Mean Price by Year of Build')
    plt.xlabel("Year")
    plt.ylabel("Mean Price")
    plt.show()
    
    #sales volume
    plt.figure()
    ts_vc = train_df['timestamp'].value_counts()
    plt.bar(left=ts_vc.index, height=ts_vc)
    plt.title("Sales Volume")
    plt.ylabel("Number of transactions")
    plt.show()
    
    #volume by sub-area
    plt.figure()
    sa_vc = train_df['sub_area'].value_counts()
    sa_vc = pd.DataFrame({'sub_area':sa_vc.index, 'count':sa_vc.values})
    sns.barplot(x="count", y="sub_area", data=sa_vc.iloc[0:10], orient='h')
    plt.title("Number of Transactions by District (Top 10)")
    plt.tight_layout()    
    plt.show()
    
    #price by floor
    plt.figure()
    fl_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
    sns.pointplot(fl_df.floor.values, fl_df.price_doc.values, alpha=0.8, color='b')
    plt.title("Median Price vs Floor Number")
    plt.xlabel("Floor Number")
    plt.ylabel("Median Price")
    plt.xticks(rotation='vertical')    
    plt.show()
    
    #random forest    
    print "running random forest..."
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
        
    fi = list(zip(X_train.columns, rf.feature_importances_))
    fi_sorted = sorted(fi, key=lambda x: x[1], reverse=True)
        
    print 'feature importances:'
    for i in range(10):
        print "%1.5f \t %s" % (fi_sorted[i][1], fi_sorted[i][0])

    y_pred_rf = np.expm1(rf.predict(X_test))    
    rf_submission = pd.DataFrame({'id':test_idx,'price_doc':y_pred_rf})
    rf_submission.to_csv("rf_submission.csv", index=False)
    
    #Lasso CV
    print "running lasso..."
    lasso = LassoCV(alphas=[100,10,1,0.1,0.01,0.001])
    lasso.fit(X_train, y_train)
    y_pred_lasso = np.expm1(lasso.predict(X_test))
    lasso_submission = pd.DataFrame({'id':test_idx,'price_doc':y_pred_lasso})
    lasso_submission.to_csv("lasso_submission.csv", index=False)
                                    
    #xgboost
    print "running xgboost..."
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=X_train.columns)
    dtest = xgb.DMatrix(X_test, feature_names=X_test.columns)    
    
    xgb_params = {
    'booster' : 'gbtree',
    'eta': 0.06,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
    }
    eval_list = [(dtrain,'train')]                                          

    num_boost_round = 100
    xg = xgb.train(xgb_params, dtrain, num_boost_round, eval_list)
    y_pred_xgb = np.expm1(xg.predict(dtest))
    xgb_submission = pd.DataFrame({'id':test_idx,'price_doc':y_pred_xgb})
    xgb_submission.to_csv("xgb_submission.csv", index=False)

    #MLP
    print "running MLP..."
    model = Sequential()
    
    model.add(Dense(64, input_dim = X_train.shape[1], init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())    
    model.add(Dense(16, init = 'he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
    model.fit(X_train.values,y_train.values, batch_size=32, nb_epoch=20, verbose=1, validation_split=0.2)
    y_pred_mlp = model.predict(X_test.values, batch_size=32, verbose=0)
    y_pred_mlp = np.reshape(y_pred_mlp, y_pred_mlp.shape[0])
    y_pred_mlp = np.expm1(y_pred_mlp)

    #catch/replace -1 
    y_pred_mlp_med = np.median(y_pred_mlp)
    for idx, item in enumerate(y_pred_mlp):
        if item == -1:
            y_pred_mlp[idx] = y_pred_mlp_med
    
    mlp_submission = pd.DataFrame({'id':test_idx,'price_doc':y_pred_mlp})
    mlp_submission.to_csv("mlp_submission.csv", index=False)
   
    #voting
    y_pred_vote = (y_pred_rf + y_pred_lasso + y_pred_xgb)/3.0
    vote_submission = pd.DataFrame({'id':test_idx,'price_doc':y_pred_vote})
    vote_submission.to_csv("vote_submission.csv", index=False)


if __name__ == "__main__":    
    main()
