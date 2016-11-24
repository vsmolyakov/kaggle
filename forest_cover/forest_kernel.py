import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv as csv

from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

if __name__ == "__main__":
    
    #load data
    train_df = pd.read_csv('./data/train.csv', sep=',', header=0)
    test_df = pd.read_csv('./data/test.csv', sep=',', header=0)
    train_df.dropna()

    #remove irrelevant columns    
    train_df = train_df.drop('Id', axis=1)
    test_idx = test_df['Id'].values 
    test_df = test_df.drop('Id', axis=1)
    
    y_data = train_df['Cover_Type'].values
    train_df = train_df.drop('Cover_Type', axis=1)
        
    #standardize non-categorical data
    idx = 10 
    cols = list(train_df.columns.values)[:idx]
    train_df[cols] = StandardScaler().fit_transform(train_df[cols])
    test_df[cols] = StandardScaler().fit_transform(test_df[cols])

    X_data = train_df.values            
    X_train,X_test,y_train,y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=123)
    
    #grid search for SVM hyperparameters
    svm_parameters = [{'kernel': ['rbf'], 'C': [1,10,100,1000]}]                
    clf = GridSearchCV(SVC(), svm_parameters, cv=3, verbose=2)
    clf.fit(X_train, y_train)    
    clf.best_params_
    clf.grid_scores_
    
    #SVM training
    C_opt = 10
    clf = SVC(C=C_opt, kernel='rbf')
    clf.fit(X_data, y_data)    
    clf.n_support_

    #predict test data    
    X_test_data = test_df.values    
    y_pred = clf.predict(X_test_data)
    
    #write out predictions
    predictions_file = open("forest_pred.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["Id","Cover_Type"])
    open_file_object.writerows(zip(test_idx, y_pred))
    predictions_file.close()       
    