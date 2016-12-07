import pandas as pd
import csv as csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

np.random.seed(0)

if __name__ == "__main__":
    
    # get titanic & test csv files as a DataFrame
    train_df = pd.read_csv("./data/train.csv")
    test_df  = pd.read_csv("./data/test.csv")
    
    train_df.head()
    train_df.info()
    test_df.info()
    
    #drop un-insightful columns
    train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
    test_df  = test_df.drop(['Name','Ticket','Cabin'], axis=1)
    
    #feature by feature analysis
    #Gender: replace male/female with integers
    mapping = {'male':1,'female':0}
    train_df['Gender'] = train_df['Sex'].map(mapping).astype(int)
    test_df['Gender'] = test_df['Sex'].map(mapping).astype(int)
    train_df = train_df.drop(['Sex'], axis=1)    
    test_df = test_df.drop(['Sex'], axis=1)
    
    #Embarked: fill na and replace C/Q/S with integers
    mapping = {'C':0,'Q':1,'S':2}
    train_df['Embarked'].value_counts()
    train_df['Embarked'] = train_df['Embarked'].fillna("S")
    test_df['Embarked'] = test_df['Embarked'].fillna("S")

    train_df['Embark'] = train_df['Embarked'].map(mapping).astype(int)
    test_df['Embark'] = test_df['Embarked'].map(mapping).astype(int)
    train_df = train_df.drop(['Embarked'], axis=1)    
    test_df = test_df.drop(['Embarked'], axis=1)  
    
    #Age: fill na with medium age
    median_age = train_df['Age'].dropna().median()
    train_df['Age'] = train_df['Age'].fillna(median_age)
    test_df['Age'] = test_df['Age'].fillna(median_age)
    
    #Fare: fill na with medium fare
    median_fare = train_df['Fare'].dropna().median()
    test_df['Fare'] = test_df['Fare'].fillna(median_fare)
        
    #machine learning
    train_data = train_df.values
    test_data = test_df.values
    
    X_train = train_data[:,1:]
    y_train = train_data[:,0]
    
    X_test = test_data[:,1:]
    idx = test_data[:,0].astype(np.int32)
    
    #random forest classifier
    rfc = RandomForestClassifier(n_estimators=100)  
    rfc.fit(X_train, y_train)
    score_rfc = rfc.score(X_train, y_train)
    out_rfc = rfc.predict(X_test)
    print "random forest classifier score: %f" %score_rfc
    
    #logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    score_logreg = logreg.score(X_train, y_train)
    out_logreg = logreg.predict(X_test)
    print "logistic regression score: %f" %score_logreg
        
    #SVM
    svc = SVC()
    svc.fit(X_train, y_train)
    score_svc = svc.score(X_train, y_train)
    out_svc = svc.predict(X_test)    
    print "SVM score: %f" %score_svc    
    
    #knn classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    score_knn = knn.score(X_train, y_train)
    out_knn = knn.predict(X_test)
    print "knn score: %f" %score_knn           

    #voting classifier    
    vclf = VotingClassifier(estimators=[('rf',rfc),('lr',logreg),('svm',svc),('knn',knn)], voting='hard', weights=[2,1,2,1])
    vclf.fit(X_train, y_train)
    out_vclf = vclf.predict(X_test)
            
    #write out predictions 
    predictions_file = open("titanic_pred.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(idx, out_vclf.astype(int)))
    predictions_file.close()         
            