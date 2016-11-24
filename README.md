# kaggle
Kaggle Competitions

### Description

**Forest Cover**

In the forest cover competition the idea is to use a set of measured cartographic variables to classify forest cover into one of 7 categories. The measurements include elevation, aspect, slope, distance to water, wilderness area and soil type among others.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/forest_cover/figures/svm.png" />
</p>

The figure above shows an SVM classifier with radial basis function (RBF) kernel decision boundary for two classes where support vectors are circled. The features can be ranked by their predictive power and it was found that elevation and soil type can discriminate well between different forest cover types. Grid search cross-validation was used to find the optimum penalty parameter for the SVM error term.

References:  
*https://www.kaggle.com/c/forest-cover-type-prediction*  

**Titanic**

In the titanic competition you are given a task of predicting the probability of survival based on the training data that includes age, gender, ticket price, passenger class etc...

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/titanic/figures/random_forrest.png" />
</p>

The figure above shows the random forrest classifier trained on a subset of data supervised by the survival signal in the training set. In comparison to SVM, logistic regression and K-NN classifier, random forrest with 100 trees produced highest test accuracy.

References:  
*https://www.kaggle.com/c/titanic*  

 
### Dependencies

Python 2.7
