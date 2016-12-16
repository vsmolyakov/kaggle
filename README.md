# kaggle
Kaggle Competitions

### Description

**word2vec**

In this competition, the goal is to predict a sentiment label for a dataset of 50,000 IMDB movie reviews. The sentiment is binary so that IMDB ratings < 5 result in a sentiment score of 0 and ratings >= 7 have a sentiment score of 1. The XGboost algorithm was trained on a term-document matrix where each row is a tf-idf vector representation of a review.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/word2vec/figures/word2vec_merged.png" />
</p>

The figure above shows a t-SNE embedding of the word vectors for the first 2000 reviews. The size of the word feature space was set to 5000. With a tree depth of 10 and 10 boosting iterations, XGboost algorithm achieves low test error rate.

References:  
*https://www.kaggle.com/c/word2vec-nlp-tutorial*  


**Sentiment Prediction**

The goal of this competition is to learn and predict sentiment of movie reviews from the Rotten Tomatoes dataset. LSTM recurrent neural net (RNN) was used to predict the sentiment based on pre-processed text. The pre-processing included tokenization, stop-word removal and word stemming. The word tokens were converted to token Ids sequences that were padded and used as an input to LSTM.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/sentiment/figures/LSTM_chain.png" />
</p>

The figure above shows LSTM architecture with three hidden units, highlighting its ability to process sequential data. 

References:  
*https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews*  


**Denoising**

In the document denoising competition the goal is to remove background noise as a preprocessing step for an Optical Character Recognition (OCR) system.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/denoising/figures/denoising_merged.png" />
</p>

The figure above shows training data consisting of a noisy and denoised images. A fully connected and a convolutional autoencoders were used to denoise the image. Increasing the depth and the width of the autonencoders enables higher test accuracy.

References:  
*https://www.kaggle.com/c/denoising-dirty-documents*  


**Digit Recognizer**

In the digit recognizer competition the goal is to classify hand drawn digits into one of ten classes. The training data are 28x28 grayscale images of digits and the associated labels.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/digits/figures/lenet_cnn.png" />
</p>

The figure above shows LeNet Convolutional Neural Network (CNN) architecture that consists of two 2D convolutional and max-pooling layers followed by a fully connected layer and a soft-max classifier. Increasing the number of training epochs enables better test accuracy of the LeNet CNN.

References:  
*https://www.kaggle.com/c/digit-recognizer*  


**Bike Sharing**

In the bike sharing competition we want to predict the bike demand given temporal data of bike checkouts and associated features such as season, temperature, humidity, weather and others.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/bike_sharing/figures/bike_demand_merged.png" />
</p>

The figure above shows hourly bike demand ploted for every day of the week (left) and the feature ranking extracted by random forest regressor (right). We can see that during the weekdays the peak demand is at 8am and 5pm, while on the weekend the demand rises between 12 noon and 3 pm. It's interesting to note that the most predictive feature is the hour, which was extracted from the timestamp, followed by temperature, humidity and day of the week.

References:  
*https://www.kaggle.com/c/bike-sharing-demand*  

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
