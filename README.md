# kaggle
Kaggle Competitions

### speech

The goal of tensorflow speech competition is to recognize simple voice commands spoken by thousands of different people. The training dataset consists of 65K one-second long utterances of 30 short words.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/speech/figures/speech_merged.png" />
</p>

The figure above (left) shows a sample waveform of the word "yes" along with its spectogram. In this competition, an ensemble of multi-input CNN and LSTM models was trained to process images of spectograms and MFCC features in order to classify the spoken word into one of 30 categories. The models were trained end-to-end on image representation of speech. Data augmentation and re-sampling techniques were used to improve robustness of speech recognition. The figure above (right) shows the training and validation accuracy and cross-entropy loss as a result of training on a single GTX Titan GPU. We see no signs of over-fitting suggesting that we could use higher model capacity or additional features to improve model accuracy.

References:  
*https://www.kaggle.com/c/tensorflow-speech-recognition-challenge*  


### cdiscount

The goal of cdiscount competition is to classify product images into more than 5000 categories. The training dataset consists of over 15 million images at 180x180 resolution stored in bson format. 

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/cdiscount/figures/cdiscount_merged.png" />
</p>

The figure above (left) shows a sample of product images with titles corresponding to categories. We use transfer learning in this competition by fine-tuning ResNet50 trained on ImageNet to the product dataset with over 5000 categories. At chance level, the probability of predicting the correct category is 1/5000 = 0.0002. The figure above (right) shows the validation accuracy of 0.5 after only 8 epochs trained on a GTX Titan GPU, a significant increase! To achieve a boost in classification accuracy, we can try an ensemble of diverse and accurate classifiers: ResNet50, InceptionV3, VGG16, VGG19 and others with different initialization, exposed to different subsets of the dataset. Multi-GPU training in this case is highly desirable.

References:  
*https://www.kaggle.com/c/cdiscount-image-classification-challenge*  

## quora

In quora question pairs challenge, the goal is to identify duplicate questions. Two approaches were taken to solve this problem: feature engineering with xgboost and a neural network classifier.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/quora/figures/dataset.png" />
</p>

The figure above shows four examples from the dataset. To measure similarity between question pairs, features such as shared words and cosine similarity between question encodings were used. Additional architectures, such as Siamese RNN, can help increase the area under ROC curve for the binary classification (similar vs non-similar question pair) task. 

References:  
*https://www.kaggle.com/c/quora-question-pairs*  

## sberbank

The goal of Sberbank Russian Housing Market competition is to predict the log housing price based on a number of real estate features and economic indicators. An ensemble of lasso regression, random forest, xgboost and MLP was used to predict the housing price.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/sberbank/figures/sberbank_merged.png" />
</p>

The figure above explores the patterns in housing dataset. In particular, we can see relationship between price and housing area (left), price and build year as well as floor number (middle), and sales volume and room count (right).

References:  
*https://www.kaggle.com/c/sberbank-russian-housing-market*  

## two sigma

In the two sigma financial modelling challenge, we are given fundamental, technical and derived indicators for a time-varying portfolio of assets and our goal is to predict time-series 'y'. 

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/two_sigma/figures/two_sigma_merged.png" />
</p>

The figure above shows the mean and standard deviation of 'y' as a function of time. To predict the time series, an ensemble of a linear model (ridge regression) and a tree model (xgboost) was used.

References:  
*https://www.kaggle.com/c/two-sigma-financial-modeling*  

## Stack Exchange

The goal of this competition is to predict a stack exchange tag for the physics site based on titles, text and tags of stack exchange questions from six different sites. An unsupervised approach is taken here, in which word frequencies are computed for titles and text and the top ten in each category are merged to yield the final list of tags sorted by the frequency of occurence.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/stack_exchange/figures/robotics.png" />
</p>

The figure above shows a wordcloud of tags for the robotics stack exchange. The physics stack exchange tags were predicted based on purely the frequency of physics titles and text words.

References:  
*https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags*  


## Facial Keypoints

The objective of this competition is to predict facial keypoint positions on face images. This can be used as a building block for tracking faces in images and videos, analysing facial expressions, and facial recognition. The keypoints include eye centers and corners, nose tip, mouth corners and others.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/keypoints/figures/keypoints_merged.png" />
</p>

The figure above shows the predicted keypoints on 16 test images. The results look good after only 100 training epochs.

References:  
*https://www.kaggle.com/c/facial-keypoints-detection*  

## word2vec

In this competition, the goal is to predict a sentiment label for a dataset of 50,000 IMDB movie reviews. The sentiment is binary so that IMDB ratings < 5 result in a sentiment score of 0 and ratings >= 7 have a sentiment score of 1. The XGboost algorithm was trained on a term-document matrix where each row is a tf-idf vector representation of a review.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/word2vec/figures/word2vec_merged.png" />
</p>

The figure above shows a t-SNE embedding of the word vectors for the first 2000 reviews. The size of the word feature space was set to 5000. With a tree depth of 10 and 10 boosting iterations, XGboost algorithm achieves low test error rate.

References:  
*https://www.kaggle.com/c/word2vec-nlp-tutorial*  

## Sentiment Prediction

The goal of this competition is to learn and predict sentiment of movie reviews from the Rotten Tomatoes dataset. LSTM recurrent neural net (RNN) was used to predict the sentiment based on pre-processed text. The pre-processing included tokenization, stop-word removal and word stemming. The word tokens were converted to token Ids sequences that were padded and used as an input to LSTM.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/sentiment/figures/LSTM_chain.png" />
</p>

The figure above shows LSTM architecture with three hidden units, highlighting its ability to process sequential data. 

References:  
*https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews*  


## Denoising

In the document denoising competition the goal is to remove background noise as a preprocessing step for an Optical Character Recognition (OCR) system.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/denoising/figures/denoising_merged.png" />
</p>

The figure above shows training data consisting of a noisy and denoised images. A fully connected and a convolutional autoencoders were used to denoise the image. Increasing the depth and the width of the autonencoders enables higher test accuracy.

References:  
*https://www.kaggle.com/c/denoising-dirty-documents*  


## Digit Recognizer

In the digit recognizer competition the goal is to classify hand drawn digits into one of ten classes. The training data are 28x28 grayscale images of digits and the associated labels.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/digits/figures/lenet_cnn.png" />
</p>

The figure above shows LeNet Convolutional Neural Network (CNN) architecture that consists of two 2D convolutional and max-pooling layers followed by a fully connected layer and a soft-max classifier. Increasing the number of training epochs enables better test accuracy of the LeNet CNN.

References:  
*https://www.kaggle.com/c/digit-recognizer*  


## Bike Sharing

In the bike sharing competition we want to predict the bike demand given temporal data of bike checkouts and associated features such as season, temperature, humidity, weather and others.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/bike_sharing/figures/bike_demand_merged.png" />
</p>

The figure above shows hourly bike demand ploted for every day of the week (left) and the feature ranking extracted by random forest regressor (right). We can see that during the weekdays the peak demand is at 8am and 5pm, while on the weekend the demand rises between 12 noon and 3 pm. It's interesting to note that the most predictive feature is the hour, which was extracted from the timestamp, followed by temperature, humidity and day of the week.

References:  
*https://www.kaggle.com/c/bike-sharing-demand*  

## Forest Cover

In the forest cover competition the idea is to use a set of measured cartographic variables to classify forest cover into one of 7 categories. The measurements include elevation, aspect, slope, distance to water, wilderness area and soil type among others.

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/forest_cover/figures/svm.png" />
</p>

The figure above shows an SVM classifier with radial basis function (RBF) kernel decision boundary for two classes where support vectors are circled. The features can be ranked by their predictive power and it was found that elevation and soil type can discriminate well between different forest cover types. Grid search cross-validation was used to find the optimum penalty parameter for the SVM error term.

References:  
*https://www.kaggle.com/c/forest-cover-type-prediction*  

## Titanic

In the titanic competition you are given a task of predicting the probability of survival based on the training data that includes age, gender, ticket price, passenger class etc...

<p align="center">
<img src="https://github.com/vsmolyakov/kaggle/blob/master/titanic/figures/random_forrest.png" />
</p>

The figure above shows the random forrest classifier trained on a subset of data supervised by the survival signal in the training set. In comparison to SVM, logistic regression and K-NN classifier, random forrest with 100 trees produced highest test accuracy.

References:  
*https://www.kaggle.com/c/titanic*  

### Datasets

Check out dataset visualizations with Tableau at:
*https://public.tableau.com/profile/vadim.smolyakov#!/*
 
### Dependencies

Python 2.7
