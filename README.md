# Twitter Sentiment Analysis

## Overview
This package will perform sentiment analysis on tweets or similar texts.
Pre-trained word embeddings from GloVe are used as a frozen input to Keras, aftwerwhich a CNN learns and predicts on the classification.

### GloVe embeddings:
* info: http://nlp.stanford.edu/projects/glove/
* data: 
    
        wget http://nlp.stanford.edu/data/glove.twitter.27B.zip

### Twitter sentiment data set:
* info: http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/
* data: 
    
        wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip

### Resulting accuracy: *~ 79%*

## Usage

1. Download theglove embedings for twitter, into a /glove directory.
2. Download the twitter sentiment data set into / directory.
3. Run 

        python train.py

    Which will create a model.h5 and weights.h5 files.
4. Run 

        echo "This is a sample tweet to predict on" | python predict.py
    Or
        
        cat file-containing-one-tweet-per-line.txt | python predict.py