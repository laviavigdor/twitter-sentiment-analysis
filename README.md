# Twitter Sentiment Analysis

## Overview
This package will perform sentiment analysis on tweets or similar short texts.
Pre-trained word embeddings from GloVe are used as a frozen 
input to Keras, afterwhich a CNN learns and predicts on the classification.

### Resulting accuracy: *~ 79%*

## Usage

1. Download the GloVe embedings for twitter, unzip into a /glove directory.
        
        wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
        
2. Download the twitter sentiment data set into / directory.

        wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip

3. Run *[Optional since model.h5 and weights.h5 have been provided]* 

        python train.py

    Which will create a model.h5 and weights.h5 files.
4. Run 

        echo "This is a sample tweet to predict on" | python predict.py
    Or
        
        cat file-containing-one-tweet-per-line.txt | python predict.py
        
## Reference
* [Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
* [Twitter sentiment data set](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
* [GloVe embeddings](http://nlp.stanford.edu/projects/glove/)