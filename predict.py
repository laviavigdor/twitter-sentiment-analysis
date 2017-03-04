from __future__ import print_function
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import os

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # str(random.randint(0, 15))

    X_raw = []
    for line in sys.stdin:
        X_raw.append(line)

    X, word_index = tokenize_data(X_raw)

    model = load_model('model.h5')
    model.load_weights("weights.h5")

    predictions = model.predict(x=X, batch_size=128)

    for index, txt in enumerate(X_raw):
        is_positive = predictions[index][1] >= 0.5
        status_txt = "Positive" if is_positive else "Negative"
        print("[",status_txt,"] ", txt)


def tokenize_data(X_raw):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    word_index = tokenizer.word_index
    X_processed = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return X_processed, word_index


if __name__ == "__main__":
    main()