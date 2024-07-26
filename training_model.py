import os
import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=50):
    word_ids = get_word_ids(KEYPOINTS_PATH) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.fit(X, y, epochs=epochs)
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    