import os
import numpy as np
from model import NUM_EPOCH, get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from helpers import get_actions, get_sequences_and_labels
from constants import MAX_LENGTH_FRAMES, MODEL_NAME

def training_model(data_path, model_path):
    actions = get_actions(data_path) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES,padding='post', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(len(actions))
    model.fit(X, y, epochs=NUM_EPOCH)
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root, "data")
    save_path = os.path.join(root, "models")
    model_path = os.path.join(save_path, MODEL_NAME)
    
    training_model(data_path, model_path)
    