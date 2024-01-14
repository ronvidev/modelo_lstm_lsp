import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from model import NUM_EPOCH, get_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from helpers import get_actions, get_sequences_and_labels
from constants import MAX_LENGTH_FRAMES, NAME_MODEL

def training_model(name_model, data_path, save_path):
    actions = get_actions(data_path)
    sequences, labels = get_sequences_and_labels(actions, data_path)
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_train, _, y_train, _ = train_test_split(X, y)
    
    model = get_model(len(actions))
    model.fit(X_train, y_train, epochs=NUM_EPOCH)
    model.summary()
    model.save(f'{save_path}/{name_model}.keras')

if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root, "data")
    save_path = os.path.join(root, "models")
    
    training_model(NAME_MODEL, data_path, save_path)