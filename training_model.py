import os
import numpy as np
from model import NUM_EPOCH, get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from helpers import *
from constants import *

def training_model(data_path, model_path):
    actions = get_word_actions(data_path) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(len(actions))
    
    try:
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
        # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCH, batch_size=8)
        model.fit(X, y, epochs=NUM_EPOCH)
        model.summary()
    except Exception as e:
        print(f"Error: {e}")
        
    model.save(model_path)
    
if __name__ == "__main__":
    save_path = os.path.join(ROOT_PATH, "models")
    model_path = os.path.join(save_path, MODEL_NAME)
    
    training_model(DATA_PATH, model_path)
    