import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels
from constants import *

def training_model(model_path, epochs=500):
    word_ids = get_word_ids(WORDS_JSON_PATH ) # ['word1', 'word2', 'word3]
    
    sequences, labels = get_sequences_and_labels(word_ids)
    
    sequences = pad_sequences(sequences, maxlen=int(MODEL_FRAMES), padding='pre', truncating='post', dtype='float16')
    
    X = np.array(sequences)
    y = to_categorical(labels).astype(int) 
    
    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    
    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, callbacks=[early_stopping])
    
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
    