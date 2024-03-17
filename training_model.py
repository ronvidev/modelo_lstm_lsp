import numpy as np
from model import NUM_EPOCH, get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from helpers import *
from constants import *
from utils import Utils

def training_model(model_path, model_num):
    utils = Utils()
    words_id = utils.get_words_id()
    
    sequences, labels = get_sequences_and_labels(words_id, model_num)
    
    sequences = pad_sequences(sequences, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(int(model_num), len(words_id))
    
    try:
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
        # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=NUM_EPOCH, batch_size=8)
        model.fit(X, y, epochs=NUM_EPOCH)
        model.summary()
        
    except Exception as e:
        print(f"Error: {e}")
        
    model.save(model_path)
    
if __name__ == "__main__":
    utils = Utils()
    for model_num in utils.data.models:
        model_path = utils.get_model_path_by_num(model_num)
        training_model(model_path, model_num)
    