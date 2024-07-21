import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from helpers import get_word_ids, get_sequences_and_labels
from constants import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def generate_confusion_matrix():
    word_ids = get_word_ids(KEYPOINTS_PATH)
    test_sequences, test_labels = [], []

    for model_num in MODEL_NUMS:
        test_sequences_num, test_labels_num = get_sequences_and_labels(word_ids, model_num)
        test_sequences_num = pad_sequences(test_sequences_num, maxlen=int(model_num), padding='pre', truncating='post', dtype='float32')
        
        test_sequences.extend(test_sequences_num)
        test_labels.extend(test_labels_num)

    all_predictions = []
    all_true_labels = []

    models = [load_model(model_path) for model_path in MODELS_PATH]

    for seq, true_label in zip(test_sequences, test_labels):
        seq_length = len(seq)
        if seq_length <= 7:
            model = models[0]
            seq = pad_sequences([seq], maxlen=7, padding='pre', truncating='post', dtype='float32')[0]
        elif seq_length <= 12:
            model = models[1]
            seq = pad_sequences([seq], maxlen=12, padding='pre', truncating='post', dtype='float32')[0]
        else:
            model = models[2]
            seq = pad_sequences([seq], maxlen=18, padding='pre', truncating='post', dtype='float32')[0]
        
        res = model.predict(np.expand_dims(seq, axis=0))[0]
        predicted_label = np.argmax(res)
        
        all_predictions.append(predicted_label)
        all_true_labels.append(true_label)

    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=word_ids)
    
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()