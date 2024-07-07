from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model