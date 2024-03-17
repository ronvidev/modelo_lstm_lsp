from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

NUM_EPOCH = 35

def get_model(max_length_frames:int, output_length: int):
    print(max_length_frames)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model