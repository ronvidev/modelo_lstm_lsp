from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from constants import LENGTH_KEYPOINTS

def get_model(max_length_frames, output_length: int):
    model = Sequential()
    
    model.add(LSTM(128, return_sequences=True, input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    # Segunda capa LSTM con regularización L2
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.5))
    
    # Capa totalmente conectada con regularización L2
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# def get_model(max_length_frames, output_length: int):
#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_length_frames, LENGTH_KEYPOINTS), kernel_regularizer=l2(0.001)))
#     # model.add(Masking(mask_value=0, input_shape=(None, LENGTH_KEYPOINTS)))
#     model.add(LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dropout(0.5))
#     model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
#     # model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
#     # model.add(Dropout(0.5))
#     model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
#     model.add(Dense(output_length, activation='softmax'))
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model