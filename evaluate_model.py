import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, format_sentences, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME, ROOT_PATH

def evaluate_model(model, threshold=0.7):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    actions = get_actions(DATA_PATH)
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            _, frame = video.read()

            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))
            
            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
                
            else:
                if count_frame >= MIN_LENGTH_FRAMES:
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]

                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                        sentence, repe_sent = format_sentences(sent, sentence, repe_sent)
                        
                    count_frame = 0
                    kp_sequence = []
            
            cv2.rectangle(image, (0,0), (640, 35), (245, 117, 16), -1)
            cv2.putText(image, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            
            draw_keypoints(image, results)
            cv2.imshow('Traductor LSP', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
        video.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    evaluate_model(lstm_model)
