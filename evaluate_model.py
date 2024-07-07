import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, format_sentences, get_actions, mediapipe_detection, there_hand
from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME
from text_to_speech import text_to_speech

def evaluate_model(model, threshold=0.7):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    actions = get_actions(DATA_PATH)
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))
            
            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
            
            elif count_frame >= MIN_LENGTH_FRAMES:
                last_frames = kp_sequence[-MAX_LENGTH_FRAMES:]
                res = model.predict(np.expand_dims(last_frames, axis=0))[0]

                if res[np.argmax(res)] > threshold:
                    sent = actions[np.argmax(res)]
                    sentence.insert(0, sent)
                    text_to_speech(sent)
                    sentence, repe_sent = format_sentences(sent, sentence, repe_sent)
                    
                count_frame = 0
                kp_sequence = []
            
            cv2.rectangle(frame, (0,0), (640, 35), (245, 117, 16), -1)
            cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            
            draw_keypoints(frame, results)
            cv2.imshow('Traductor LSP', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
        video.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    evaluate_model(lstm_model)
