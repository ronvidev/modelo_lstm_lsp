import cv2
import numpy as np
import os
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from helpers import *
from constants import *

def evaluate_model(models, threshold=0.6):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    actions = ["words"]
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            _, frame = video.read()

            image, results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                kp_sequence.append(extract_keypoints(results))
                count_frame += 1
                
            elif count_frame >= MIN_LENGTH_FRAMES:
                if count_frame <= 7:
                    print("carga modelo 7")
                    kp_sequence = pad_secuences(kp_sequence, 7)
                    model = models[0]
                    
                elif count_frame <= 12:
                    print("carga modelo 12")
                    kp_sequence = pad_secuences(kp_sequence, 12)
                    model = models[1]
             
                else:
                    print("carga modelo 18")
                    kp_sequence = pad_secuences(kp_sequence, 18)
                    model = models[2]                
                 
                res = model.predict(np.expand_dims(kp_sequence, axis=0))[0]
                
                if res[np.argmax(res)] > threshold:
                    sent = words_format[actions[np.argmax(res)]]
                    sentence.insert(0, sent)
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
    model_path_7 = os.path.join(MODELS_PATH, f"actions_7.keras")
    model_path_12 = os.path.join(MODELS_PATH, f"actions_12.keras")
    model_path_18 = os.path.join(MODELS_PATH, f"actions_18.keras")
    lstm_models = [load_model(model_path_7), load_model(model_path_12), load_model(model_path_18)]
    evaluate_model(lstm_models)