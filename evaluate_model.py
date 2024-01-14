import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, NAME_MODEL

def evaluate_model(model, threshold=0.7):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    
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
                    # print(res[np.argmax(res)])
                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                        
                        # LOGICA DE REPETICIONES DE PALABRA
                        if len(sentence) > 1:
                            if sent in sentence[1]:
                                repe_sent += 1
                                sentence.pop(0)
                                sentence[0] = f"{sent} (x{repe_sent})"
                            else:
                                repe_sent = 1
                        
                    count_frame = 0
                    kp_sequence = []
            
            cv2.rectangle(image, (0,0), (640, 35), (245, 117, 16), -1)
            cv2.putText(image, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            save_txt('outputs/sentences.txt', '\n'.join(sentence))
            
            draw_keypoints(image, results)
            cv2.imshow('Traductor LSP', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
        video.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root, "data")
    actions = get_actions(data_path)
    lstm_model = load_model(f'{root}/models/{NAME_MODEL}.keras')
    
    evaluate_model(lstm_model)