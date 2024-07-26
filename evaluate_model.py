import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import *
from constants import *
from text_to_speech import text_to_speech


def evaluate_model(src=None, threshold=0.4):
    count_frame = 0
    kp_sequence, sentence = [], []
    word_ids = get_word_ids(KEYPOINTS_PATH)
    model = load_model(MODEL_PATH)
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                kp_sequence.append(extract_keypoints(results))
                count_frame += 1
            
            elif count_frame >= MIN_LENGTH_FRAMES + 4:
                kp_sequence = pad_secuences(kp_sequence, int(MODEL_FRAMES))
                res = model.predict(np.expand_dims(kp_sequence, axis=0))[0]
                
                # print(res[np.argmax(res)])
                if res[np.argmax(res)] > threshold:
                    word_id = word_ids[np.argmax(res)].split('-')[0]
                    sent = words_text.get(word_id)
                    sentence.insert(0, sent)
                    text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                    
                count_frame = 0
                kp_sequence = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                
                draw_keypoints(frame, results)
                cv2.imshow('Traductor LSP', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()
