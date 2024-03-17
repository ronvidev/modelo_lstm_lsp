import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import *
from constants import *
from process_video import process_video
from utils import Utils

def evaluate_model(video_path, threshold=0.7):
    utils = Utils()
    count_frame = 0
    kp_sequence, sentences = [], []
    actions = utils.get_words_id()
    models = [load_model(model_path) for model_path in utils.get_models_path()]
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(video_path)
        
        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                kp_sequence.append(extract_keypoints(results))
                count_frame += 1
                
            elif count_frame >= MIN_LENGTH_FRAMES:
                print(count_frame)
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
                
                sent = utils.get_word_by_id(actions[np.argmax(res)])
                
                # percent = res[np.argmax(res)]
                # if percent > threshold:
                #     pass
                # else:
                #     sent = f"D: {sent} ({round(float(percent), 2)*100}%)"
                    
                print(sent)
                sentences.append(sent)
                    
                count_frame = 0
                kp_sequence = []
                
            # draw_keypoints(image, results)
            # cv2.imshow('Traductor LSP', image)
            # cv2.waitKey(10)
            
        video.release()
        cv2.destroyAllWindows()
        return sentences
    

if __name__ == "__main__":
    video_path = r"F:\CarpetasW\Imágenes\Álbum de cámara\WIN_20240316_22_31_41_Pro.mp4"
    video_path = process_video(video_path, 12)
    resp = evaluate_model(video_path)
    print(resp)
