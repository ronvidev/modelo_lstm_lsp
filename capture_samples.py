import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH
from datetime import datetime


def capture_samples(path, margin_frame=1, min_cant_frames=5, delay_frames=3):
    '''
    ### CAPTURA DE MUESTRAS PARA UNA PALABRA
    Recibe como parámetro la ubicación de guardado y guarda los frames
    
    `path` ruta de la carpeta de la palabra \n
    `margin_frame` cantidad de frames que se ignoran al comienzo y al final \n
    `min_cant_frames` cantidad de frames minimos para cada muestra \n
    `delay_frames` cantidad de frames que espera antes de detener la captura después de no detectar manos
    '''
    create_folder(path)
    
    count_frame = 0
    frames = []
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
            else:
                if len(frames) >= min_cant_frames + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    frames = frames[: - (margin_frame + delay_frames)]
                    today = datetime.now().strftime('%y%m%d%H%M%S%f')
                    output_folder = os.path.join(path, f"sample_{today}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                
                recording, fix_frames = False, 0
                frames, count_frame = [], 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
            
            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "buenos_dias"
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
