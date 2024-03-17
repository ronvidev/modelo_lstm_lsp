import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *
from utils import Utils

def create_keypoints(frames_path, hdf_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `hdf_path`
    '''
    data = pd.DataFrame([])
    
    with Holistic() as model_holistic:
        word_id = os.path.basename(frames_path)
        print(f'Creando keypoints de "{word_id}"...')
        
        sample_list = os.listdir(frames_path)
        sample_count = len(sample_list)
        n_sample = 1
        
        for sample_name in sample_list:
            sample_path = os.path.join(frames_path, sample_name)
            frames_count = len(os.listdir(sample_path))
            
            model_num = "7" if frames_count in range(MIN_LENGTH_FRAMES, 7) else "12" if frames_count in range(8, 12) else "18"
                
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, model_num, keypoints_sequence)
                
            print(f"{n_sample}/{sample_count}", end="\r")
            n_sample += 1
                
    try:
        data.to_hdf(hdf_path, key="data", mode="w")
        # TODO: actualizar JSON has_keypoints
        print(f"Keypoints creados! ({n_sample} muestras)")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    utils = Utils()
    
    # Crea la carpeta `keypoints` en caso no exista
    create_folder(KEYPOINTS_PATH)
    
    # GENERAR TODAS LAS PALABRAS
    words_id = utils.get_words_id()
    
    # GENERAR PARA UNA PALABRA O CONJUNTO
    # words_id = ["hola-izq"]
    
    for word_id in words_id:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        hdf_path = os.path.join(KEYPOINTS_PATH, f"{word_id}.h5")
        create_keypoints(word_path, hdf_path)