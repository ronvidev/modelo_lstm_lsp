import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *

def create_keypoints(frames_path, save_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `save_path`
    '''
    data = pd.DataFrame([])
    
    with Holistic() as model_holistic:
        sample_list = os.listdir(frames_path)
        n_sample = 1
        for sample_name in sample_list:
            sample_path = os.path.join(frames_path, sample_name)
            # TODO: Hacer que se agreguen en un mismo archivo las 3 variantes
            if len(os.listdir(sample_path)) in range(*RANGE_FRAMES):
                keypoints_sequence = get_keypoints(model_holistic, sample_path)
                data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)
                print(f"{n_sample}", end="\r")
                n_sample += 1
    try:
        data.to_hdf(save_path, key="data", mode="w")
        print(f"Keypoints creados! ({n_sample} samples)")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    
    # for word_name in os.listdir(FRAME_ACTIONS_PATH):
    #     word_path = os.path.join(FRAME_ACTIONS_PATH, word_name)
    #     create_folder(DATA_PATH)
    #     hdf_path = os.path.join(DATA_PATH, f"{word_name}.h5")
    #     print(f'Creando keypoints de "{word_name}"...')
    #     create_keypoints(word_path, hdf_path)

        # GENERAR SOLO DE UNA PALABRA
        word_id = "hola-izq"
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        hdf_path = os.path.join(DATA_PATH, f"{word_id}.h5")
        create_keypoints(word_path, hdf_path)
