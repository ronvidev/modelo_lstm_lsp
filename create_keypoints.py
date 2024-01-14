import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
import pandas as pd
from mediapipe.python.solutions import holistic
from helpers import get_keypoints, insert_keypoints_sequence

def create_keypoints(path, save_path):
    '''
    ### CREAR KEYPOINTS PARA UNA PALABRA
    Recorre la carpeta de frames de la palabra y guarda sus keypoints en `save_path`
    '''
    data = pd.DataFrame([])
    
    with holistic.Holistic() as model:
        for n_sample, sample_name in enumerate(os.listdir(path), 1):
            sample_path = os.path.join(path, sample_name)
            keypoints_sequence = get_keypoints(model, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    data.to_hdf(save_path, key="data", mode="w")

if __name__ == "__main__":
    root = os.getcwd()
    words_path = os.path.join(root, "action_samples")
    data_path = os.path.join(root, "data")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # GENERAR LOS KEYPOINTS DE TODAS LAS PALABRAS
        for word_name in os.listdir(words_path):
            word_path = os.path.join(words_path, word_name)
            hdf_path = os.path.join(data_path, f"{word_name}.h5")
            print(f'Creando keypoints de "{word_name}"...')
            create_keypoints(word_path, hdf_path)
            print(f"Keypoints creados!")
            
        # GENERAR SOLO DE UNA PALABRA
        # word_name = "hola"
        # word_path = os.path.join(words_path, word_name)
        # hdf_path = os.path.join(data_path, f"{word_name}.h5")
        # create_keypoints(word_path, hdf_path)
        # print(f"Keypoints creados!")
