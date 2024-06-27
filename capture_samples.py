import os
import cv2
import shutil
import numpy as np
import pygetwindow as gw
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH, RED_COLOR , BLUE_COLOR


def handle_action(action, *args, **kwargs):
    try:
        action(*args, **kwargs)
    except Exception as e:
            print(f"{type(e).__name__}: {str(e)}")
            return None


def bring_window_to_front(window_name):
    try:
        window = gw.getWindowsWithTitle(window_name)[0]
        window.activate()
    except IndexError:
        print(f"No se encontró la ventana con el nombre: {window_name}")
    except gw.PyGetWindowException:
        pass


def capture_samples(path, margin_frame=2, min_cant_frames=5, video_device=0):
    """
    Captura de muestras para una palabra.

    Parameters:
    - path (str): Ruta de la carpeta de la palabra.
    - margin_frame (int): Cantidad de frames que se ignoran al comienzo y al final.
    - min_cant_frames (int): Cantidad de frames mínimos para cada muestra.
    - video_device (int): Dispositivo de video a usar (default es 0 para webcam).

    Controles:
    - 'Esc' para salir y eliminar la muestra actual.
    - 'Enter' para terminar el registro de la muestra actual y comenzar una nueva.
    - ' ' (espacio) para pausar y reanudar la captura.
    - 'q' para salir y cerrar el programa.
    """
    create_folder(path)

    cant_sample_exist = len(os.listdir(path))
    quantity_sample = cant_sample_exist
    new_sample_requested = False
    capturing = True
    count_frame = 0
    frames = []

    with Holistic() as holistic_model:
        video = cv2.VideoCapture(video_device)

        if not video.isOpened():
            raise RuntimeError("No se pudo abrir el dispositivo de video.")

        window_name = f'Toma de muestras para "{os.path.basename(path)}"'

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic_model)

            if capturing and there_hand(results):
                count_frame += 1
                if count_frame > margin_frame:
                    cv2.putText(image, 'Capturando...', FONT_POS, FONT, FONT_SIZE, BLUE_COLOR)
                    frames.append(np.asarray(frame))
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    quantity_sample +=1
                    frames = frames[:-margin_frame]
                    output_folder = os.path.join(path, f"sample_{quantity_sample}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)

                frames = []
                count_frame = 0
                cv2.putText(image, 'Listo para capturar...', FONT_POS, FONT, FONT_SIZE, RED_COLOR, 3)
                cv2.putText(image, f'Muestra numero: {quantity_sample}', (10, 100), FONT, FONT_SIZE, RED_COLOR, 3)

            draw_keypoints(image, results)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', image)
            bring_window_to_front(window_name)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                break
            elif key == ord('q'):
                video.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord(' '):
                capturing = not capturing
            elif key == 13:
                new_sample_requested = True
                break


        video.release()
        cv2.destroyAllWindows()

    return new_sample_requested


def main():
    while True:
        sample_label = input('Ingresa la etiqueta de la muestra: ')
        sample_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, sample_label)
        new_sample_requested = capture_samples(sample_path, video_device=2)
        if new_sample_requested:
            continue
        delete_sample = input('¿Desea eliminar la muestra y reiniciar el muestreo? (y/n): ')
        if delete_sample.lower() == 'y':
            handle_action(shutil.rmtree, sample_path)
        else:
            break


if __name__ == "__main__":
    main()
