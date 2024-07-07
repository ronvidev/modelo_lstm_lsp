import os
import cv2
import numpy as np

def process_video(input_path, target_fps=12):
    # Ubicación de guardado
    new_name = f"{os.path.splitext(os.path.basename(input_path))[0]}_PROCESADO.mp4"
    output_path = os.path.join(os.path.dirname(input_path), new_name)
    
    # Información del video
    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height, original_width, _ = frame.shape
    
    # Dimensiones de salida - Proporción 3:4
    target_width = 640
    target_height = 480
    
    # Puntos de corte
    if original_width / original_height > target_width / target_height:
        height_fixed = original_height
        width_fixed = int(original_height * (4 / 3))
        left_cut = int((original_width - width_fixed) / 2)
        right_cut = original_width - left_cut
        top_cut = 0
        bottom_cut = original_height
    else:
        width_fixed = original_width
        height_fixed = int(original_width * (4 / 3))
        top_cut = int((original_height - height_fixed) / 2)
        bottom_cut = original_height - top_cut
        left_cut = 0
        right_cut = original_width
    
    # Objeto para colocar los frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

    # Colocar cada frame a la tasa target_fps
    frames_written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Saltarse los frames necesarios para tener la tasa deseada
        if frames_written % round(original_fps / target_fps) == 0:
            
            # Recortar con los puntos de corte
            frame_recortado = frame[top_cut:bottom_cut, left_cut:right_cut]
            frame_cortado = cv2.resize(frame_recortado, (width_fixed, height_fixed))
            height, width, _ = frame_cortado.shape
            
            # Calcular escalado para mantener la proporción
            if width / height > target_width / target_height:
                new_width = target_width
                new_height = int((height * target_width) / width)
            else:
                new_width = int((width * target_height) / height)
                new_height = target_height

            # Calcular posiciones para centrar el video
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Redimensionar el frame
            resized_frame = cv2.resize(frame_cortado, (new_width, new_height))

            # Crear un frame negro del tamaño deseado
            black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            black_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

            # Escribir el frame en el nuevo video
            out.write(black_frame)

        frames_written += 1

    # Limpiar memoria
    cap.release()
    out.release()
    
    return output_path


if __name__ == "__main__":
    input_path = r"F:\CarpetasW\Imágenes\Álbum de cámara\WIN_20240315_20_55_13_Pro.mp4"
    # input_path = r"E:\Data\LSP Project\RED NEURONAL (PYTHON 3.8)\tmp\REC7173119773952296862.mp4"
    process_video(input_path)
    print("¡Conversión completa!")
