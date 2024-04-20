import os

from flask import Flask, request

from werkzeug.utils import secure_filename
from process_video import process_video
from translate_lsp import evaluate_model
import numpy as np
import cv2
from PIL import Image


app = Flask(__name__)


@app.route('/')
def hello():
    return 'Servidor encendido'


@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    try:
        # Obtener los datos binarios de la imagen enviada desde Flutter
        frame_data = request.data
        
        imagen = Image.frombytes('L', (1280, 720), frame_data)
        frame_np = np.array(imagen)
        frame = cv2.cvtColor(frame_np, cv2.COLOR_BAYER_BG2BGR)
        
        
        
        # # Guardar el frame como un archivo en la carpeta de frames
        filename = os.path.join(os.getcwd(), f"frames_tmp/frame_{request.headers['Frame-Number']}.jpg")
        cv2.imwrite(filename, frame)

        # En este ejemplo, simplemente devolvemos un mensaje de éxito
        return 'Frame recibido correctamente', 200
    except Exception as e:
        print(str(e))
        # Manejar cualquier error que pueda ocurrir durante el proceso
        return str(e), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    file_name = secure_filename(video_file.filename)
    tmp_file = os.path.join('tmp', file_name)
    video_file.save(tmp_file)
    
    resp = evaluate_model(process_video(tmp_file))
    
    return " - ".join(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

