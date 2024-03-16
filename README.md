Este es un modelo de una red neuronal que traduce Lengua de Señas Peruana (LSP) a texto (y voz). Utilicé MediaPipe para obtener los puntos de la seña y para el entrenamiento usé TensorFlow y Keras.

## DEPENDENCIAS
Por ahora se ha probado con las versiones 3.9.0 (Windows 11) y 3.8.10 (Ubuntu 20.04), les dejo los módulos que tienen que instalar en cada version:

3.9.0 (Windows 11):
pip install tensorflow==2.10.1, mediapipe==0.10.11, numpy==1.26.4, tables==3.9.2, opencv-python==4.9.0.80, pandas==2.2.1, protobuf==3.20.3, keras==2.10.0, h5py==3.10.0, flatbuffers==24.3.7, gTTS==2.5.1, pygame==2.5.2

3.8.10 (Ubuntu 20.04):
pip install tensorflow, mediapipe, opencv-python, tables, gtts, pygame, pandas

Para evitar incovenientes con las versiones, se recomienda instalar todo en un entorno virtual.

## SCRIPTS PRINCIPALES
- capture_samples.py → captura las muestras y las ubica en la carpeta frame_actions.
- create_keypoints.py → crea los keypoints que se usarán en el entrenamiento.
- training_model.py → entrena la red neuronal.
- evaluate_model.py → donde se realiza la prueba de la red neuronal.

## SCRIPTS SECUNDARIOS
- model.py → aquí se ajusta el modelo de la red neuronal.
- constants.py → ajustes de la red neuronal.
- helpers.py → funciones que se utilizan en los scripts principales.

## Pasos para probar la red neuronal
1. Capturar las muestras con `capture_samples.py`
2. Generar los .h5 (keypoints) de cada palabra con `create_keypoints.py`
3. Entrenar el modelo con `training_model.py`
4. Realizar pruebas con `evaluate_model.py`

## Observaciones
La información que está en Data fue creada en Python 3.9.0 y TensorFlow 2.10.1 usando GPU. Testeada en Python 3.8.10 con TensorFlow 2.12.0.

El entrenamiento fue hecho en Windows (Python 3.9.0) y la evaluación del modelo en el mismo sistema y Ubuntu 20.04 (Python 3.8.10)

## Video de la explicación del código:
https://youtu.be/3EK0TxfoAMk 
