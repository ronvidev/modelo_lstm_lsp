Este es un modelo de una red neuronal que traduce Lengua de Señas Peruana (LSP) a texto (y voz). Utilicé MediaPipe para obtener los puntos de la seña y para el entrenamiento usé TensorFlow y Keras.

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
1. Capturar las muestras con capture_samples.py
2. Generar los .h5 (keypoints) de cada palabra con create_keypoints.py
3. Entrenar el modelo con training_model.py
4. Realizar pruebas con evaluate_model.py

## Observaciones
La información que está en Data fue creada en Python 3.11.3.
Así que se recomienda volver a generar las muestras y los keypoints con la version que tengas.

Por ahora se ha probado con las versiones 3.11.3 y 3.8.10, les dejo los modulos que tienen que instalar en cada version:

3.11.3:
pip install tensorflow, mediapipe, opencv-python, pytables, gtts, pygame, pandas

3.8.10:
pip install tensorflow, mediapipe, opencv-python, tables, gtts, pygame, pandas

## Video de la explicación del código:
https://youtu.be/3EK0TxfoAMk 
