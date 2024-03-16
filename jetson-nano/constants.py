import cv2
import os

# SETTINGS
MIN_LENGTH_FRAMES = 5

# PATHS
ROOT_PATH = os.getcwd()
MODELS_PATH = os.path.join(ROOT_PATH, "models")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1.5
FONT_POS = (5, 30)

# TEMP
words_format = {
  "hola-der": "Hola!",
  "hola-izq": "Hola!",
  "como_estas": "como estas?",
  "adios": "Adios!",
  "buenos_dias": "Buenos dias!"
}
