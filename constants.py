import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5

# MIN_FRAMES = MIN_LENGTH_FRAMES
# MAX_LENGTH_FRAMES = 7
# MIN_FRAMES = 8
# MAX_LENGTH_FRAMES = 12
MIN_FRAMES = 13
MAX_LENGTH_FRAMES = 18

# RANGE_FRAMES = (MIN_FRAMES, MAX_LENGTH_FRAMES+1)
RANGE_FRAMES = (MIN_FRAMES, 30)
LENGTH_KEYPOINTS = 1662

MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
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