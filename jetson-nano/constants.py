import cv2
import os

# SETTINGS
MIN_LENGTH_FRAMES = 5

# PATHS
ROOT_PATH = os.getcwd()
DATA_JSON_PATH = os.path.join(ROOT_PATH, "data", "data.json")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1.5
FONT_POS = (5, 30)