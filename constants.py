import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5

LENGTH_KEYPOINTS = 1662

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