import os
import cv2
from mediapipe.python.solutions.holistic import Holistic

from helpers import draw_keypoints, mediapipe_detection

def video_to_frame(video_path):
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(video_path)
        output_filename = os.path.join(os.path.dirname(video_path), f"output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, 30.0, (1280, 720))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for n_frame in range(total_frames):
            frame = video.read()[1]
            image, results = mediapipe_detection(frame, holistic_model)
            cv2.rectangle(image, (0,0), (1280, 720), (0, 255, 0), -1)
            draw_keypoints(image, results)
            out.write(image)
        
if __name__ == "__main__":
    video_path = r"F:\CarpetasW\Descargas\original.mp4"
    video_to_frame(video_path)