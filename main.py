import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from PyQt5.uic import loadUi

import numpy as np
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from helpers import *
from constants import *
from text_to_speech import text_to_speech


class VideoRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('mainwindow.ui', self)
        
        self.is_recording = False
        self.capture = cv2.VideoCapture(0)
        
        self.init_lsp()
        
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update frame every 30ms
    
    def init_lsp(self):
        self.holistic_model = Holistic()
        self.kp_sequence, self.sentence = [], []
        self.count_frame = 0
        self.models = [load_model(model_path) for model_path in MODELS_PATH]
    
    def update_frame(self):
        word_ids = get_word_ids(KEYPOINTS_PATH)
        ret, frame = self.capture.read()
        if not ret: return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.is_recording:
            results = mediapipe_detection(frame, self.holistic_model)
            
            if there_hand(results):
                self.kp_sequence.append(extract_keypoints(results))
                self.count_frame += 1
                
            elif self.count_frame >= MIN_LENGTH_FRAMES:
                if self.count_frame <= 7:
                    print("carga modelo 7")
                    self.kp_sequence = pad_secuences(self.kp_sequence, 7)
                    model = self.models[0]
                    
                elif self.count_frame <= 12:
                    print("carga modelo 12")
                    self.kp_sequence = pad_secuences(self.kp_sequence, 12)
                    model = self.models[1]
             
                else:
                    print("carga modelo 18")
                    self.kp_sequence = pad_secuences(self.kp_sequence, 18)
                    model = self.models[2]
                
                res = model.predict(np.expand_dims(self.kp_sequence, axis=0))[0]
                
                if res[np.argmax(res)] > 0.7:
                    print(res[np.argmax(res)])
                    sent = word_ids[np.argmax(res)].replace('_', ' ').split('-')[0].upper()
                    self.sentence.insert(0, sent)
                    text_to_speech(sent) # ONLY LOCAL (NO SERVER)
                    
                self.count_frame = 0
                self.kp_sequence = []
            
            self.lbl_output.setText(", ".join(self.sentence))
            draw_keypoints(image, results)
        
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        
        scaled_qImg = qImg.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.lbl_video.setPixmap(QPixmap.fromImage(scaled_qImg))

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            # self.video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            # self.video_writer.release()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
    
    def closeEvent(self, event):
        self.capture.release()
        # if self.is_recording:
            # self.video_writer.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoRecorder()
    window.show()
    # window.showFullScreen()
    sys.exit(app.exec_())
