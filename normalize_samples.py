import cv2
import numpy as np
import os
import shutil
from constants import *

def read_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.jpg'):
            frame = cv2.imread(os.path.join(directory, filename))
            frames.append(frame)
    return frames

def interpolate_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count == target_frame_count:
        return frames
    
    indices = np.linspace(0, current_frame_count - 1, target_frame_count)
    interpolated_frames = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        interpolated_frame = cv2.addWeighted(frames[lower_idx], 1 - weight, frames[upper_idx], weight, 0)
        interpolated_frames.append(interpolated_frame)
    
    return interpolated_frames

def normalize_frames(frames, target_frame_count=15):
    current_frame_count = len(frames)
    if current_frame_count < target_frame_count:
        return interpolate_frames(frames, target_frame_count)
    elif current_frame_count > target_frame_count:
        step = current_frame_count / target_frame_count
        indices = np.arange(0, current_frame_count, step).astype(int)[:target_frame_count]
        return [frames[i] for i in indices]
    else:
        return frames

def process_directory(word_directory, target_frame_count=15):
    for sample_name in os.listdir(word_directory):
        sample_directory = os.path.join(word_directory, sample_name)
        if os.path.isdir(sample_directory):
            frames = read_frames_from_directory(sample_directory)
            normalized_frames = normalize_frames(frames, target_frame_count)
            clear_directory(sample_directory)
            save_normalized_frames(sample_directory, normalized_frames)

def save_normalized_frames(directory, frames):
    for i, frame in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(directory, f'frame_{i:02}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if __name__ == "__main__":
    # GENERAR PARA TODAS LAS PALABRAS
    word_ids = [word for word in os.listdir(os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH))]
    
    # GENERAR PARA UNA PALABRA O CONJUNTO
    # word_ids = ["buenos_dias"]
    
    for word_id in word_ids:
        word_path = os.path.join(FRAME_ACTIONS_PATH, word_id)
        if os.path.isdir(word_path):
            print(f'Normalizando frames para "{word_id}"...')
            process_directory(word_path, MODEL_FRAMES)
    
    # sample_directory = r"E:\Data\LSP Project\RED NEURONAL\frame_actions\buenos_dias\sample_240113195007489206"
    # frames = read_frames_from_directory(sample_directory)
    # normalized_frames = normalize_frames(frames, 15)
    # clear_directory(sample_directory)
    # save_normalized_frames(sample_directory, normalized_frames)