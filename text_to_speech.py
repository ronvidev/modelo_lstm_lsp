from gtts import gTTS
import os
import pygame
from time import sleep

def text_to_speech(text):
    tts = gTTS(text=text, lang='es')
    filename = "speech.mp3"
    tts.save(filename)
    
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        sleep(1)

    pygame.mixer.quit()
    pygame.quit()

    os.remove(filename)

if __name__ == "__main__":
    text = "texto"
    text_to_speech(text)
