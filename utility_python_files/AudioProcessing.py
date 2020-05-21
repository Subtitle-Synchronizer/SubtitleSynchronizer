import sys
from moviepy.editor import *
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

audio_data = 'D:/Users/344929/git/aila/python/videos/video_d4bc3b9b/audio.wav'
x,sr = librosa.load(audio_data)
print(x.shape, sr)
y=librosa.amplitude_to_db(abs(x))
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)
refDBVal = np.max(y)        #finding the max value of the signal to pass as a reference to top_db parameter to find mute sections
nonMuteSections = librosa.effects.split(y, top_db=refDBVal)  # audio above 20dB
print(1)