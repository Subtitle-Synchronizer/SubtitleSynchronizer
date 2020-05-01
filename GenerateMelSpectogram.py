from IPython.display import Audio
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import numpy as np
import os
import glob
import shutil
import librosa.display
from pathlib import Path 
import random
import librosa

def audioClassification(audio, sr, refID, melSpectogramPath):
    split_audio_duration = 3*sr
    startIndex = 0
    image_predictions = {} 

    if len(audio) > split_audio_duration:
        iteration = int(np.ceil(len(audio)/split_audio_duration))
        for i in range(iteration):
            endIndex = startIndex + split_audio_duration
            if endIndex > len(audio):
                endIndex = len(audio)
            split_audio = audio[startIndex: endIndex]
            startIndex = endIndex + 1
            imagePath = generateMelspectrogram(split_audio, sr, melSpectogramPath, refID, i)
            #image_predictions[str(i)] = self.predictAudioImageClass(imagePath)
    else:
        split_audio = audio[startIndex: len(audio)]
        self.generateMelspectrogram(split_audio, sr, melSpectogramPath, refID, 0)

    return image_predictions

def generateMelspectrogram(split_audio, sr, melSpectogramPath, refID, i):
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    fileName = os.path.join(melSpectogramPath, refID + '_' + str(i) + '.png')

    S = librosa.feature.melspectrogram(y=split_audio, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(fileName, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    return fileName

basepath = path.dirname(__file__)
melSpectogramPath = os.path.join(basepath,'mel-spectogram-images')    
audio_path = os.path.join(basepath,'audio') 

files= list(Path(audio_path).glob('*.wav'))
for audio_file in files:
    file = str(audio_file).split('\\')[-1].split('.')[0]
    print('processing file: ', file)
    audio, sr = librosa.load(audio_file)
    audioClassification(audio, sr, file, melSpectogramPath)
