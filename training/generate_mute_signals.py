#!/usr/bin/python3

from pydub import AudioSegment
from pydub.silence import  detect_nonsilent
from pydub.silence import split_on_silence
from pydub.silence import detect_silence
import subprocess
import datetime
import os
from os import path


def extractVocalSignals(file, output_dir):
    extract_sound_cmd = [
         'spleeter',
         'separate',  # overwrite if exists
         '-i', file,  # input
         '-p', 'spleeter:2stems',  # convert to mono
        '-d', '600',
        '-o', output_dir
     ]
    output = subprocess.call(extract_sound_cmd)


basepath = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(basepath,'video')

input_video_fie_name = 'AlltheBrightPlaces.mp4'

input_video_file = os.path.join(input_video_path, input_video_fie_name)
spleeter_output_dir = os.path.join(basepath,'output')

print('Step-1 of 2: Extracting vocal signals using spleeter and output will be available at : '+ spleeter_output_dir)
extractVocalSignals(input_video_file, spleeter_output_dir)

spleeter_vocal_outputDirPath = os.path.join(spleeter_output_dir, input_video_fie_name.split('.')[0]) 
outputMuteSignalDirPath = os.path.join(spleeter_vocal_outputDirPath, 'mute_signals')
os.makedirs(outputMuteSignalDirPath) 

vocal_wav_file = spleeter_vocal_outputDirPath + '/vocals.wav'

print('Step-2 of 2: Extracting mute signals from Spleeter vocal file i.e.: '+ vocal_wav_file)
audio_signal = AudioSegment.from_file(vocal_wav_file, "wav")

silent_ranges = detect_silence(audio_signal, min_silence_len=1000, silence_thresh=-33)
index = 0;
for silence_range in silent_ranges:
    split_mute_output_file_name = outputMuteSignalDirPath + '/' + str(index) + '_mutesignal.wav'
    index= index +1 
    audio_signal_split = audio_signal[silence_range[0]: silence_range[1]]
    audio_signal_split.export(split_mute_output_file_name, format="wav")

print('Extracted mute signals are present at  '+ outputMuteSignalDirPath)