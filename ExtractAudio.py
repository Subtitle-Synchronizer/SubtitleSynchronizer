import os
from os import path
import subprocess

basepath = path.dirname(__file__)
videoDirPath = os.path.join(basepath,'videos')
for filename in os.listdir(videoDirPath):
    if filename.endswith(".mp4"):
        input_video_file = os.path.join(videoDirPath, filename)
        fileNameWithOutExt = os.path.splitext(filename)[0]
        output_sound_file = os.path.join(videoDirPath, fileNameWithOutExt+'.flac')
        print(os.path.join(videoDirPath, filename))
        extract_sound_cmd = [
             'ffmpeg',
             '-y',  # overwrite if exists
             '-loglevel', 'error',
             '-i', input_video_file,  # input
             '-ac', '1',  # convert to mono
             output_sound_file
         ]
        subprocess.call(extract_sound_cmd)

