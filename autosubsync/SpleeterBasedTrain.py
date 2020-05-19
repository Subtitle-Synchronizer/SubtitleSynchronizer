import os
import sys
import argparse
from os import path
from os.path import dirname
import pandas as pd
import numpy as np

from autosubsync import features, model
from autosubsync.preprocessing import import_sound


def cli(packaged_model=False):
    p = argparse.ArgumentParser(description="Pass arguments to train the vocal classification model with Spleeter based processing.")
    p.add_argument('train_files_path', help='Path of train audio files')
    p.add_argument('mode_file_path', help='Path of model file where to be saved')
    basepath = dirname(path.dirname(__file__))
    print(basepath)
    dfObj = pd.DataFrame(columns=['features', 'label'])
    labels = ['1','0']

    all_x = []
    all_y = []
    for labelVal in labels:
        trainingAudioDirPath = os.path.join(basepath, 'audioFiles',labelVal)
        for filename in os.listdir(trainingAudioDirPath):
            if filename.endswith(".flac") or filename.endswith(".wav"):
                input_audio_file = os.path.join(trainingAudioDirPath, filename)

                sound_data = import_sound(input_audio_file)
                training_x = features.computeSoundFeatures(sound_data)

                training_y = np.full(shape=(len(training_x),1),fill_value=labelVal, dtype=np.int)
                training_y = np.hstack(training_y)

                #d = pd.DataFrame(zero_data, columns=feature_list)
                #training_y = np.array([labelVal] * len(training_x)).T
                all_x.append(training_x)
                all_y.append(training_y)

    all_x = np.vstack(all_x)
    all_y = np.hstack(all_y)

    #np.save('spleeter_features_file_X_0519.npy', all_x)
    #np.save('spleeter_features_file_Y_0519.npy', all_y)

    #all_x = np.load('spleeter_features_file_X.npy', allow_pickle=True)
    #all_y = np.load('spleeter_features_file_Y.npy', allow_pickle=True)

    print('len of X: ', len(all_x))
    print('len of Y: ', len(all_y))
    print(all_y.shape)
    print(all_x.shape)

    trained = model.trainWithSpleeterOutput(all_x, all_y) # model.load('trained-model-original.bin')  #
    y_scores = model.predict(trained, all_x)
    y_converted_scores = []
    for score in y_scores:
        if score > 0.5:
            y_converted_scores.append(1)
        else:
            y_converted_scores.append(0)

    print(y_converted_scores)
    print(np.unique(y_converted_scores))
    target_file = "trained.model.spleeter.0519.bin"
    print('serializing model to ' + target_file)
    #model.save(trained, target_file)


if __name__ == '__main__':
    # Entry point for running from repository root folder
    sys.path.append('.')
    cli(packaged_model=True)
