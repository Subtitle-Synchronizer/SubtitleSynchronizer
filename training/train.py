import os
import sys
from os import path
from os.path import dirname
import numpy as np

from autosubsync import features, model
from autosubsync.preprocessing import import_sound


def load_features():
    basepath = dirname(path.dirname(__file__))
    labels = ['1', '0']

    all_x = []
    all_y = []
    for labelVal in labels:
        training_audio__dir_path = os.path.join(basepath, 'audioFiles', labelVal)
        for filename in os.listdir(training_audio__dir_path):
            if filename.endswith(".flac") or filename.endswith(".wav"):
                input_audio_file = os.path.join(training_audio__dir_path, filename)

                sound_data = import_sound(input_audio_file)
                training_x = features.compute_train_features(sound_data)

                training_y = np.full(shape=(len(training_x), 1), fill_value=labelVal, dtype=np.int)
                training_y = np.hstack(training_y)

                all_x.append(training_x)
                all_y.append(training_y)

    all_x = np.vstack(all_x)
    all_y = np.hstack(all_y)

    # np.save('spleeter_features_file_X_0519.npy', all_x)
    # np.save('spleeter_features_file_Y_0519.npy', all_y)
    return all_x, all_y


def train():
    all_x, all_y = load_features()
    trained = model.train_with_spleeter_output(all_x, all_y)
    target_file = "subtitlesynchronizer.trained.model.bin"
    print('serializing model to ' + target_file)
    model.save(trained, target_file)


if __name__ == '__main__':
    # Entry point for running from repository root folder
    sys.path.append('.')
    train()
