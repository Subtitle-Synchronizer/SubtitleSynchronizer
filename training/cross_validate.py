import os
import tempfile
import numpy as np
import pandas as pd
from os import path
from os.path import dirname
from autosubsync import find_transform
from autosubsync import quality_of_fit

from autosubsync import features, model
from autosubsync.preprocessing import import_sound
from sklearn.model_selection import train_test_split

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


def cv_split_by_file(data_meta, data_x):
    files = np.unique(data_meta.file_number)
    np.random.shuffle(files)

    n_train = int(round(len(files)*0.5))
    train_files = files[:n_train]
    print(train_files)

    train_cols = data_meta.file_number.isin(train_files)
    test_cols = ~train_cols
    return data_meta[train_cols], data_x[train_cols,:], data_meta[test_cols], data_x[test_cols,:]


def validate_speech_detection(result_meta):
    print('---- speech detection accuracy ----')

    r = result_meta.groupby('file_number').agg('mean')
    print(r)
    from sklearn.metrics import roc_auc_score
    print('AUC-ROC:', roc_auc_score(result_meta.label, result_meta.predicted_score))
    return r


def test_correct_sync(result_meta, bias=0):
    print('---- synchronization accuracy ----')

    results = []
    for unique_label in np.unique(result_meta.label):
        part = result_meta[result_meta.label == unique_label]
        skew, shift, quality = find_transform.find_transform_parameters(part.label, part.predicted_score, bias=bias)
        skew_error = skew != 1.0
        results.append([skew_error, shift, quality])

    sync_results = pd.DataFrame(np.array(results), columns=['skew_error', 'shift_error', 'quality'])
    print(sync_results)

    print('skew errors:', sync_results.skew_error.sum())
    print('shift RMSE:', np.sqrt(np.mean(sync_results.shift_error**2)))

    return sync_results


if __name__ == '__main__':

    # data_x, data_y = load_features()

    data_x = np.load('spleeter_features_file_X_0519.npy', allow_pickle=True)
    data_y = np.load('spleeter_features_file_Y_0519.npy', allow_pickle=True)

    print('loaded training features of size', data_x.shape)
    n_folds = 4
    np.random.seed(1)

    sync_results = []

    for i in range(n_folds):
        print('### Cross-validation fold %d/%d' % (i+1, n_folds))
        X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20, random_state=42)

        print('Training...', X_train.shape)
        trained_model = model.train_with_spleeter_output(X_train, y_train)

        # save some memory
        del X_train
        del y_train

        # test serialization
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, 'model.bin')
            print('testing serialization in temp file', tmp_file)
            model.save(trained_model, tmp_file)
            trained_model = model.load(tmp_file)

        print('Validating...')
        predicted_score = model.predict(trained_model, X_test)
        predicted_label = np.round(predicted_score)
        correct = predicted_label == y_test

        result_meta = pd.DataFrame(np.array([y_test, predicted_score, predicted_label, correct]).T,
                                 columns=['label', 'predicted_score', 'predicted_label', 'correct'])
        result_meta['label'] = result_meta['label'].astype(float)

        from sklearn.metrics import roc_auc_score
        print('AUC-ROC:', roc_auc_score(y_test, predicted_label))

        bias = trained_model[1]
        r = result_meta.groupby('label').agg('mean')
        sync_r = test_correct_sync(result_meta, bias)
        sync_results.append(sync_r.assign(speech_detection_accuracy=list(r.correct)))

    sync_results = pd.concat(sync_results)
    print(sync_results)
    print('skew errors:', sync_results.skew_error.sum())
    print('shift RMSE:', np.sqrt(np.mean(sync_results.shift_error**2)))
    print('shift max:', sync_results.shift_error.abs().max())
    print('shift bias, mean:', sync_results.shift_error.mean(), 'median:', sync_results.shift_error.median())
    print('speech detection accuracy (mean of means):', sync_results.speech_detection_accuracy.mean())

    # save some more memory
    del data_x
    del data_y

    correct_qualities = np.asarray(sync_results.quality)
    print('false negative quality errors:', np.sum(correct_qualities < quality_of_fit.threshold))