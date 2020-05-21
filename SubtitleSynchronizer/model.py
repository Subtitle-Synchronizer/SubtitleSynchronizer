import numpy as np
import json
from . import features
from .trained_logistic_regression import TrainedLogisticRegression


def transform(data_x):
    return np.hstack([
        features.expand_to_adjacent(data_x, width=1),
        features.rolling_aggregates(data_x, width=2, aggregate=np.max),
        features.rolling_aggregates(data_x, width=5, aggregate=np.max)
    ])


def normalize(data_x):
    return data_x


def train_with_spleeter_output(training_x, training_y):
    from sklearn.linear_model import LogisticRegression as classifier
    training_weights = features.weight_by_group(training_y)
    training_x_normalized = transform(training_x)
    
    speech_detection = classifier(penalty='l1', C=0.001, solver='liblinear')
    speech_detection.fit(training_x_normalized, training_y, sample_weight=training_weights)

    speech_detection = TrainedLogisticRegression.from_sklearn(speech_detection)

    # save some memory
    del training_weights

    bias = 0
    return [speech_detection, bias]


def predict(model, test_x):
    speech_detection = model[0]
    test_x = transform(test_x)
    return speech_detection.predict_proba(test_x)[:,1]


def serialize(model):
    return json.dumps({
        'logistic_regression': model[0].to_dict(),
        'bias': model[1]
    })


def deserialize(data):
    d = json.loads(data)
    return [
        TrainedLogisticRegression.from_dict(d['logistic_regression']),
        d['bias']
    ]


def load(model_file):
    with open(model_file, 'r') as f:
        return deserialize(f.read())


def save(trained_model, target_file):
    with open(target_file, 'w') as f:
        f.write(serialize(trained_model))
