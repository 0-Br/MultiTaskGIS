import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def score(logit, label):
    """"""
    logit = np.argmax(logit, axis=3).reshape(-1)
    label = np.array(label).reshape(-1)

    return {
        "matrix": confusion_matrix(label, logit),
        "accuracy": accuracy_score(label, logit),
        "f1": f1_score(label, logit, average="weighted")
    }
