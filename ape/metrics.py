import numpy as np


def amca_from_confusion_matrix(confusion_matrix):
    """Average Mean Class Accuracy (AMCA) from a confusion matrix tensor."""
    confusion_matrix = confusion_matrix.numpy()
    class_accuracies = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    return np.nanmean(class_accuracies)


def weighted_f1_from_confusion_matrix(confusion_matrix):
    """Weighted F1-score computed from a confusion matrix."""
    if hasattr(confusion_matrix, 'numpy'):
        confusion_matrix = confusion_matrix.numpy()

    precision = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diagonal(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    weights = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    return np.nansum(f1_scores * weights)
