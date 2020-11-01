import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import lap


def nmi_score(pred_labels, real_labels):
    return normalized_mutual_info_score(real_labels, pred_labels, average_method='arithmetic')


def accuracy(pred_labels, real_labels, num_classes, num_samples):
    num_correct = np.zeros((num_classes, num_classes))

    for c_1 in range(num_classes):
        for c_2 in range(num_classes):
            num_correct[c_1, c_2] = int(((pred_labels == c_1) * (real_labels == c_2)).sum())

    _, assignments, _ = lap.lapjv(num_samples - num_correct)

    reordered_pred_labels = np.zeros(num_samples)

    for c in range(num_classes):
        reordered_pred_labels[pred_labels == c] = assignments[c]

    # accuracy per cluster
    cluster_accs = []
    for c in range(num_classes):
        cluster_accs.append(np.average(reordered_pred_labels[real_labels == c] == c))

    return np.average(reordered_pred_labels == real_labels), cluster_accs, assignments
