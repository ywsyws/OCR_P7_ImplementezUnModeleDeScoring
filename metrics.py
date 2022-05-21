from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_metrics(
    y_true: Union[np.array, pd.Series],
    y_pred: Union[np.array, pd.Series],
    y_score: np.array = None,
) -> Tuple[
    float,
    float,
    float,
    float,
    np.array,
    np.array,
    np.array,
    float,
    np.array,
    np.array,
    np.array,
]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    prec_pr, recall_pr, thresholds_pr = precision_recall_curve(y_true, y_score)
    confusionmatrix = confusion_matrix(y_true, y_pred)
    return (
        acc,
        prec,
        recall,
        f1,
        fpr,
        tpr,
        thresholds_roc,
        roc_auc,
        prec_pr,
        recall_pr,
        thresholds_pr,
        confusionmatrix,
    )