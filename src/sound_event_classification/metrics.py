from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    """计算单标签分类指标。

    当前 baseline 只处理 ESC-50 单标签分类，因此先返回 Accuracy 和混淆矩阵。
    后续扩展到 FSD50K 时，需要新增 mAP、micro-F1、macro-F1 等多标签指标。
    """

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "num_samples": int(np.asarray(y_true).shape[0]),
    }

