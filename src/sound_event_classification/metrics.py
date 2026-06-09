from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score


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


def multilabel_classification_metrics(
    y_true: list[list[float]],
    y_score: list[list[float]],
    threshold: float = 0.5,
) -> dict[str, object]:
    """计算 FSD50K 多标签分类指标。

    FSD50K 是 clip-level 多标签任务，整体 Accuracy 不再合适，因此记录 mAP、
    micro-F1 和 macro-F1。`y_score` 是 sigmoid 后的类别概率，`threshold`
    用于把概率转换为二值预测。
    """

    true_array = np.asarray(y_true)
    score_array = np.asarray(y_score)
    pred_array = (score_array >= threshold).astype(int)

    return {
        "mAP": float(average_precision_score(true_array, score_array, average="macro")),
        "micro_f1": float(f1_score(true_array, pred_array, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true_array, pred_array, average="macro", zero_division=0)),
        "threshold": threshold,
        "num_samples": int(true_array.shape[0]),
        "num_classes": int(true_array.shape[1]) if true_array.ndim == 2 else 0,
    }
