# FSD50K Pretrained AST 多标签结果摘要

## 整体结果

- 训练轮数：5
- 验证样本数：4170
- 类别数：200
- 最佳验证 mAP：0.6208（第 4 轮）
- 最佳轮验证 Loss：0.0302
- 最佳轮验证 micro-F1：0.6955
- 最佳轮验证 macro-F1：0.4810
- 最后一轮验证 mAP：0.6064
- 最后一轮验证 micro-F1：0.6857
- 最后一轮验证 macro-F1：0.5015

## 初步结论

- 预训练 AST 已成功扩展到 FSD50K 多标签任务，验证 mAP 达到 0.6208。
- 训练 mAP 持续上升而验证 mAP 在第 4 轮达到峰值，说明继续训练可能开始带来过拟合或阈值不稳定。
- macro-F1 低于 micro-F1，说明长尾类别或低频类别仍然更难，需要后续做类别不均衡和阈值策略分析。

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `validation_metrics.png`：验证 mAP、micro-F1 和 macro-F1 曲线。
