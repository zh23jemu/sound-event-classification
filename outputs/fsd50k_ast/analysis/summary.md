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

## 类别级分析

AP 最高的 5 个类别：
- `Burping_and_eructation`：AP=0.9852，support=33，F1=0.9180
- `Cat`：AP=0.9827，support=38，F1=0.9041
- `Thunder`：AP=0.9817，support=66，F1=0.9365
- `Thunderstorm`：AP=0.9801，support=66，F1=0.9531
- `Toilet_flush`：AP=0.9616，support=31，F1=0.9153

AP 最低的 5 个有正样本类别：
- `Tick`：AP=0.0149，support=10，F1=0.0000
- `Screech`：AP=0.0703，support=18，F1=0.0000
- `Wood`：AP=0.0821，support=20，F1=0.0000
- `Chatter`：AP=0.1065，support=10，F1=0.0000
- `Rattle`：AP=0.1336，support=40，F1=0.1455

## 阈值敏感性

- micro-F1 最优阈值：0.20，micro-F1=0.7101。
- macro-F1 最优阈值：0.15，macro-F1=0.5747。
- 如果最优阈值明显低于 0.50，说明固定 0.50 阈值可能低估了低频类别召回率，需要在报告中说明阈值选择策略。

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `validation_metrics.png`：验证 mAP、micro-F1 和 macro-F1 曲线。
- `class_metrics.csv`：类别级 AP、Precision、Recall 和 F1。
- `threshold_sensitivity.csv` / `threshold_sensitivity.png`：不同阈值下的 F1 变化。
