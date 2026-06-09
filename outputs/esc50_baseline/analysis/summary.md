# ESC-50 CNN Baseline 结果摘要

## 整体结果

- 训练轮数：20
- 验证样本数：400
- 最佳验证 Accuracy：0.4125（第 19 轮）
- 最佳轮验证 Loss：2.0202
- 最后一轮训练 Accuracy：0.4688
- 最后一轮验证 Accuracy：0.4075
- 最后一轮验证 Loss：2.1383

## 类别级表现较好

- crickets：Accuracy 1.0000，8/8
- hand_saw：Accuracy 0.8750，7/8
- siren：Accuracy 0.8750，7/8
- brushing_teeth：Accuracy 0.8750，7/8
- pouring_water：Accuracy 0.8750，7/8

## 类别级表现较弱

- pig：Accuracy 0.0000，0/8
- hen：Accuracy 0.0000，0/8
- sheep：Accuracy 0.0000，0/8
- chirping_birds：Accuracy 0.0000，0/8
- breathing：Accuracy 0.0000，0/8

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `training_accuracy.png`：训练/验证 Accuracy 曲线。
- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵。
- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。

## 初步结论

- 当前实验结果明显高于 50 类随机水平，说明数据读取、训练流程和评估记录有效。
- 后续应结合多实验对比摘要判断该方法相对 baseline 的提升幅度，并继续补充类别级错误分析。
