# ESC-50 CNN + SpecAugment 结果摘要

## 整体结果

- 训练轮数：20
- 验证样本数：400
- 最佳验证 Accuracy：0.4200（第 19 轮）
- 最佳轮验证 Loss：2.0417
- 最后一轮训练 Accuracy：0.4006
- 最后一轮验证 Accuracy：0.3825
- 最后一轮验证 Loss：2.0839

## 类别级表现较好

- hand_saw：Accuracy 1.0000，8/8
- church_bells：Accuracy 1.0000，8/8
- sea_waves：Accuracy 1.0000，8/8
- rain：Accuracy 1.0000，8/8
- crying_baby：Accuracy 0.8750，7/8

## 类别级表现较弱

- frog：Accuracy 0.0000，0/8
- cat：Accuracy 0.0000，0/8
- crow：Accuracy 0.0000，0/8
- water_drops：Accuracy 0.0000，0/8
- coughing：Accuracy 0.0000，0/8

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `training_accuracy.png`：训练/验证 Accuracy 曲线。
- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵。
- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。

## 初步结论

- 当前实验结果明显高于 50 类随机水平，说明数据读取、训练流程和评估记录有效。
- 后续应结合多实验对比摘要判断该方法相对 baseline 的提升幅度，并继续补充类别级错误分析。
