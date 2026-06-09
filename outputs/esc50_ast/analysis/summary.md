# ESC-50 Pretrained AST 结果摘要

## 整体结果

- 训练轮数：10
- 验证样本数：400
- 最佳验证 Accuracy：0.9300（第 4 轮）
- 最佳轮验证 Loss：0.2205
- 最后一轮训练 Accuracy：1.0000
- 最后一轮验证 Accuracy：0.9225
- 最后一轮验证 Loss：0.2589

## 类别级表现较好

- church_bells：Accuracy 1.0000，8/8
- engine：Accuracy 1.0000，8/8
- car_horn：Accuracy 1.0000，8/8
- siren：Accuracy 1.0000，8/8
- glass_breaking：Accuracy 1.0000，8/8

## 类别级表现较弱

- helicopter：Accuracy 0.5000，4/8
- pig：Accuracy 0.6250，5/8
- door_wood_creaks：Accuracy 0.7500，6/8
- airplane：Accuracy 0.7500，6/8
- cow：Accuracy 0.8750，7/8

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `training_accuracy.png`：训练/验证 Accuracy 曲线。
- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵。
- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。

## 初步结论

- 当前实验结果明显高于 50 类随机水平，说明数据读取、训练流程和评估记录有效。
- 后续应结合多实验对比摘要判断该方法相对 baseline 的提升幅度，并继续补充类别级错误分析。
