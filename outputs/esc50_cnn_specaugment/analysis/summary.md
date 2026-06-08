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

- class_49：Accuracy 1.0000，8/8
- class_46：Accuracy 1.0000，8/8
- class_11：Accuracy 1.0000，8/8
- class_10：Accuracy 1.0000，8/8
- class_20：Accuracy 0.8750，7/8

## 类别级表现较弱

- class_04：Accuracy 0.0000，0/8
- class_05：Accuracy 0.0000，0/8
- class_09：Accuracy 0.0000，0/8
- class_15：Accuracy 0.0000，0/8
- class_24：Accuracy 0.0000，0/8

## 生成文件

- `training_loss.png`：训练/验证 Loss 曲线。
- `training_accuracy.png`：训练/验证 Accuracy 曲线。
- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵。
- `class_metrics.csv`：每个类别的样本数、正确数和类别准确率。

## 初步结论

- 当前 CNN baseline 明显高于 50 类随机水平，说明数据读取、Log-Mel 特征和训练流程有效。
- 验证准确率仍有较大提升空间，下一步适合加入数据增强，并接入预训练 AST/ViT 做改进比较。
