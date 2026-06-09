# 项目级 AGENTS.md

## 项目目标

- 研究方向：基于深度学习的声音事件分类（Sound Event Classification）。
- 当前优先方向：将音频转换为 Mel-Spectrogram，使用 Vision Transformer / Audio Spectrogram Transformer 类模型完成分类。
- 备选方向：尝试将音视频相关模型或音视频预训练表征迁移到声音事件分类任务中。
- 当前阶段目标：先完成文献综述与初步设计方案，待方案确认后再进入模型代码实现与训练。

## 技术栈

- 语言：Python
- 深度学习框架：PyTorch（预期）
- 音频处理：librosa、torchaudio（预期）
- 模型方向：ViT / AST / 自监督音频 Transformer / 音视频迁移
- 文档产出：Markdown、Word（按课程要求整理）

## 当前架构

- 当前仓库已从文档初始化阶段进入 baseline 代码框架阶段。
- 已有输入材料：`模型要求.docx`
- 已有代码结构：`configs/` 保存训练配置，`scripts/` 保存数据检查和训练入口，`src/sound_event_classification/` 保存数据读取、特征提取、模型和指标代码，`slurm/` 保存集群提交脚本。
- 当前 baseline 以 ESC-50 + Log-Mel Spectrogram + 轻量 CNN 为第一步，后续再接入 AST/ViT 预训练模型做改进比较。

## 开发规范

- 优先最小修改，避免无关重构。
- 新增代码默认补充较详细中文注释。
- Python 相关执行默认使用项目本地虚拟环境 `.venv`。
- 优先记录研究过程、问题、解决方案与实验结论，满足项目日志要求。
- 先完成文献综述与方案论证，再进入训练实现阶段。

## TODO

- 完成指定会议范围内的参考文献初筛：NeurIPS、ICLR、ISMIR。
- 确定主数据集与辅数据集，并说明选择理由、风险与替代方案。
- 输出文献综述初稿提纲：背景、现状、意义、研究问题、预期成果、成功标准、时间计划。
- 确定主线模型方案与备选方案。
- 设计项目日志模板，覆盖日期、目标、问题、解决方案、实验记录与结论。

## 当前进度

- 已读取并整理 `模型要求.docx` 中的任务要求。
- 已明确项目应先提交文献综述（初步设计方案），再进行模型实现与训练。
- 已初步确定主线更适合选择 ViT/AST 音频分类方向，音视频迁移作为第二方案或扩展方案。
- 已形成当前推荐的数据路线：ESC-50 用于快速验证，FSD50K 用于主实验；AudioSet 用于研究背景和预训练权重来源，VGGSound 用于音视频迁移备选方向论证。

## 风险问题

- 当前没有现成数据集，需要先完成数据集选型与可获取性验证。
- AudioSet、VGGSound 等大规模数据集可能存在下载复杂、资源需求较高、样本失效等问题。
- 纯 ViT/AST 在小数据集上可能过拟合，需要考虑迁移学习、自监督预训练或强数据增强。
- 如果时间有限，音视频迁移方案的工程复杂度和复现实验成本可能偏高。

## Current Status

- 项目处于文献综述与初步设计方案准备阶段。
- 已完成 `模型要求.docx` 的需求复核，明确当前先不写训练代码，优先完成数据集推荐、参考文献初筛、研究问题、成功标准和项目计划。
- 已完成第四部分“文献综述与项目计划”的 Markdown 初稿，文件为 `文献综述与项目计划.md`。
- 已按用户新要求新增学术论文风格的 `文献综述.md`，参考文献采用 IEEE 编号格式，正文聚焦 6000-8000 中文综述。
- 当前建议实验路线：先用 ESC-50 做快速原型验证，再以 FSD50K 作为主数据集；AudioSet 主要作为预训练/对照背景，VGGSound 作为第二方向的音视频迁移备选。
- 已在 `文献综述.md` 中明确 AudioSet 不作为本项目直接训练或完整下载的数据集，只作为研究背景和预训练权重来源。
- 已在 `文献综述.md` 中明确 VGGSound 不作为主线训练数据集，主要用于支撑 video-audio 模型迁移备选方向和未来扩展实验。
- 已进一步收束 `文献综述.md` 中 AudioSet 与 VGGSound 的写法：二者都保留简要讨论，但不作为第一阶段直接训练数据集。
- 已新增 `项目计划.md`，按 6 月初中期展示、6 月底模型完成、7-8 月报告撰写、9 月 1 日最终提交的时间线规划项目。
- 曾将 `文献综述.md` 转换为 Word 文档 `文献综述.docx`；后续 2026-06-05 已重新生成新版 Word 文档，新版仍需在具备 Word/LibreOffice 的环境中做页面级视觉检查。
- 已完成 2026-06-05 项目现状复核：仓库当前仍以文献综述、项目计划和 Word 交付文档为主，尚未进入训练代码实现阶段。
- 已读取 `反馈修改.docx`，确认最新反馈重点是重写/调整 Literature Review 与 Project Plan 的边界，而不是改变项目技术路线。
- 已按 `反馈修改.docx` 完成 `文献综述.md` 与 `项目计划.md` 修订，并重新生成 `文献综述.docx` 与 `项目计划.docx`。
- 修订后的 `文献综述.md` 聚焦已有研究、方法比较、综合分析和研究空白，参考文献扩展到 18 篇，弱化项目实施方案描述。
- 修订后的 `项目计划.md` 明确补充 Research Question、Intended Outcomes、实验范围、时间安排、成功标准、风险应对和 AI 工具使用说明。
- 已开始按照项目计划进入代码实现阶段，新增 ESC-50 baseline 的最小可运行工程骨架。
- 已新增 `README.md`、`requirements.txt`、`configs/esc50_baseline.yaml`、`scripts/prepare_esc50.py`、`scripts/train.py`、`src/sound_event_classification/`、`slurm/train_esc50_baseline.sbatch` 和 `项目日志.md`。
- 服务器端 ESC-50 baseline 已完成首次 20 epoch 训练，最佳验证 Accuracy 为 0.4125，最佳模型保存到 `outputs/esc50_baseline/best_model.pt`。
- 已基于同步回本地的 `history.json` 和 `latest_val_metrics.json` 生成训练曲线、归一化混淆矩阵、类别级指标 CSV 和 Markdown 结果摘要。
- 已新增 CNN + SpecAugment 对比实验入口，配置为 `configs/esc50_cnn_specaugment.yaml`，默认输出到 `outputs/esc50_cnn_specaugment`。
- 服务器端 CNN + SpecAugment 已完成训练，最佳验证 Accuracy 为 0.4200，相比 CNN baseline 的 0.4125 提升 +0.0075；已生成 `outputs/esc50_comparison` 对比摘要。
- 已新增预训练 AST 微调入口，配置为 `configs/esc50_ast.yaml`，训练脚本为 `scripts/train_ast.py`，Slurm GPU 脚本为 `slurm/train_esc50_ast.sbatch`。
- 服务器端预训练 AST 已完成训练，最佳验证 Accuracy 为 0.9300，第 4 轮达到最佳；相比 CNN baseline 提升 +0.5175，当前是项目最强主线结果。
- 已新增 `实验结果分析.md`，将 CNN baseline、CNN + SpecAugment 和 Pretrained AST 三组结果整理成最终报告可用的实验结果分析章节草稿。
- 已新增 `模型报告.md`，整合文献背景、研究问题、方法、实验结果、讨论、局限性、未来工作和参考文献，形成完整模型报告初稿。
- 已同步 ESC-50 官方元数据 `metadata/esc50.csv`，并重新生成三组实验的真实类别名混淆矩阵、类别级指标和分析摘要。

## Recent Changes

- 新建项目级 `AGENTS.md`，开始记录项目目标、约束、风险与后续计划。
- 根据任务要求，明确主方案为 Mel-Spectrogram + ViT/AST 的声音事件分类路线。
- 明确备选方案为音视频模型迁移到声音事件分类场景。
- 补充了首选数据集建议：FSD50K 作为主实验入口，ESC-50 作为快速原型验证集，AudioSet 作为预训练背景，VGGSound 作为音视频迁移备选依据。
- 复核课程文档后，补充了阶段性交付顺序：先提交文献综述与初步设计方案，方案确认后再进入模型代码实现与训练。
- 完成首轮外部资料核对，确认可围绕 NeurIPS、ICLR、ISMIR 组织 ViT/AST、自监督音频 Transformer、音视频迁移和音乐/声音分类相关文献。
- 新增 `文献综述与项目计划.md`，覆盖项目背景、研究现状、预期成果、研究问题、时间计划、甘特图、成功标准和参考文献。
- 新增 `文献综述.md`，改为摘要、关键词、引言、相关研究、趋势分析、结论和 IEEE 参考文献的学术综述结构。
- 更新 `文献综述.md` 中 AudioSet 的定位说明，避免读者误解为项目会直接下载并训练完整 AudioSet。
- 更新 `文献综述.md` 中 VGGSound 的定位说明，强调其服务于音视频迁移备选方向，而非替代主线纯音频实验。
- 调整 `文献综述.md` 摘要和数据集段，明确 AudioSet/VGGSound “都提但不重点展开”的写作策略。
- 新增 `项目计划.md`，明确阶段安排、甘特图、中期展示半成品策略、预期成果、成功标准、风险应对和最终交付清单。
- 更新 `.gitignore`，忽略 Word 临时锁文件 `~$*.docx`。
- 新增 `文献综述.docx`，按学术报告样式排版 Markdown 文献综述，并修正 Word 中 Markdown 强调符号的显示问题。
- 读取并整理 `反馈修改.docx` 的修改要求：文献综述应减少项目实施方案描述，增强已有研究讨论、引用密度、主题化章节结构、文献综合分析和研究空白总结；Research Question 与 Intended Outcomes 更适合放入 Project Plan。
- 重写 `文献综述.md`：按任务与数据集、时频表示与 CNN、频谱图 Transformer、自监督音频表征、数据增强、音视频学习和研究空白组织内容。
- 重写 `项目计划.md`：将 Research Question 与 Intended Outcomes 前置，并将数据集使用范围、方法流程、时间计划、成功标准和风险控制集中到计划文档。
- 使用新版 Markdown 重新生成 `文献综述.docx` 和 `项目计划.docx`，并通过反提 Word 正文确认内容已同步；由于本机缺少 LibreOffice/`soffice`，未能完成页面 PNG 渲染视觉检查。
- 新增 ESC-50 baseline 工程骨架：数据目录检查脚本、Log-Mel Spectrogram 特征模块、ESC-50 Dataset、轻量 CNN baseline、单标签分类指标和训练入口。
- 新增 Slurm CPU baseline 脚本，默认 `defq` 分区、`gpo-ifv7xx` 账号、`normal` QOS；注释说明后续可用 `sbatch --partition=gpu` 覆盖分区。
- 更新 `.gitignore`，忽略原始数据目录、大模型权重、checkpoint 和 Slurm 运行日志，但不整体忽略 `outputs/`，保留小型指标和图表用于报告分析。
- 服务器首次运行 ESC-50 baseline 时，数据检查通过且任务成功切换到 `ISP` 分区，但训练在 `torchaudio.load()` 阶段触发 TorchCodec/FFmpeg 依赖问题；已改用 `scipy.io.wavfile` 读取 ESC-50 WAV，避开 TorchCodec。
- 服务器重跑 ESC-50 baseline 成功完成：Slurm 作业 `34857536` 在 `ISP` 分区完成 20 epoch，最佳验证 Accuracy = 0.4125，说明当前数据读取、Log-Mel 特征、CNN 训练和 Slurm 提交流程均已跑通。
- 新增 `scripts/analyze_esc50_results.py`，生成 `outputs/esc50_baseline/analysis/training_loss.png`、`training_accuracy.png`、`confusion_matrix_normalized.png`、`class_metrics.csv` 和 `summary.md`。
- 新增 `SpecAugment` 训练增强模块，并更新 `scripts/train.py` 支持训练阶段增强、验证阶段关闭；更新 Slurm 脚本支持 `CONFIG` 环境变量切换配置。
- 新增 `scripts/compare_esc50_experiments.py`，并生成 CNN baseline 与 CNN + SpecAugment 的对比表；SpecAugment 最佳验证 Accuracy = 0.4200，baseline = 0.4125。
- 新增 Hugging Face AST 微调脚本和 GPU Slurm 脚本，补充 `transformers`、`accelerate` 依赖；AST 输出沿用 `history.json` 和 `latest_val_metrics.json`，便于复用现有分析流程。
- 已同步 AST 训练结果并生成 `outputs/esc50_ast/analysis`；已更新 `outputs/esc50_comparison`，纳入 CNN baseline、CNN + SpecAugment 和 Pretrained AST 三组实验。
- 新增 `实验结果分析.md`，覆盖实验目的、数据划分、实验设置、结果表、训练曲线、混淆矩阵、讨论、局限性和后续工作。
- 新增 `模型报告.md`，将 `文献综述.md`、`项目计划.md` 和 `实验结果分析.md` 的核心内容整合为最终报告初稿。
- 新增 `metadata/esc50.csv`，重新生成 `outputs/esc50_baseline/analysis`、`outputs/esc50_cnn_specaugment/analysis` 和 `outputs/esc50_ast/analysis`，类别名已从 `class_00` 等编号更新为 ESC-50 真实类别名称。

## Next TODO

- 将更新真实类别名后的 `模型报告.md` 转换为 Word 文档，并检查图表、表格、页数和参考文献格式。
- 如果时间允许，围绕 AST 较弱类别 `helicopter`、`pig`、`door_wood_creaks`、`airplane` 补充错误分析，或做多 fold 验证/轻量调参。
- 后续若具备 LibreOffice/Word 环境，应打开或渲染检查 `文献综述.docx` 与 `项目计划.docx` 的实际页数、表格宽度和分页效果，确认 Draft Literature Review + Project Plan 总篇幅不超过 12 页。

## Open Issues

- 仍需结合实际算力与时间，最终确认是否直接以 FSD50K 作为主实验入口，或先用 ESC-50 完成一轮基线复现后再切换。
- 尚未确认算力条件，暂时无法精确锁定预训练、微调还是从头训练的方案。
- 音视频迁移方向需要进一步评估数据可得性与实现复杂度。
- VGGSound 官方下载入口不再直接提供完整视频文件，后续如走音视频方向，需要提前验证可下载样本比例。
- AudioSet 规模大且依赖 YouTube 片段可用性，不适合作为课程项目的第一落地数据集。
- 后续实验实现阶段应优先查找并使用已发布的 AudioSet 预训练权重，而不是自行处理完整 AudioSet。
- 6 月初中期展示需要刻意保留“接近完成但可继续优化”的半成品状态，避免展示内容看起来已经完全结束。
- `文献综述.docx` 和 `项目计划.docx` 已由 Markdown 重新生成，但当前环境缺少 `soffice`，尚未完成页面级视觉渲染检查；如果导师要求严格版式，需要在 Word/LibreOffice 中再做最终检查。
- 当前总篇幅约为 `文献综述.md` 4810 个中文字符、`项目计划.md` 2610 个中文字符，理论上接近但应能控制在 12 页内；实际页数仍需以 Word 渲染为准。
- 当前 `.venv` 依赖尚未完整安装；本次执行 `.venv\Scripts\python.exe -m pip install -r requirements.txt` 在 120 秒后超时，仍缺 `torch`、`pandas` 等依赖。
- 本地 Windows 环境仍未完成完整依赖安装，真实训练目前以服务器 Slurm 环境为准。
- ESC-50 baseline 已生成类别级指标和混淆矩阵，但本地缺少 ESC-50 元数据时类别名只能显示为编号；如报告需要可读类别名称，需要同步 `meta/esc50.csv` 或在服务器上运行分析脚本。
- AST 首次运行的 Hugging Face 预训练权重已成功加载；分类头从 AudioSet 527 类重建为 ESC-50 50 类时出现 MISMATCH 提示属于预期现象。

## Architecture Decisions

- 第一阶段以“可顺利提交综述和初步方案”为目标，优先选择文献充足、数据开放、实现风险低的路线。
- 主线模型采用音频频谱图输入的 Transformer 路线，因为它与课程要求中的 ViT 方向直接一致。
- 备选路线不单独立项，作为主线失败或时间允许时的增强实验方向。
- 数据集策略采用“快速验证 + 主实验 + 预训练背景 + 备选迁移”：ESC-50 用于快速跑通，FSD50K 用于主实验，AudioSet 用于预训练背景，VGGSound 用于音视频迁移备选论证。
- 第四部分文档先以 Markdown 交付，便于后续继续扩展为 Word 版最终报告。
- 根据最新反馈，后续文档组织应采用“Literature Review 聚焦已有研究综合分析，Project Plan 单独承接研究问题、预期成果、实施路线和时间计划”的双文档边界。
