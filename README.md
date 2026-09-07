# LSTM-Machinery-Trajectory-prediction

基于 LSTM 的机械轨迹预测：使用 robomimic 数据预测机器人末端执行器未来三维位置。

Git 归档范围、数据恢复方式及最新实验状态见 [Git 归档说明](docs/GIT_ARCHIVE.md)。数据集和训练输出保留在本地，不随 Git 仓库分发。

本项目基于 robomimic 的 `low_dim_dense_v15.hdf5` 数据集，将每段机器人演示视为独立时间序列：利用过去的机器人状态，预测末端执行器未来多个时刻的三维位置。

## 当前结论

截至 2026-08-30，已完成基线建立、随机超参数筛选、历史窗口二次调优、训练正则化以及损失函数对照。当前最佳的可复现实验采用 **8 帧历史窗口 + Smooth L1 损失**：测试集 ADE 为 **0.03265 m**，FDE 为 **0.05351 m**。

相对于已保存的 12 帧初始基线（ADE 0.03323 m，FDE 0.05473 m），最佳模型的 ADE 降低约 **1.7%**、FDE 降低约 **2.2%**。该结论只适用于本数据集、当前随机轨迹划分和已搜索的有限参数空间，不能视为全局最优。

详细阶段总结见 `LSTM近期模型工作总结.docx`。

## 数据、任务与切分

- **输入：** 每个历史帧 42 个特征：末端位置（3）、末端四元数（4）、关节位置（7）、关节速度（7）、动作（7）和对象状态（14）。
- **目标：** 未来 12 帧 `robot0_eef_pos` 的三维位置。模型预测相对历史末帧的位置增量，评估时再还原为绝对位置。
- **切分：** 按完整演示轨迹随机分为训练/验证/测试集 80% / 10% / 10%，归一化统计量仅由训练集计算。
- **当前最佳实验规模：** 3120 / 390 / 390 条演示；408,720 / 51,090 / 51,090 个训练、验证、测试窗口。
- **频率：** 数据控制频率为 20 Hz；8 帧历史约为 0.4 秒，预测 12 帧约为 0.6 秒。

## 模型与训练方法

模型为 LSTM 编码器加多步三维回归头：LSTM 的末状态经 `Linear -> ReLU -> Dropout -> Linear` 头部直接输出 12 x 3 个位置增量。

当前最佳配置位于 `outputs_history_window_tuning/best_config.json`：

| 参数 | 值 |
| --- | --- |
| 历史窗口 | 8 帧 |
| 预测窗口 | 12 帧 |
| LSTM 隐藏维度 / 层数 | 128 / 1 |
| Dropout | 0.2 |
| 优化器 | AdamW |
| 学习率 | 3e-4 |
| 批大小 | 64 |
| 损失函数 | Smooth L1 |
| 正则化 | weight decay 1e-4、梯度裁剪 1.0 |
| 调度与早停 | ReduceLROnPlateau（factor 0.5，patience 2）；早停 patience 8 |

## 已完成的参数优化

### 第一轮：随机组合筛选

在 `outputs_tuning/` 中，已从以下有限空间随机筛选 12 组组合：历史窗口 8/12/16，隐藏维度 64/128/256，层数 1/2，学习率 3e-4/1e-3，批大小 64/128。候选模型仅按**验证集 ADE**排序，测试集不参与选择。

该轮最优筛选组合为 12 帧历史、256 隐藏维度、单层 LSTM、学习率 3e-4、批大小 64；正式复训测试 ADE 为 0.03309 m。

### 第二轮：历史窗口聚焦调优

在 `outputs_history_window_tuning/` 中，固定其余关键参数，仅比较 8、12 和 20 帧历史窗口。8 帧窗口在验证集 ADE、FDE 和 MSE 上均为最优，并经 40 轮上限训练与早停复训确认。

### 稳定性与损失函数对照

训练流程已加入 AdamW、权重衰减、回归头 Dropout、梯度裁剪、学习率调度和早停。`outputs_history8_mse/` 显示，在同为 8 帧历史的条件下，MSE 损失的测试 ADE 为 0.03326 m，高于 Smooth L1 的 0.03265 m，因此当前保留 Smooth L1。

## 结果对比

MAE、ADE、FDE 的单位为米，MSE 的单位为平方米；ADE 是所有未来点的平均欧氏距离，FDE 是最后一个预测点的欧氏距离。

| 实验阶段 | 关键设置 | 测试 MSE | 测试 MAE | 测试 ADE | 测试 FDE |
| --- | --- | ---: | ---: | ---: | ---: |
| 初始基线 | 12 帧、128 隐藏维度 | 0.000742 | 0.01651 | 0.03323 | 0.05473 |
| 第一轮调参复训 | 12 帧、256 隐藏维度 | 0.000717 | 0.01645 | 0.03309 | 0.05462 |
| 正则化对照 | Smooth L1 + AdamW | 0.000737 | 0.01643 | 0.03311 | 0.05455 |
| 历史窗口最优 | 8 帧、128 隐藏维度、Smooth L1 | **0.000702** | **0.01622** | **0.03265** | **0.05351** |
| 损失函数对照 | 8 帧、MSE | 0.000711 | 0.01652 | 0.03326 | 0.05435 |

## 运行

在 `E:\LSTM_trajectory_prediction` 中使用已验证的 PyTorch 环境：

```powershell
& 'D:\Anaconda\envs\traisformer\python.exe' train_lstm.py --config config.json --output-dir outputs
```

运行第一轮随机筛选：

```powershell
& 'D:\Anaconda\envs\traisformer\python.exe' hyperparameter_tuning.py --tuning-config tuning_config.json
```

运行历史窗口二次调优：

```powershell
& 'D:\Anaconda\envs\traisformer\python.exe' hyperparameter_tuning.py --tuning-config history_window_tuning.json
```

## 输出说明

- `best_model.pt`：最佳验证模型权重、结构及归一化信息。
- `metrics.json`：测试集 MSE、MAE、ADE、FDE、数据规模与训练设置。
- `training_history.json`：逐轮训练、验证损失和验证指标。
- `tuning_results.json`：候选参数及其验证集排序。
- `trajectory_example.png` / `representative_trajectory_comparison.png`：预测轨迹与真实轨迹的三维对比。

## 下一步建议

现有结果支持优先保留较短历史窗口和 Smooth L1，但尚未证明其在其他随机种子或数据分布上稳定领先。按当前工作安排，跳过额外的多随机种子复核，直接执行下面的局部参数搜索。

## 当前工作：8 帧模型局部精调（2026-09-05）

状态更新（2026-09-07）：24 组搜索和最优配置正式复训均已完成。最优仍为原配置，验证及测试 ADE 改善均为 0.00%；以下保留搜索设置和运行说明，供复现使用。

基础配置为 `config_fine_tuning.json`，沿用历史最优的 8 帧输入、12 帧预测、单层 LSTM、Smooth L1、AdamW、批大小 64、梯度裁剪 1.0 和随机种子 42。

`fine_tuning_config.json` 枚举以下 **24 组完整组合**：

| 参数 | 搜索值 |
| --- | --- |
| 隐藏维度 | 128、64 |
| 学习率 | 3e-4、1e-4 |
| 回归头 Dropout | 0.2、0.1、0.3 |
| 权重衰减 | 1e-4、1e-3 |

当前参数组合包含在这 24 组内，默认排在第一组；不额外安排多随机种子实验。每组最多 20 轮，验证 ADE 连续 6 轮没有改善则早停。全部筛选完成后，仅对验证 ADE 最优组合进行一次正式复训，最多 40 轮、早停 patience 8。总计最多 24 次筛选训练和 1 次复训，并非每组都会跑满轮数。

本轮统一按**验证 ADE**进行权重保存、早停、学习率调度和参数排序。历史单独训练默认按验证损失保存，新旧训练控制略有差异；汇总中的历史提升百分比仅作单次结果参考。筛选阶段不运行测试评估、不绘制测试轨迹；最终复训完成后才计算测试指标。所有候选历史长度相同，使用相同划分和预测窗口。

### 直接启动

在 PowerShell 中执行：

```powershell
Set-Location 'E:\LSTM_trajectory_prediction'
& 'D:\Anaconda\envs\traisformer\python.exe' -u hyperparameter_tuning.py --tuning-config fine_tuning_config.json --resume
```

首次执行也可带 `--resume`。结果写入新的 `outputs_fine_tuning/`，历史实验目录保留。终端会显示每组参数、运行设备、批次进度、逐轮训练损失、验证 ADE/FDE、学习率和耗时。

只查看 24 组参数，不读取 HDF5、不训练、不创建结果目录：

```powershell
& 'D:\Anaconda\envs\traisformer\python.exe' hyperparameter_tuning.py --tuning-config fine_tuning_config.json --dry-run
```

### 暂停和继续

按 `Ctrl+C` 停止后，再执行同一条带 `--resume` 的启动命令。已完成的候选和已完成的正式复训会跳过；**中断的那一组从第 1 轮重新训练**，不恢复其优化器状态。

续跑会核对配置、代码指纹和数据文件大小/修改时间；如果中途修改了搜索设置或训练代码，需要还原原设置，或通过 `--output-dir outputs_fine_tuning_v2` 使用新目录。数据指纹不是完整内容哈希，不要原地替换数据文件。旧版实验目录没有本轮的续跑清单，不能直接作为新搜索的续跑目录。

同一结果目录一次只运行一个训练进程；想在已有任务运行时修改代码，应先停止该进程。数据预处理结果会在同一进程内复用，重启后重新准备一次。

### 训练后交给我分析的文件

优先提供 `outputs_fine_tuning/tuning_results.json` 和 `outputs_fine_tuning/summary.md`，或直接让我读取该目录。

| 文件 | 内容 |
| --- | --- |
| `tuning_results.json` | 实时完成状态、候选排名、最优参数和最终指标 |
| `leaderboard.csv` | 可用 Excel 打开的排名表，包含耗时、参数量和实际训练轮数 |
| `summary.md` | 中文结果摘要及相对历史参考的变化；负提升表示结果变差 |
| `search_manifest.json` | 本批实验的配置、数据标识和代码指纹 |
| `trials/trial_XXX/` | 各组权重、逐轮历史、验证指标、完整配置和数据划分 |
| `best_config.json` | 正式复训的完整参数 |
| `best_model/` | 正式复训的权重、训练历史、验证及测试指标、轨迹图 |

`training_history.json` 在每轮结束后保存；完成一组后更新总排名。因此中途停止也能分析已经完成的实验。`metrics.json` 在一组成功完成后才写入，避免把仅有部分训练日志的目录误当作已完成。
