# Git 归档说明

归档日期：2026-09-07。项目目录：`E:\LSTM_trajectory_prediction`。

## 版本管理范围

纳入 Git：Python 源码、测试、依赖声明、训练与参数搜索配置、README 和项目文档。

不纳入 Git：`robomimic_datasets/`、所有根目录 `outputs*/`、模型权重、Python 缓存、本地虚拟环境、编辑器个人配置、临时文件和压缩包。规则见根目录 `.gitignore`。忽略只影响版本管理，不删除本地文件。

Git 仓库是代码与配置归档，不包含完整实验数据备份。分享或克隆仓库后，需要另外取得数据集；如果需要复用已有模型或检查完整实验日志，还需单独备份和传递相关输出目录。

## 当前实验状态

`outputs_fine_tuning/summary.md` 记录 24/24 组搜索及最优配置正式复训已完成，验证和测试 ADE 相对历史参考改善均为 0.00%。

- 任务：42 维机器人运动状态，历史 8 帧，预测未来 12 帧末端三维位置。
- 配置：单层 LSTM、隐藏维度 128、学习率 3e-4、Dropout 0.2、AdamW、权重衰减 1e-4、Smooth L1、batch size 64、梯度裁剪 1.0。
- 验证 ADE：0.03217481 m；测试 ADE：0.03265489 m；测试 FDE：0.05351266 m；最佳检查点：第 8 轮。
- 可版本管理的对应配置：`config_fine_tuning.json`；24 组搜索配置：`fine_tuning_config.json`。
- 本轮结果仅支持有限搜索范围内未获得改善，不能证明达到全局性能极限。

## 恢复运行

1. 在项目根目录准备 Python 与 PyTorch 环境，并安装 `requirements.txt` 中的依赖。PyTorch 的 CUDA 版本需匹配运行机器。
2. 将数据文件另行放置到 `robomimic_datasets/low_dim_dense_v15.hdf5`，或修改配置中的 `dataset_path`。
3. 使用一个新的输出目录启动训练：

```powershell
python train_lstm.py --config config_fine_tuning.json --output-dir outputs_reproduction
```

4. 若需复核代码测试：

```powershell
python -m pytest -q
```

不同硬件与依赖版本可能影响数值复现；`requirements.txt` 是依赖范围声明，并非精确版本锁定文件。
