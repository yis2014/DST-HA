## DST-HA: A Dynamic Spatio-Temporal Hybrid Attention Model for Metro Passenger Flow Prediction

面向地铁客流预测的动态时空混合注意力模型（DST-HA）参考实现。模型通过分层空间注意力（全局/局部）与上下文感知门控结合编码器-解码器架构，实现多步客流预测。

- 训练入口：[run_model.py](run_model.py)
- 训练与评估逻辑：[train.py](trainer/train.py)
- 模型实现：[DST_HA.py](model/DST_HA.py)
- 数据加载：杭州 [HZMetro.py](loader/HZMetro.py)，上海 [SHMetro.py](loader/SHMetro.py)
- 默认配置：杭州 [hz.yaml](config/hz.yaml)，上海 [sh.yaml](config/sh.yaml)


### 1. 环境准备

- Python 3.9–3.11
- PyTorch 2.x（建议CUDA）
- 其他依赖见 [requirements.txt](requirements.txt)

安装示例：

```bash
pip install -r requirements.txt
# 如需GPU，请根据本机CUDA版本安装匹配的torch（参考PyTorch官网或国内镜像）。
```


### 2. 数据获取与放置

- 数据集（dataset.zip）：
  - 链接: [https://pan.baidu.com/s/15jxJAIkTCSJA55wWdmkDnw](https://pan.baidu.com/s/1Ukn5JTAx0rOyRtCawlv58g?pwd=1hqa)
  - 提取码: 1hqa
- 解压后将 data 目录置于项目根目录，使结构包含：

```
DST-HA/
  data/
    hangzhou/
      train.pkl  val.pkl  test.pkl
      restday.csv  weather.xlsx
      graph_hz_conn.pkl  graph_hz_cor.pkl  graph_hz_sml.pkl
    shanghai/
      train.pkl  val.pkl  test.pkl
      restday.csv  weather.xlsx
      graph_sh_conn.pkl  graph_sh_cor.pkl  graph_sh_sml.pkl
```

配置文件中的 dataset.root 已预设：
- 杭州：config/hz.yaml → data/hangzhou
- 上海：config/sh.yaml → data/shanghai


### 3. 预训练模型（可选）

- 模型权重（model.zip）：
  - 链接: https://pan.baidu.com/s/1ty0WJmZgDV7nOUCei5y_7w?pwd=wspm
  - 提取码: wspm
- 解压到 log/ 目录，例如：
```
log/
  hz/ best.pt
  sh/ best.pt
```
使用时将目录传给 --load_param（目录内需包含 best.pt）。


### 4. 快速开始

训练（从头或继续）：
```bash
# 杭州（默认CUDA，如需CPU改为 --device cpu）
python run_model.py --config ./config/hz.yaml --device cuda --log_dir ./log/hz

# 上海
python run_model.py --config ./config/sh.yaml --device cuda --log_dir ./log/sh

# 载入预训练权重（目录内需有 best.pt）
python run_model.py --config ./config/hz.yaml --device cuda --log_dir ./log/hz --load_param ./log/hz
```

评估/测试：
- 训练过程中会自动在验证/测试集上评估并保存最优权重至 log/.../best.pt。
- 单独脚本：
  - 简单评估：[evaluation.py](evaluation.py)
  - 指定站点/时段评估：[evaluation_pick.py](evaluation_pick.py)


### 5. 消融实验（可选）

- 配置位于 config/ablation/
- 一键运行所有实验并汇总结果：
```bash
python ./scripts/run_ablation_experiments.py
```
结果保存在 results/ablation/（含CSV与LaTeX表格）。


### 6. 配置要点

以下字段在 [hz.yaml](config/hz.yaml) / [sh.yaml](config/sh.yaml) 中常用：
- loader: 'hz' 或 'sh'（选择数据集）
- dataset.root: 数据根目录
- model.*:
  - num_nodes（杭州80，上海288）、n_heads、st_layers、use_global、use_dynamic_gate、use_curriculum_learning 等
- train.*:
  - epoch, base_lr, steps, warm_up 等
  - 可选：loss_type（'weighted_mae' 或 'combined'）、mape_mode（'standard'/'symmetric'/'weighted'）、use_mape_optimizer


### 7. 日志与结果

- 日志与权重保存在 --log_dir 指定目录（默认根据配置名自动创建，如 log/hz）。
- 每隔 save_every_n_epochs 保存一次 epoch-*.pt 与对应配置快照 config-*.yaml，并保存全程最佳 best.pt。


### 8. 代码索引

- 训练入口：[run_model.py](run_model.py)
- 训练与评估：[train.py](trainer/train.py)
- 模型主体：[DST_HA.py](model/DST_HA.py)
- 空间注意力：[spatial_attention.py](model/spatial_attention.py)
- 数据加载：[HZMetro.py](loader/HZMetro.py), [SHMetro.py](loader/SHMetro.py)
- 运行消融：[run_ablation_experiments.py](scripts/run_ablation_experiments.py)


如遇问题，请首先检查数据路径、PyTorch/CUDA 环境是否匹配，并查看对应 log/log.txt 以定位。 
