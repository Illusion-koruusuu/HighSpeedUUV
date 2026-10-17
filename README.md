# HighSpeedUUV

这是当前版本的 HoloOcean-only 三维 UUV 导航项目。训练和验证统一使用
HoloOcean 的 `Torpedo` Fossen 动力学，任务奖励使用 Dijkstra-v5 代价场。
旧的 Python REMUS100 模型、欧式距离基线和历史训练脚本不属于本项目。

## 目录

- `highspeed_uuv/`: 环境、HoloOcean 动力学适配、Dijkstra、传感器和执行器配置。
- `training/train.py`: Dijkstra-v5 PPO 训练，支持 `--resume`。
- `training/evaluate.py`: 评估模型并把 CSV、PNG 输出到一个目录。
- `runs/ppo_dijkstra_v5/`: 当前保留的 v5 模型、配置和评估产物。

## 运行

请先在安装了 HoloOcean、Stable-Baselines3 和 PyTorch 的环境中安装本项目：

```bash
pip install -e '.[train]'
python -m training.train --timesteps 1000000 --output runs/ppo_dijkstra_v5
python -m training.evaluate runs/ppo_dijkstra_v5/ppo_dijkstra.zip --output runs/ppo_test
```

`z` 是 NED 深度坐标，正方向向下。Dijkstra 规划区域固定为
`x[-20,100]、y[-70,70]、z[0,40]`，越界立即结束回合并处罚。
