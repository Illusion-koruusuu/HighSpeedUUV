# HighSpeedUUV

这是 HoloOcean Fossen Torpedo 动力学上的三维 UUV 导航项目，任务奖励使用
Dijkstra-v5 代价场。训练不启动 Unreal/HoloOcean 世界；声呐使用 HoloOcean 配置的
三路视场与量程，通过球形障碍物几何射线生成，不包含回波强度渲染。

## 目录

- `highspeed_uuv/`: 环境、HoloOcean 动力学适配、Dijkstra、三路声呐和 CNN 编码器。
- `training/train.py`: 三路声呐 CNN + 普通 PPO 训练，支持 `--resume`。
- `training/evaluate.py`: 评估模型并把 CSV、PNG 输出到一个目录。
- `runs/ppo_sonar_dijkstra_v1/`: 新模型的默认输出目录。

## 运行

请先在安装了 HoloOcean Fossen、Stable-Baselines3 和 PyTorch 的环境中安装本项目：

```bash
pip install -e '.[train]'
python -m training.train --timesteps 1000000 --output runs/ppo_sonar_dijkstra_v5
python -m training.evaluate runs/ppo_sonar_dijkstra_v5/ppo_sonar.zip --output runs/ppo_sonar_test
```

训练默认每回合随机生成 2～5 个障碍物；可用 `--min-obstacles` 和
`--max-obstacles` 修改训练范围，评估时使用同名参数。障碍物中心和半径独立随机，
不同障碍物之间允许重叠。

`z` 是 NED 深度坐标，正方向向下。Dijkstra 规划区域固定为
`x[-20,100]、y[-70,70]、z[0,40]`，越界立即结束回合并处罚。当前目标捕获球半径为
6 m。远处回波仍输入 CNN，但只有前向 18 m、舷侧 10 m 内的回波才触发即时避障
状态；清水远距离时最低油门对应约 2 m/s，近障碍时允许降到约 1.6 m/s，避免艇
为了避障把推进完全关掉而失去舵面机动能力。

## 声呐输入

每路图包含归一化距离与回波掩码两个通道，三路合并为 `(6, 24, 64)`：

- 前视 `ImagingSonar` 几何视场：水平 120°、垂直 30°。
- 左右 `SidescanSonar` 几何视场：NED 艇体系中分别朝向 -90°/+90°，
  水平 120°、垂直 5°。
- 三路量程均为 0.5～40 m；CNN 将声呐图特征与目标/速度状态融合。

这些是训练端几何传感器的配置，不是 Unreal 中实际渲染的声呐图像。尤其是
HoloOcean 原生 `SidescanSonar` 输出一维回波强度，本项目将侧向视场采样成二维
角度-距离图以便 CNN 使用；将来接入真实 HoloOcean 回波时需要做输入适配和验证。
声呐噪声和失真、LSTM 均未启用。训练以 0～2 个障碍物开始，逐渐增加到目标
障碍物范围。奖励使用 VecNormalize 归一化，价值/回报诊断写入输出目录下的
`logs/progress.csv`。由于奖励版本已更新，旧模型不能用于 resume；请从头训练。

目标 20 m 内额外使用绝对方向误差、径向距离进展和目标方向速度奖励，避免只靠
角度变化奖励导致潜航器在目标附近等距转圈。

## v6 兼容评估参数

`runs/ppo_sonar_dijkstra_v6/run.json` 中的 `evaluation_assist` 记录了一组经过
50 回合验证的低层辅助参数：开启目标方向引导与末端捕获辅助，捕获转向距离设为
30 m、捕获油门下限设为 0.2、舵向下限设为 0.5。`training.evaluate` 会自动读取
这些参数；评估命令仍为：

```bash
/home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.evaluate \
  runs/ppo_sonar_dijkstra_v6/ppo_sonar.zip --episodes 50
```

对应 `runs/ppo_test/evaluation_metrics.csv` 的 50 回合结果为成功率 90%、碰撞率
0%、越界率 6%、超时率 4%。这是同一 v6 权重配合低层辅助参数的结果，不代表去掉
辅助后的纯 PPO 成功率。

## 纯 PPO 重新训练

`training/train.py` 的默认配置现在是纯 PPO：`goal_guidance=False`、
`capture_assist=False`、`cruise_assist=False`。策略只根据三路声呐、目标状态和
速度输出舵角/油门，不再依赖运行期规则导向。为改善训练稳定性，入口同时加入了：

- 有界 `tanh` 高斯动作分布，避免环境裁剪导致动作概率与执行动作不一致；
- `ent_coef=0.001`、`log_std_init=-1.0`，降低长期随机探索；
- 18–30 m 无障碍起步，逐步扩展到 50–70 m、2–5 个障碍物的四阶段课程；
- 每 25k 步用 10 个固定种子做确定性评估，并保留成功率最高的 checkpoint；
- 初始动作均值偏向 0.6 油门，但该偏置仍可被 PPO 完全学习修改。

建议先运行 100 万步：

```bash
cd /home/lu/code/HighSpeedUUV
/home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.train \
  --timesteps 1000000 --output runs/ppo_sonar_pure_v7 --overwrite
```

训练结束后 `ppo_sonar.zip` 会自动替换为周期性评估中成功率最高的模型，最终模型
保存在 `ppo_sonar_final.zip`，评估记录在 `best_eval.json`。纯 PPO 评估命令：

```bash
/home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.evaluate \
  runs/ppo_sonar_pure_v7/ppo_sonar.zip --episodes 50
```

### v8：路径观测与防遗忘训练

v7 纯 PPO 的 50 回合成功率约为 14%，80% 为超时，失败回合最小距离中位数约
18.7 m，说明主要问题是全局导航没有稳定学会。v8 训练入口默认加入：

- 策略观测新增 4 维局部 Dijkstra 路径信息：路径俯仰、路径偏航、8 m 前视距离、
  剩余路径代价；PPO 仍直接输出舵角和油门，没有运行期动作控制器；
- 每回合以 25% 概率抽取 25–40 m、无障碍的短距离复习场景，防止后期课程覆盖基础
  导航能力；
- 策略 MLP 从 `[128, 128]` 增大到 `[256, 256]`。

v8 观测空间与 v7 不兼容，必须从头训练：

```bash
cd /home/lu/code/HighSpeedUUV
/home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.train \
  --timesteps 1500000 --output runs/ppo_sonar_pure_v8 --overwrite
```

### v9：教师 BC 预热 + 纯 PPO 微调

v8 的 500k 步实验暴露了纯 PPO 从零探索的瓶颈：路径偏航约 `0.79 rad` 时，
策略方向舵长期只有 `±0.03`，无法学会持续转弯，最终高速越界。使用同一 `path`
观测的简单路径跟随教师则可达到约 85%，说明问题在策略优化而不是环境。

因此 v9 默认先执行 5,000 步行为克隆，仅用于初始化神经网络：

- 教师根据局部路径偏航/俯仰和目标速度误差生成舵角、油门；
- BC 损失优先拟合方向舵，100 个 epoch 后动作 MSE 可降到约 `2×10⁻⁴`；
- BC 模型会先单独做确定性评估并作为初始最佳 checkpoint；
- 随后以较低学习率 `1e-4` 做纯 PPO 微调；执行阶段完全没有教师或规则控制器；
- 周期性评估仍保留成功率最高的模型，因此 PPO 微调退化时不会丢掉更好的 BC 初始化模型。

已验证的 BC 初始化模型在完整难度 50 回合中成功率为 **96%**（48/50），
模型保存在 `runs/ppo_sonar_bc_init_v9/ppo_sonar.zip`。正式训练命令：

```bash
/home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.train \
  --timesteps 500000 --output runs/ppo_sonar_pure_v9 --overwrite
```

## HoloOcean 可视化回放

`scripts/visualize.sh` 会启动 HoloOcean 窗口，加载与评估相同的随机场景，并用
策略实时控制 `TorpedoAUV`。金色球体是训练场景中的障碍物，白色球体是 6 m 目标
捕获区域。

```bash
cd /home/lu/code/HighSpeedUUV
./scripts/visualize.sh runs/ppo_sonar_pure_v7/ppo_sonar.zip \
  --seed 0 --seconds 60
```

如果需要同时显示 HoloOcean 原生前视/侧扫声呐，可加 `--render-sonar`：

```bash
./scripts/visualize.sh runs/ppo_sonar_pure_v7/ppo_sonar.zip \
  --seed 0 --seconds 60 --render-sonar
```

策略在训练和回放中仍使用几何三路声呐，HoloOcean 原生声呐图像目前只用于视觉
检查，不作为策略输入。

=================================================================================

# 最大速度测试
按当前 HoloOcean Torpedo Fossen 模型测试，满油门、无障碍、水平舵和方向舵为零时：

| 配置 | `nMax` | 稳态速度 | 约合节速 |
|---|---:|---:|---:|
| `standard` | 1525 RPM | 2.55 m/s | 4.96 节 |
| 当前 `boosted` | 3000 RPM | 5.02 m/s | 9.76 节 |

因此，当前项目配置下实际最快约为：

```text
5.0 m/s ≈ 9.8 节 ≈ 18.1 km/h
```

这个速度是推进力和水动力阻力平衡后的稳态速度，不是简单的速度上限。测试过程中环境可能因为固定规划边界 `x=100 m` 提前终止，但终止前速度已经稳定在约 `5.02 m/s`。

参数位于：

[vehicle_profile.py](/home/lu/code/HighSpeedUUV/highspeed_uuv/vehicle_profile.py)

当前 `boosted` 主要只修改了：

```python
{"nMax": 3000}
```

我进一步做了参数外推测试：

| `nMax` | 稳态速度 |
|---:|---:|
| 3000 RPM | 5.02 m/s |
| 4000 RPM | 6.70 m/s |
| 5000 RPM | 8.37 m/s |
| 6000 RPM | 10.05 m/s |

在当前模型中，速度大致随转速上限近似线性增加。

如果目标是 **70 节**：

```text
70 节 ≈ 36.0 m/s
```

按照当前模型粗略外推，需要大约：

```text
nMax ≈ 21000～22000 RPM
```

这已经不是普通 REMUS100 参数的合理范围，而是模型公式的极端外推。

要提高速度，有三种层次：

1. **简单提高 `nMax`**

   例如把：

   ```python
   {"nMax": 3000}
   ```

   改为 4000、5000 或 6000 RPM。

   优点是改动简单，速度会提高。缺点是物理可信度很差，推进器扭矩、空化、结构强度和水动力参数都没有同步校准。

2. **同步修改推进器和执行器模型**

   需要同时重新标定：

   - 螺旋桨推力系数；
   - 螺旋桨扭矩系数；
   - 推进器效率；
   - 舵面最大角度和响应速度；
   - 车体阻力系数；
   - 质量、惯量和附加质量；
   - 高速下的阻尼模型。

   这才是合理的高速潜航器模型修改方式。

3. **建立独立的高速 UUV Fossen 模型**

   如果目标确实是 70 节，应建立新的高速艇体参数，而不是继续把 REMUS100 的 `nMax` 拉到两万 RPM。需要重新定义：

   - 高速艇体尺寸和质量；
   - 推进器功率；
   - 高速阻力；
   - 舵面和控制面尺寸；
   - 高速姿态稳定性；
   - 对应的控制周期和训练场景。

结论是：当前模型可信的最高速度大约是 **5 m/s，也就是 10 节左右**。继续提高到 6～10 m/s 可以作为仿真外推实验，但要达到 70 节，必须更换或重建高速动力学模型，单独提高 `nMax` 会得到数值上更快、物理上不可信的结果。
