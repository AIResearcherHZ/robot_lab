# 扭矩记录功能实现总结

## 实现概述

已成功为训练和推理脚本添加了扭矩记录功能，支持通过键盘控制记录、自动识别关节名称、保存数据和生成可视化曲线图。

## 核心功能

### 1. 自动关节识别 ✅
- 自动从环境中提取关节名称（`env.scene["robot"].data.joint_names`）
- 自动识别任务配置信息（如 `UnitreeG1RoughEnvCfg`）
- 智能回退机制：无法提取时使用 `joint_0`, `joint_1` 等

### 2. 键盘控制 ✅
- 按 `,` 键开始记录
- 按 `.` 键停止记录并保存
- 使用 `pynput` 库实现全局键盘监听

### 3. 数据保存 ✅
- 保存为 `.npz` 格式，包含时间步和所有关节扭矩
- 包含元数据（任务名称、关节数量等）
- 自动生成时间戳文件名

### 4. 可视化 ✅
- 自动生成多子图布局
- 每个关节独立显示扭矩曲线
- 使用真实关节名称作为标题
- 保存为高分辨率 PNG 图片

### 5. 统计信息 ✅
- 显示记录的关节数量
- 显示前3个关节的统计信息（平均值、最大值、最小值）

## 修改的文件

### 新增文件
1. **`torque_recorder.py`** - 核心记录器模块
   - `TorqueRecorder` 类：主要记录逻辑
   - 自动环境信息提取
   - 键盘监听和数据保存
   - 曲线绘制功能

2. **`TORQUE_RECORDING_README.md`** - 使用文档
   - 功能说明
   - 使用示例
   - 故障排除

### 修改的文件
1. **`rsl_rl/train.py`**
   - 添加 `--record_torque` 参数
   - 初始化扭矩记录器（传入环境）
   - 关闭记录器

2. **`rsl_rl/play.py`**
   - 添加 `--record_torque` 参数
   - 初始化扭矩记录器（传入环境）
   - 在推理循环中记录扭矩
   - 关闭记录器

3. **`cusrl/train.py`**
   - 添加 `--record_torque` 参数
   - 初始化扭矩记录器（传入环境）
   - 关闭记录器

4. **`cusrl/play.py`**
   - 添加 `--record_torque` 参数
   - 添加 `TorqueRecorderHook` 类
   - 初始化扭矩记录器（传入环境）
   - 注册 hook 进行记录
   - 关闭记录器

## 使用示例

### RSL-RL 训练
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 4096 \
    --record_torque
```

### RSL-RL 推理
```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 64 \
    --record_torque
```

### CusRL 训练
```bash
python scripts/reinforcement_learning/cusrl/train.py \
    --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 4096 \
    --record_torque
```

### CusRL 推理
```bash
python scripts/reinforcement_learning/cusrl/play.py \
    --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 \
    --num_envs 50 \
    --record_torque
```

## 输出示例

```
[TorqueRecorder] 已启用扭矩记录器
[TorqueRecorder] 按 ',' 键开始记录，按 '.' 键结束记录并保存
[TorqueRecorder] 检测到 23 个关节
[TorqueRecorder] 环境信息: UnitreeG1RoughEnvCfg

[TorqueRecorder] ✓ 开始记录扭矩数据...
[TorqueRecorder] ✓ 停止记录，共记录 1500 步
[TorqueRecorder] 数据已保存至: logs/rsl_rl/experiment/torque_logs/torque_data_20241122_205030.npz
[TorqueRecorder] 记录了 23 个关节的数据
  - left_hip_pitch_joint: 平均=12.345, 最大=45.678, 最小=-23.456
  - left_hip_roll_joint: 平均=8.234, 最大=34.567, 最小=-12.345
  - left_hip_yaw_joint: 平均=5.678, 最大=23.456, 最小=-8.901
[TorqueRecorder] 曲线图已保存至: logs/rsl_rl/experiment/torque_logs/torque_plot_20241122_205030.png
```

## 技术细节

### 环境信息提取逻辑
```python
def _extract_env_info(self):
    # 1. 获取unwrapped环境
    env = self.env
    while hasattr(env, 'unwrapped'):
        env = env.unwrapped
    
    # 2. 提取任务名称
    if hasattr(env, 'cfg'):
        self.env_info['task_name'] = env.cfg.__class__.__name__
    
    # 3. 从scene中获取机器人信息
    if hasattr(env, 'scene'):
        robot = env.scene.get("robot")
        if robot and hasattr(robot, 'data'):
            if hasattr(robot.data, 'joint_names'):
                self.joint_names = list(robot.data.joint_names)
```

### 数据结构
保存的 `.npz` 文件包含：
- `time_steps`: 时间步数组
- `[joint_name]`: 每个关节的扭矩数组
- `metadata`: 环境元数据（可选）

### 可视化布局
- 自动计算子图布局（最多3列）
- 每个关节一个子图
- 显示关节名称、时间步和扭矩值

## 依赖项

```bash
pip install pynput matplotlib numpy torch
```

## 特点

1. **零配置**: 无需手动指定关节名称，自动从环境提取
2. **环境自适应**: 支持不同机器人（Unitree G1, A1等）和任务
3. **实时控制**: 通过键盘实时控制记录开始和结束
4. **性能友好**: 只在记录时才处理数据，对训练/推理性能影响极小
5. **完整信息**: 保存元数据，方便后续分析

## 扩展性

可以轻松扩展以支持：
- 记录其他物理量（位置、速度、加速度等）
- 不同的保存格式（CSV、HDF5等）
- 实时绘图显示
- 多环境并行记录
