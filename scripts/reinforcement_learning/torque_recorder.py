# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""扭矩记录器模块，用于在训练和推理时记录和可视化扭矩数据"""

import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from pynput import keyboard

class TorqueRecorder:
    """扭矩记录器，使用pynput监听键盘输入控制记录"""

    def __init__(self, enabled: bool = False, save_dir: Optional[str] = None, env=None):
        """
        初始化扭矩记录器
        
        Args:
            enabled: 是否启用扭矩记录功能
            save_dir: 保存目录，如果为None则使用当前目录下的torque_logs
            env: 环境实例，用于自动提取关节信息
        """
        self.enabled = enabled
        self.is_recording = False
        self.torque_data = defaultdict(list)  # {joint_name: [torque_values]}
        self.time_steps = []
        self.current_step = 0
        self.env = env
        self.joint_names = None
        self.env_info = {}
        
        # 设置保存目录
        if save_dir is None:
            self.save_dir = os.path.join(os.getcwd(), "torque_logs")
        else:
            self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 自动提取环境信息
        if self.enabled and self.env is not None:
            self._extract_env_info()
        
        # 启动键盘监听
        if self.enabled:
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.start()
            print("[TorqueRecorder] 已启用扭矩记录器")
            print("[TorqueRecorder] 按 ',' 键开始记录，按 '.' 键结束记录并保存")
            if self.joint_names:
                print(f"[TorqueRecorder] 检测到 {len(self.joint_names)} 个关节")
            if self.env_info:
                print(f"[TorqueRecorder] 环境信息: {self.env_info.get('task_name', 'Unknown')}")
    
    def _extract_env_info(self):
        """从环境中提取关节名称和其他信息"""
        try:
            # 获取unwrapped环境
            env = self.env
            while hasattr(env, 'unwrapped'):
                env = env.unwrapped
            
            # 尝试获取任务名称
            if hasattr(env, 'cfg'):
                cfg = env.cfg
                if hasattr(cfg, '__class__'):
                    self.env_info['task_name'] = cfg.__class__.__name__
            
            # 尝试从scene中获取机器人信息
            if hasattr(env, 'scene'):
                try:
                    robot = env.scene.get("robot") or env.scene.get("articulation")
                    if robot is not None:
                        # 获取关节名称
                        if hasattr(robot, 'data'):
                            if hasattr(robot.data, 'joint_names'):
                                self.joint_names = list(robot.data.joint_names)
                            elif hasattr(robot, 'joint_names'):
                                self.joint_names = list(robot.joint_names)
                        
                        # 获取机器人名称
                        if hasattr(robot, 'cfg') and hasattr(robot.cfg, 'prim_path'):
                            self.env_info['robot_path'] = robot.cfg.prim_path
                except Exception as e:
                    pass
            
            # 如果还没有获取到关节名称，尝试其他方法
            if self.joint_names is None:
                # 尝试从articulation中获取
                if hasattr(env, '_robot'):
                    robot = env._robot
                    if hasattr(robot, 'data') and hasattr(robot.data, 'joint_names'):
                        self.joint_names = list(robot.data.joint_names)
                
                # 尝试从配置中获取
                if self.joint_names is None and hasattr(env, 'cfg'):
                    cfg = env.cfg
                    if hasattr(cfg, 'joint_names'):
                        self.joint_names = list(cfg.joint_names)
            
            # 记录成功提取的信息
            if self.joint_names:
                self.env_info['num_joints'] = len(self.joint_names)
                self.env_info['joint_names'] = self.joint_names
        
        except Exception as e:
            print(f"[TorqueRecorder] 警告: 无法自动提取环境信息: {e}")
    
    def _on_key_press(self, key):
        """键盘按键回调"""
        try:
            if hasattr(key, 'char'):
                if key.char == ',':
                    self._start_recording()
                elif key.char == '.':
                    self._stop_recording()
        except AttributeError:
            pass
    
    def _start_recording(self):
        """开始记录"""
        if not self.is_recording:
            self.is_recording = True
            self.torque_data.clear()
            self.time_steps.clear()
            self.current_step = 0
            print("\n[TorqueRecorder] ✓ 开始记录扭矩数据...")
    
    def _stop_recording(self):
        """停止记录并保存数据"""
        if self.is_recording:
            self.is_recording = False
            print(f"[TorqueRecorder] ✓ 停止记录，共记录 {self.current_step} 步")
            self._save_and_plot()
    
    def record_step(self, torques: torch.Tensor, joint_names: Optional[list] = None):
        """
        记录一个时间步的扭矩数据
        
        Args:
            torques: 扭矩张量，形状为 (num_envs, num_joints) 或 (num_joints,)
            joint_names: 关节名称列表，如果为None则自动使用环境中提取的名称
        """
        if not self.enabled or not self.is_recording:
            return
        
        # 转换为numpy数组
        if isinstance(torques, torch.Tensor):
            torques = torques.detach().cpu().numpy()
        
        # 处理形状
        if torques.ndim == 2:
            # 取第一个环境的数据
            torques = torques[0]
        
        # 确定使用的关节名称
        num_joints = len(torques)
        if joint_names is None:
            # 优先使用初始化时提取的关节名称
            if self.joint_names is not None and len(self.joint_names) == num_joints:
                joint_names = self.joint_names
            else:
                joint_names = [f"joint_{i}" for i in range(num_joints)]
        
        # 记录数据
        for i, (joint_name, torque_value) in enumerate(zip(joint_names, torques)):
            self.torque_data[joint_name].append(float(torque_value))
        
        self.time_steps.append(self.current_step)
        self.current_step += 1
    
    def _save_and_plot(self):
        """保存数据并绘制曲线"""
        if not self.torque_data:
            print("[TorqueRecorder] 没有记录到数据")
            return
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备保存的数据
        save_data = {
            'time_steps': np.array(self.time_steps),
            **{name: np.array(values) for name, values in self.torque_data.items()}
        }
        
        # 添加元数据
        if self.env_info:
            save_data['metadata'] = np.array([str(self.env_info)])
        
        # 保存原始数据
        data_file = os.path.join(self.save_dir, f"torque_data_{timestamp}.npz")
        np.savez(data_file, **save_data)
        print(f"[TorqueRecorder] 数据已保存至: {data_file}")
        
        # 打印统计信息
        print(f"[TorqueRecorder] 记录了 {len(self.torque_data)} 个关节的数据")
        for joint_name, values in list(self.torque_data.items())[:3]:  # 只显示前3个
            values_array = np.array(values)
            print(f"  - {joint_name}: 平均={values_array.mean():.3f}, 最大={values_array.max():.3f}, 最小={values_array.min():.3f}")
        
        # 绘制曲线
        self._plot_torques(timestamp)
    
    def _plot_torques(self, timestamp: str):
        """绘制扭矩曲线"""
        num_joints = len(self.torque_data)
        if num_joints == 0:
            return
        
        # 计算子图布局
        ncols = min(3, num_joints)
        nrows = (num_joints + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
        if num_joints == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes
        
        # 绘制每个关节的扭矩曲线
        for idx, (joint_name, torque_values) in enumerate(self.torque_data.items()):
            ax = axes[idx]
            ax.plot(self.time_steps, torque_values, linewidth=1.5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Torque (N·m)')
            ax.set_title(f'{joint_name}')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(num_joints, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.save_dir, f"torque_plot_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[TorqueRecorder] 曲线图已保存至: {plot_file}")
        
        # 显示图片（可选，在无GUI环境下会失败）
        try:
            plt.show(block=False)
            plt.pause(0.1)
        except:
            pass
        finally:
            plt.close()
    
    def close(self):
        """关闭记录器"""
        if self.enabled:
            if self.is_recording:
                self._stop_recording()
            if hasattr(self, 'listener') and self.listener is not None:
                self.listener.stop()
            print("[TorqueRecorder] 扭矩记录器已关闭")

# 全局记录器实例
_global_recorder: Optional[TorqueRecorder] = None

def get_torque_recorder() -> Optional[TorqueRecorder]:
    """获取全局扭矩记录器实例"""
    return _global_recorder


def init_torque_recorder(enabled: bool = False, save_dir: Optional[str] = None, env=None) -> TorqueRecorder:
    """
    初始化全局扭矩记录器
    
    Args:
        enabled: 是否启用扭矩记录功能
        save_dir: 保存目录
        env: 环境实例，用于自动提取关节信息
    
    Returns:
        TorqueRecorder实例
    """
    global _global_recorder
    if _global_recorder is not None:
        _global_recorder.close()
    _global_recorder = TorqueRecorder(enabled=enabled, save_dir=save_dir, env=env)
    return _global_recorder


def close_torque_recorder():
    """关闭全局扭矩记录器"""
    global _global_recorder
    if _global_recorder is not None:
        _global_recorder.close()
        _global_recorder = None
