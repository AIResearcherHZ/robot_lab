from isaaclab.utils import configclass
from .rough_env_cfg import UnitreeG1RoughEnvCfg

@configclass
class UnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):
    def __post_init__(self):
        # 调用父类后处理
        super().__post_init__()

        # 重新配置奖励
        self.rewards.base_height_l2.params["sensor_cfg"] = None  # 取消高度传感器
        # 地形改为平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # 关闭高度扫描
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # 取消地形课程
        self.curriculum.terrain_levels = None

        # -------------奖励部分（多注释）----------------
        self.rewards.track_ang_vel_z_exp.weight = 1.0  # 鼓励Z轴角速度追踪
        self.rewards.lin_vel_z_l2.weight = -0.2  # 惩罚垂直线速度偏差
        self.rewards.action_rate_l2.weight = -0.005  # 控制动作变化平滑
        self.rewards.joint_acc_l2.weight = -1.0e-7  # 细微惩罚关节加速度变化
        self.rewards.joint_torques_l2.weight = -2.0e-6  # 强化对力矩的约束
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]  # 仅约束腿部

        # 奖励权重为0时禁用
        if self.__class__.__name__ == "UnitreeG1FlatEnvCfg":
            self.disable_zero_weight_rewards()