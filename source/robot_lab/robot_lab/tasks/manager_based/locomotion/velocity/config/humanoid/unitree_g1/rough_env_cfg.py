from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG  # isort: skip

from isaaclab.utils import configclass
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards

@configclass
class UnitreeG1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "torso_link"  # 基座链接
    foot_link_name = ".*_ankle_roll_link"  # 脚部链接
    # fmt: off
    # joint_names = [
    #     "left_hip_pitch_joint",          # 0  L_LEG_HIP_PITCH
    #     "left_hip_roll_joint",           # 1  L_LEG_HIP_ROLL
    #     "left_hip_yaw_joint",            # 2  L_LEG_HIP_YAW
    #     "left_knee_joint",               # 3  L_LEG_KNEE
    #     "left_ankle_pitch_joint",        # 4  L_LEG_ANKLE_B
    #     "left_ankle_roll_joint",         # 5  L_LEG_ANKLE_A
    #     "right_hip_pitch_joint",         # 6  R_LEG_HIP_PITCH
    #     "right_hip_roll_joint",          # 7  R_LEG_HIP_ROLL
    #     "right_hip_yaw_joint",           # 8  R_LEG_HIP_YAW
    #     "right_knee_joint",              # 9  R_LEG_KNEE
    #     "right_ankle_pitch_joint",       # 10 R_LEG_ANKLE_B
    #     "right_ankle_roll_joint",        # 11 R_LEG_ANKLE_A
    #     "waist_yaw_joint",               # 12 WAIST_YAW
    #     "left_shoulder_pitch_joint",     # 15 L_SHOULDER_PITCH
    #     "left_shoulder_roll_joint",      # 16 L_SHOULDER_ROLL
    #     "left_shoulder_yaw_joint",       # 17 L_SHOULDER_YAW
    #     "left_elbow_pitch_joint",        # 18 L_ELBOW
    #     "left_elbow_roll_joint",         # 19 L_WRIST_ROLL
    #     "right_shoulder_pitch_joint",    # 22 R_SHOULDER_PITCH
    #     "right_shoulder_roll_joint",     # 23 R_SHOULDER_ROLL
    #     "right_shoulder_yaw_joint",      # 24 R_SHOULDER_YAW
    #     "right_elbow_pitch_joint",       # 25 R_ELBOW
    #     "right_elbow_roll_joint",        # 26 R_WRIST_ROLL
    # ]
    # fmt: on

    def __post_init__(self):
        # 父类初始化
        super().__post_init__()

        # ------------------------------场景设置------------------------------
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # 载入机器人配置
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name  # 高度扫描器
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name  # 基座扫描器

        # ------------------------------观测设置------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0  # 线速度缩放
        self.observations.policy.base_ang_vel.scale = 0.25  # 角速度缩放
        self.observations.policy.joint_pos.scale = 1.0  # 关节位置缩放
        self.observations.policy.joint_vel.scale = 0.05  # 关节速度缩放
        self.observations.policy.base_lin_vel = None  # 关闭部分观测
        self.observations.policy.height_scan = None  # 关闭高度扫描
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------动作设置------------------------------
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE  # 行为缩放
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}  # 动作裁剪
        # self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------事件设置------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]  # 随机化底座质量
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]  # 随机化其他部件质量
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]  # 随机质心
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]  # 外力扰动

        # ------------------------------奖励设置------------------------------
        # 总体终止代价
        self.rewards.is_terminated.weight = -200.0  # 终止时惩罚较大，鼓励稳定运行

        # 根部相关代价
        self.rewards.lin_vel_z_l2.weight = 0  # 忽略垂直速度
        self.rewards.ang_vel_xy_l2.weight = -0.1  # 控制横滚俯仰角速度
        self.rewards.flat_orientation_l2.weight = -0.2  # 惩罚机器人倾斜
        self.rewards.base_height_l2.weight = 0  # 高度偏差暂不考虑
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0  # 不惩罚线加速度
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 关节相关代价
        self.rewards.joint_torques_l2.weight = -1.5e-7  # 轻微惩罚关节力矩
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        self.rewards.joint_vel_l2.weight = 0  # 速度惩罚关闭
        self.rewards.joint_acc_l2.weight = -1.25e-7  # 轻惩加速度变化
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*hip_yaw.*", ".*hip_roll.*"])  # 臀部偏差
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_arms_l1", -0.1, [".*shoulder.*", ".*elbow.*"])  # 手臂偏差
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_torso_l1", -0.1, ["waist_yaw_joint"])  # 腰部偏差
        self.rewards.joint_pos_limits.weight = -0.5  # 关节位置边界惩罚
        self.rewards.joint_vel_limits.weight = 0  # 速度边界暂不惩罚
        self.rewards.joint_power.weight = 0  # 功率惩罚关闭
        self.rewards.stand_still.weight = 0  # 不鼓励静止
        self.rewards.joint_pos_penalty.weight = -1.0  # 关节位置惩罚
        self.rewards.joint_mirror.weight = 0  # 镜像惩罚关闭
        self.rewards.joint_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # 动作变化惩罚
        self.rewards.action_rate_l2.weight = -0.005  # 限制动作变化速率
        self.rewards.action_mirror.weight = 0  # 动作镜像惩罚关闭
        self.rewards.action_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle).*", "right_(hip|knee|ankle).*"]]

        # 接触传感器相关惩罚
        self.rewards.undesired_contacts.weight = 0  # 默认不惩罚非脚部接触
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0  # 接触力不纳入奖励
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # 速度追踪奖励
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 强化线速度平面追踪
        self.rewards.track_lin_vel_xy_exp.func = rewards.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0  # 强化角速度追踪
        self.rewards.track_ang_vel_z_exp.func = rewards.track_ang_vel_z_world_exp

        # 其他奖励设置
        self.rewards.feet_air_time.weight = 0.25  # 奖励脚部空中时间
        self.rewards.feet_air_time.func = rewards.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0  # 脚部接地奖励关闭
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0  # 无命令下接触
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0  # 脚部绊倒惩罚关闭
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2  # 滑动惩罚
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0  # 高度奖励关闭
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0  # 身体高度奖励关闭
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0  # 鼓励身体向上姿态

        # 奖励权重为零时自动禁用
        if self.__class__.__name__ == "UnitreeG1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------终止条件------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]  # 非脚部接触终止

        # ------------------------------课程训练------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None  # 关闭线速度课程
        self.curriculum.command_levels_ang_vel = None  # 关闭角速度课程

        # ------------------------------指令设置------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # 线速度范围 X
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)  # 线速度范围 Y
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # 角速度范围 Z