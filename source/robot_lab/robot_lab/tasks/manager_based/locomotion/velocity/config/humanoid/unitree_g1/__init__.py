import gymnasium as gym  # 导入 Gymnasium 库
from . import agents  # 导入当前包中的 agents 模块

##
#  注册 Gym 环境
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeG1RoughEnvCfg",  # 粗糙地形环境配置
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1RoughPPORunnerCfg",  # PPO 运行器配置
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeG1RoughTrainerCfg",  # 自定义 RL 训练配置
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1FlatEnvCfg",  # 平坦地形环境配置
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1FlatPPORunnerCfg",  # PPO 运行器配置
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeG1FlatTrainerCfg",  # 自定义 RL 训练配置
    },
)