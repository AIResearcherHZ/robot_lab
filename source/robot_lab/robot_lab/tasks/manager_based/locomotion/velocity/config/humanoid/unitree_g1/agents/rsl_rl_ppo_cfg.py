from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class UnitreeG1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境每轮收集的时间步数，直接影响一个 rollout 的长度
    num_steps_per_env = 24
    # 最大训练轮数，超过则终止训练
    max_iterations = 3000
    # 模型保存间隔（单位为迭代次数）
    save_interval = 50
    # 实验名称，用于日志和模型目录区分
    experiment_name = "unitree_g1_rough"
    # 策略网络及价值网络配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始化动作噪声标准差
        actor_obs_normalization=False,  # 不对策略网络的观测进行归一化
        critic_obs_normalization=False,  # 不对价值网络的观测进行归一化
        actor_hidden_dims=[512, 256, 128],  # 策略网络隐藏层结构
        critic_hidden_dims=[512, 256, 128],  # 价值网络隐藏层结构
        activation="elu",  # 激活函数类型
    )
    # PPO 算法超参数设定
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # 价值损失权重
        use_clipped_value_loss=True,  # 使用裁剪的价值损失，提升稳定性
        clip_param=0.2,  # PPO 的裁剪参数
        entropy_coef=0.008,  # 熵项系数，鼓励探索
        num_learning_epochs=5,  # 每次 rollout 的训练迭代次数
        num_mini_batches=4,  # 分 minibatch 的数量
        learning_rate=1.0e-3,  # 基础学习率
        schedule="adaptive",  # 学习率调度方式
        gamma=0.99,  # 折扣因子
        lam=0.95,  # GAE 的 λ 参数
        desired_kl=0.01,  # 期望的 KL 发散，用于自适应学习率
        max_grad_norm=1.0,  # 梯度裁剪阈值，避免爆炸
    )

@configclass
class UnitreeG1FlatPPORunnerCfg(UnitreeG1RoughPPORunnerCfg):
    def __post_init__(self):
        # 调用父类初始化，确保继承的配置得到处理
        super().__post_init__()

        # 平坦地形训练的最大迭代次数更少，节省计算资源
        self.max_iterations = 1500
        # 对应的实验名称，用于单独区分日志与模型
        self.experiment_name = "unitree_g1_flat"