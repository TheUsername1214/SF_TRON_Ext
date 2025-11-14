class Env_Config:
    class EnvParam:  # 训练环境的参数
        agents_num = 2000
        agents_num_in_play = 10
        file_path = "C:/Users/21363/PycharmProjects/Isaac_Lab/SF_TRON_Ext/model/Robot_Model/SF_TRON1A.usd"  # abs path, not relative path
        dt = 0.025
        sub_step = 5
        friction_coef = 1
        device = 'cuda'
        backend = "torch"
        headless = False  # True: no GUI, False: GUI
        train = headless


class Robot_Config:
    class ActuatorParam:  # 机器人的参数
        Kp = [60, 60, 60, 60, 60, 60, 30, 30]
        Kd = [5, 5, 5, 5, 5, 5, 2.5, 2.5]  # Do not try to reduce Kd, because the action scale is not 0.25 but 1
        default_PD_angle = [[0, 0,
                             -0, 0,
                             0, 0,
                             0, 0]]

        actuator_num = 8

    class InitialState:
        initial_height = 0.85
        initial_body_linear_vel_range = 0.2
        initial_body_angular_vel_range = 0.2
        initial_joint_pos_range = 0.2
        initial_joint_vel_range = 0.2
        initial_joint_angle = [0, 0,
                               -0, 0,
                               0, 0,
                               0, 0]

class PPO_Config:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 33 + 17 * 11 # 机器人本体与外部指令感知
        critic_layers_num = 256
        critic_lr = 3e-4
        critic_update_frequency = 200

    class ActorParam:  # Actor 神经网络 参数
        action_scale = 1
        std_scale = 0.5
        act_layers_num = 256
        actuator_num = Robot_Config.ActuatorParam.actuator_num
        actor_lr = 3e-4
        actor_update_frequency = 100

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        maximum_step = 50
        episode = 300
        entropy_coef = -0.05  # positive means std increase, else decrease
        batch_size = 10000
