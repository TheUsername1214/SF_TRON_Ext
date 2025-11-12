from utils.Env.Isaac_Lab_Env import Isaac_Lab_Environment
from utils.PPO.Actor_Critic import Actor_Critic
from utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train
basic_state = PPO_Config.CriticParam.state_dim
AC = Actor_Critic(PPO_Config, Env_Config)
if not train:
    AC.load_best_model()

env = Isaac_Lab_Environment(Env_Config, Robot_Config, PPO_Config)
import torch

env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    env.resample_command()
    for step in range(maximum_step):
        """获取当前状态"""

        if not train:
            env.vel_cmd[:] = 1
            if epi>2:
                env.vel_cmd[:] = 0
                print("stop!!!!!!!")
        state = env.get_current_observations()
        state[:,basic_state:] = 0 # basic state 之后就是地图信息，第一阶段机器人盲走

        """做动作"""
        action, scaled_action = AC.sample_action(state)

        """更新环境"""
        env.update_world(action=scaled_action)

        """获取下一个状态"""

        next_state = env.get_next_observations()
        next_state[:, basic_state:] = 0

        """计算奖励 判断是否结束"""

        reward, over, extra_over = env.compute_reward()


        """存储经验"""
        if train:
            AC.store_experience(state,
                                action,
                                next_state,
                                reward,
                                over,
                                step)

        """重置挂掉的机器人"""
        over += extra_over
        env.prim_initialization(torch.nonzero(over.flatten()).flatten())

    """每个回合结束后训练一次"""
    if train:
        AC.update()
        env.print_reward_sum()
