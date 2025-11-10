from utils.Env.Isaac_Lab_Env import Isaac_Lab_Environment
from utils.PPO.Actor_Critic import Actor_Critic
from utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train
AC_trained = Actor_Critic(PPO_Config, Env_Config,index=0)
AC_trained.load_best_model()
AC = Actor_Critic(PPO_Config, Env_Config,index=1)
if not train:
    AC.load_best_model()
env = Isaac_Lab_Environment(Env_Config, Robot_Config, PPO_Config)
import torch

env.prim_initialization(reset_all=True)
for epi in range(episode):
    print(f"===================episode: {epi}===================")
    for step in range(maximum_step):
        """获取当前状态"""

        state = env.get_current_observations()

        state_trained = state.clone()
        state_trained[:,32:] = 0

        """做动作"""

        action1, scaled_action1 = AC_trained.sample_action(state_trained)
        action2, scaled_action2 = AC.sample_action(state)

        """更新环境"""
        env.update_world(action=scaled_action1*0.25+scaled_action2*0.75)

        """获取下一个状态"""

        next_state = env.get_next_observations()

        """计算奖励 判断是否结束"""

        reward, over, extra_over = env.compute_reward()

        """存储经验"""
        if train:
            AC.store_experience(state,
                                action2,
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
