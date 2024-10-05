import numpy as np
from envs.env_core import EnvCore
from envs.multi_discrete import MultiDiscrete
from gym.spaces import Box

class DiscreteActionEnv(object):
    """
    Wrapper for discrete action environment.
    """

    def __init__(self):
        self.env_core = EnvCore()
        self.map = self.env_core.map
        self.num_agent = self.map.num_couriers

        self.signal_obs_dim = self.env_core.obs_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True

        self.movable = True

        # configure spaces
        self.action_space = self.env_core.action_space
        self.observation_space = []
        # self.share_observation_space = []

        # share_obs_dim = self.env_core.map.num_orders * 6
        for agent_idx in range(self.num_agent):

            # order_dim = len(self.env_core.agents[agent_idx].waybill) + len(self.env_core.agents[agent_idx].wait_to_pick)
            order_dim = self.map.couriers[0].capacity
            speed_dim = self.env_core.num_speeds

            # if order_dim == 0:
            #     action_space = MultiDiscrete([[0, 0], [1, speed_dim]]) # [0, 0] just for the requirement of the form
            # else:
            #     action_space = MultiDiscrete([[0, order_dim - 1], [1, speed_dim]])
            action_space = MultiDiscrete([[0, order_dim - 1], [1, speed_dim]])
            self.action_space.append(action_space)

            # observation space
            # share_obs_dim += 2

            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.signal_obs_dim,), dtype=np.float32))

        # self.share_observation_space = [Box(low=0.0, high=1.0, shape=(share_obs_dim,), dtype=np.float32)]

    def step(self, actions):
        """
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 5-dimensional one_hot encoding
        """
        results = self.env_core.step(actions)
        obs, rews, dones, infos = results
        # obs, rews, dones, infos, share_obs = results


        self.map = self.env_core.map
        self.num_agent = self.map.num_couriers
        self.signal_obs_dim = self.env_core.obs_dim
        self.action_space = self.env_core.action_space

        return np.stack(obs), np.stack(rews), np.stack(dones), infos
        # return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(share_obs)


    def reset(self):
        obs = self.env_core.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

