import numpy as np
from envs.env_core import EnvCore
from gym.spaces import Discrete
from envs.observationspace import ObservationSpace
from gym.spaces import Box

class DiscreteActionEnv(object):
    """
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self):
        self.env_core = EnvCore()
        self.num_agent = self.env_core.agent_num
        self.gridmap = self.env_core.gridmap

        self.signal_obs_dim = self.env_core.obs_dim
        self.signal_action_dim = self.env_core.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = self.env_core.gridmap.num_orders * 4
        total_action_space = []
        for agent_idx in range(self.num_agent):
            # physical action space
            u_action_space = Discrete(self.signal_action_dim)

            total_action_space.append(u_action_space)

            self.action_space.append(total_action_space[agent_idx])

            # observation space
            # share_obs_dim += self.signal_obs_dim
            share_obs_dim += 2

            # self.observation_space.append(
            #     ObservationSpace(gridmap, gridmap.couriers[agent_idx]).get_obs()
            # )
            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.signal_obs_dim,), dtype=np.float32))

        self.share_observation_space = [Box(low=0.0, high=1.0, shape=(share_obs_dim,), dtype=np.float32)]
        # self.share_observation_space = [
        #     ObservationSpace(gridmap, share=True).get_obs()
        # ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of the environment, with 2 intelligent agents inside, and each intelligent agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env_core.step(actions)
        obs, rews, dones, infos, share_obs= results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos, np.stack(share_obs)

    def reset(self):
        obs = self.env_core.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass


# class MultiDiscrete:
#     """
#     - The multi-discrete action space consists of a series of discrete action spaces with different parameters
#     - It can be adapted to both a Discrete action space or a continuous (Box) action space
#     - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
#     - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
#        where the discrete action space can take any integers from `min` to `max` (both inclusive)
#     Note: A value of 0 always need to represent the NOOP action.
#     e.g. Nintendo Game Controller
#     - Can be conceptualized as 3 discrete action spaces:
#         1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
#         2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
#         3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
#     - Can be initialized as
#         MultiDiscrete([ [0,4], [0,1], [0,1] ])
#     """

#     def __init__(self, array_of_param_array):
#         super().__init__()
#         self.low = np.array([x[0] for x in array_of_param_array])
#         self.high = np.array([x[1] for x in array_of_param_array])
#         self.num_discrete_space = self.low.shape[0]
#         self.n = np.sum(self.high) + 2

#     def sample(self):
#         """Returns a array with one sample from each discrete action space"""
#         # For each row: round(random .* (max - min) + min, 0)
#         random_array = np.random.rand(self.num_discrete_space)
#         return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

#     def contains(self, x):
#         return (
#             len(x) == self.num_discrete_space
#             and (np.array(x) >= self.low).all()
#             and (np.array(x) <= self.high).all()
#         )

#     @property
#     def shape(self):
#         return self.num_discrete_space

#     def __repr__(self):
#         return "MultiDiscrete" + str(self.num_discrete_space)

#     def __eq__(self, other):
#         return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    from gridmap import GridMap
    from agent.courier import Courier
    from agent.order import Order
    gm = GridMap((10,10), [Courier('1', (0,0))], [Order('001', (3,4), (5,5))])
    print(DiscreteActionEnv(gm).step(actions=None))
