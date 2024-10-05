import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs_discrete = [fn() for fn in env_fns]
        env = self.envs_discrete[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        # self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs_discrete)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        # obs, rews, dones, infos, share_obs = map(np.array, zip(*results))

        # for (i, done) in enumerate(dones):
        #     if 'bool' in done.__class__.__name__:
        #         if done:
        #             obs[i] = self.envs[i].reset()
        #     else:
        #         if np.all(done):
        #             obs[i] = self.envs[i].reset()

        self.actions = None
        # return obs, rews, dones, infos, share_obs
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs_discrete] # [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs_discrete:
            env.close()
