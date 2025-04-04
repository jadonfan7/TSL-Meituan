from multiprocessing import Process, Pipe
import numpy as np
from abc import ABC, abstractmethod

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    def __init__(self, num_envs, observation_space, action_space, share_observation_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.share_observation_space = share_observation_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

def worker(remote, parent_remote, env_fn_wrapper):

    parent_remote.close()
    env = env_fn_wrapper.x()
    remote.send(env)
    while True:
        cmd, data = remote.recv()
        
        if cmd == 'step':
            ob, reward, done, share_obs = env.step(data)
            env_map = env.get_map()
            
            remote.send((ob, reward, done, share_obs, env_map))
            
        elif cmd == 'reset':
            index, eval = data
            env.reset(index, eval)
            env_map = env.get_map()
            obs = env.get_env_obs()
            action_space, observation_space = env.get_env_space()
            remote.send((env_map, obs, action_space, observation_space))

        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space, env.share_observation_space))            
        else:
            raise NotImplementedError
            
class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        # self.envs_map = [fn() for fn in env_fns]
        nenvs = len(env_fns)
        
        self.num_envs = nenvs
        self.envs_map = []
        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
        for remote in self.remotes:
            env = remote.recv()
            self.envs_map.append(env.map)

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space, share_observation_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, action_space, share_observation_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, share_obs, self.envs_map = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(share_obs)

    def reset(self, index, eval=False):
        for remote in self.remotes:
            remote.send(('reset', (index, eval)))
        
        results = [remote.recv() for remote in self.remotes]
        self.envs_map = [result[0] for result in results]
        obs = [result[1] for result in results]
        
        self.action_space, self.observation_space = results[0][2], results[0][3]
        return np.array(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True