from multiprocessing import Process, Pipe, Manager
import numpy as np
import torch
import socket
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

    def __init__(self, num_envs):
        self.num_envs = num_envs

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

    def step(self):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async()
        return self.step_wait()

def worker(remote, parent_remote, env_fn_wrapper):

    parent_remote.close()
    env = env_fn_wrapper.x()
    remote.send(env)
    while True:
        cmd, data = remote.recv()
        
        if cmd == 'step':
            env.step()
            env_map = env.get_map()
            remote.send((env_map))
                        
        elif cmd == 'reset':
            env.reset(data)
            env_map = env.get_map()
            remote.send((env_map))

        elif cmd == 'close':
            remote.close()
            break
        
        elif cmd == 'eval_env_step':
            env.map.eval_step()
            env_map = env.get_map()
            
            remote.send((env_map))
            
        else:
            raise NotImplementedError
            
class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
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

        ShareVecEnv.__init__(self, len(env_fns))

    def step_async(self):
        for remote in self.remotes:
            remote.send(('step', None))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        self.envs_map = [result for result in results]

    def reset(self, index):
        for remote in self.remotes:
            remote.send(('reset', index))
        
        results = [remote.recv() for remote in self.remotes]
        self.envs_map = [result for result in results]
        
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
    
    def eval_env_step(self):
        for remote in self.remotes:
            remote.send(('eval_env_step', None))
            
        results = [remote.recv() for remote in self.remotes]
    
        self.envs_map = [result for result in results]