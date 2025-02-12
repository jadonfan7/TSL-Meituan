import sys
import os
import socket
# import setproctitle
import numpy as np
from pathlib import Path
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import SubprocVecEnv

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            
            from envs.env_core import EnvCore
            env = EnvCore(rank)

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def main(args):
    
    eval_envs = make_eval_env(all_args)

    from runner.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()
        
    eval_envs.close()

    runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])
