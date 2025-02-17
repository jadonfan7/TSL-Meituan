import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from envs.env_wrappers import SubprocVecEnv

def make_eval_env(n_eval_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            
            from envs.env_core import EnvCore
            env = EnvCore(rank)

            return env

        return init_env

    return SubprocVecEnv([get_env_fn(i) for i in range(n_eval_rollout_threads)])


class Runner(object):
    def __init__(self):
        self.eval_episodes_length = 450
        self.num_algo = 1
        self.eval_envs = make_eval_env(self.num_algo)
        
        logger.remove()
        logger.add('Simulation_logs/env_step_log.log', rotation="50 MB", level="DEBUG")
        logger.add("Simulation_logs/step_procedure.log", rotation="50 MB", level="INFO")
        logger.add("Simulation_logs/env_episode_log.log", rotation="500 MB", level="SUCCESS")

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def log_env(self, step, env_index):
        basic_info = "-" * 25 + "\n"
        basic_info += f"THIS IS STEP {step+1}\n"
        basic_info += f"ENVIRONMENT {env_index+1}\n"
        logger.info(basic_info)
        
        step_info = "Couriers:\n"

        count_overspeed = 0
        num_active_couriers = 0
        courier_count = 0
        dist = 0
        
        for c in self.eval_envs.envs_map[env_index].couriers:
            dist += c.travel_distance
            if c.state == 'active':
                step_info += f"{c}\n"
                num_active_couriers += 1
                if c.speed > 4:
                    count_overspeed += 1
            if c.state == 'active' or c.travel_distance > 0:
                courier_count += 1
        
        step_info += "Orders:\n"
        for o in self.eval_envs.envs_map[env_index].orders:
            step_info += f"{o}\n"
            
        step_info += f"The average travel distance per courier is {round(dist / courier_count, 2)} meters\n"
        step_info += f"The rate of overspeed {round(count_overspeed / num_active_couriers, 2)}\n"

        logger.debug(step_info)