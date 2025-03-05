import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from utils.shared_buffer import SharedReplayBuffer
from utils.separated_buffer import SeparatedReplayBuffer
from algorithms.algorithm.mappo import RMAPPO as TrainAlgo
from algorithms.algorithm.MAPPO_Policy import R_MAPPOPolicy as Policy

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):
    
        self.all_args = config["all_args"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.eval_episodes_length = self.all_args.eval_episodes_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        
        # self.run_dir = config["run_dir"]
        # self.log_dir = str(self.run_dir / "logs")
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        # self.writter = SummaryWriter(self.log_dir)
        # self.save_dir = str(self.run_dir / "models")
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)
        
        share_observation_space = self.eval_envs.share_observation_space if self.use_centralized_V else self.eval_envs.observation_space[0]
        
        self.policy1 = Policy(self.all_args, self.eval_envs.observation_space[0], share_observation_space, self.eval_envs.action_space[0], device=self.device)
        self.policy2 = Policy(self.all_args, self.eval_envs.observation_space[0], share_observation_space, self.eval_envs.action_space[0], device=self.device)
        
        if self.model_dir is not None:
            self.restore()
        
        self.trainer1 = TrainAlgo(self.all_args, self.policy1, device = self.device)
        self.trainer2 = TrainAlgo(self.all_args, self.policy2, device = self.device)

        logger.remove()
        logger.add("Logs/env_step_log.log", rotation="50 MB", level="INFO")
        logger.add("Logs/env_episode_log.log", rotation="500 MB", level="SUCCESS")
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor1.pt", map_location=torch.device('cpu'))
        self.policy1.actor.load_state_dict(policy_actor_state_dict)
        # policy_critic_state_dict = torch.load(
        #     str(self.model_dir) + "/critic1.pt", map_location=torch.device('cpu')
        # )
        # self.policy1.critic.load_state_dict(policy_critic_state_dict)

        policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor2.pt", map_location=torch.device('cpu'))
        self.policy2.actor.load_state_dict(policy_actor_state_dict)
        # policy_critic_state_dict = torch.load(
        #     str(self.model_dir) + "/critic2.pt", map_location=torch.device('cpu')
        # )
        # self.policy2.critic.load_state_dict(policy_critic_state_dict)

    def log_env(self, episode, step, env_index, eval=False):
        step_info = "-" * 25 + "\n"
        if eval:
            step_info += f"THIS IS EVAL EPISODE {episode+1}, STEP {step+1}\n"
        else:
            step_info += f"THIS IS EPISODE {episode+1}, STEP {step+1}\n"
        step_info += f"ENVIRONMENT {env_index+1}\n"
        step_info += "Couriers:\n"

        count_overspeed = 0
        num_active_couriers = 0
        courier_count = 0
        dist = 0
        
        if eval:
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

        else:
            for c in self.envs.envs_map[env_index].couriers:
                dist += c.travel_distance
                if c.state == 'active':
                    step_info += f"{c}\n"
                    num_active_couriers += 1
                    if c.speed > 4:
                        count_overspeed += 1
                if c.state == 'active' or c.travel_distance > 0:
                    courier_count += 1
            
            step_info += "Orders:\n"
            for o in self.envs.envs_map[env_index].orders:
                step_info += f"{o}\n"

            step_info += f"The average travel distance per courier is {round(dist / courier_count, 2)} meters\n"
            step_info += f"The rate of overspeed {round(count_overspeed / num_active_couriers, 2)}\n"

        logger.info(step_info)
        
    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(2):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)