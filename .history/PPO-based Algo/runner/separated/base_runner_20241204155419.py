import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from utils.separated_buffer import SeparatedReplayBuffer

from algorithms.algorithm.mappo import RMAPPO as TrainAlgo
from algorithms.algorithm.MAPPO_Policy import R_MAPPOPolicy as Policy


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        
        # if self.use_wandb:
        #     self.save_dir = str(wandb.run.dir)
        # else:
        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        logger.remove()
        logger.add("ippo_logs/env_step_log.log", rotation="500 MB", level="INFO")
        logger.add("ippo_logs/env_episode_log.log", rotation="500 MB", level="SUCCESS")

        self.policy = []
        for agent_id in range(self.num_agents):
            # share_observation_space = (
            #     # self.envs.share_observation_space[agent_id]
            #     self.envs.share_observation_space # delete [agent_id]
            #     if self.use_centralized_V
            #     else self.envs.observation_space[agent_id]
            # )
            # policy network

            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                # share_observation_space[0], # add [0]
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            # share_observation_space = (
            #     self.envs.share_observation_space # delete [agent_id]
            #     if self.use_centralized_V
            #     else self.envs.observation_space[agent_id]
            # )
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                # share_observation_space[0], # add [0]
                self.envs.action_space[agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_agent" + str(agent_id) + ".pt")
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + "/critic_agent" + str(agent_id) + ".pt"
            )
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    # def log_train(self, train_infos, total_num_steps):
    #     for agent_id in range(self.num_agents):
    #         for k, v in train_infos[agent_id].items():
    #             agent_k = "agent%i/" % agent_id + k
    #             # if self.use_wandb:
    #             #     pass
    #             # wandb.log({agent_k: v}, step=total_num_steps)
    #             # else:
    #             self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, episode, step, env_index, eval=False):
        step_info = "-" * 25 + "\n"
        step_info += f"THIS IS EPISODE {episode+1}, STEP {step+1}\n"
        step_info += f"ENVIRONMENT {env_index+1}\n"
        step_info += "Couriers:\n"

        count_overspeed = 0
        num_active_couriers = 0
        dist = 0
        if eval:
            
        for c in self.envs.envs_discrete[env_index].couriers:
            dist += c.travel_distance
            if c.state == 'active':
                step_info += f"{c}\n"
                num_active_couriers += 1
                if c.speed > 4:
                    count_overspeed += 1
        
        step_info += "Orders:\n"
        for o in self.envs.envs_discrete[env_index].orders:
            step_info += f"{o}\n"

        step_info += f"The average travel distance per courier is {round(dist / len(self.envs.envs_discrete[env_index].couriers, 2))} meters\n"
        step_info += f"The rate of overspeed {round(count_overspeed / num_active_couriers, 2)}\n"

        logger.info(step_info)
        
    def add_new_agent(self, num):
        for agent_id in range(num):
            po = Policy(
                self.all_args,
                self.envs.observation_space[self.num_agents + agent_id],
                # share_observation_space[0], # add [0]
                self.envs.action_space[self.num_agents + agent_id],
                device=self.device,
            )
            self.policy.append(po)
            
        for agent_id in range(num):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[self.num_agents + agent_id],
                # share_observation_space[0], # add [0]
                self.envs.action_space[self.num_agents + agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)