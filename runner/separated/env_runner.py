import time
import numpy as np
import torch

from runner.separated.base_runner import Runner
import matplotlib.pyplot as plt

from loguru import logger



def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):

        start = time.time()
        # episodes = int(self.num_env_steps) // self.episode_length
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        distance = []
        episode_rewards = []
        rate_of_overspeed = []
        rate_of_late_order = []
        rate_of_ETA_usage = []

        for episode in range(episodes):
            print(f"THE START OF EPISODE {episode+1}")

            courier_distance_per_episode = 0

            episode_reward_sum = 0

            count_overspeed = 0
            num_active_couriers = 0

            late_orders = 0
            ETA_usage = 0
            count_dropped_orders = 0


            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
 
            
            self.envs.reset()

            for step in range(self.episode_length):
                # print("-"*25)
                # print(f"THIS IS STEP {step}")
            
                dead_count = 0 # end the code

                for i in range(self.envs.num_envs):
                    # print(f"ENVIRONMENT {i+1}")

                    # print("Couriers:")
                    # for c in self.envs.envs_discrete[i].couriers:
                    #     if c.state == 'active':
                    #         print(c)
                    # print("Orders:")
                    # for o in self.envs.envs_discrete[i].orders:
                    #     print(o)  
                    # print("\n")
                    self.log_env(episode, step, i)

                    # if self.game_success(step, self.envs.envs_discrete[i].map):
                    #     dead_count += 1
                    #     continue

                if dead_count == 5:
                    break
                
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                # print(actions)

                # Obser reward and next obs
                # obs, rewards, dones, infos, share_obs = self.envs.step(actions_env)
                obs, rewards, dones, infos = self.envs.step(actions_env)

                episode_reward_sum += rewards.sum() / self.envs.num_envs

                for i in range(self.envs.num_envs):
                    for c in self.envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            num_active_couriers += 1
                            if c.speed > 4:
                                count_overspeed += 1

                data = (
                    obs,
                    # share_obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                # print(rewards)

                # insert data into buffer
                self.insert(data)


                self.envs.env_step()
                add_courier_num = self.envs.envs_discrete[0].num_couriers - self.num_agents
                self.add_new_agent(add_courier_num)
                                
                self.num_agents = self.envs.envs_discrete[0].num_couriers
            
            # Evaluation over periods
            for i in range(self.envs.num_envs):
                courier_count = 0
                for c in self.envs.envs_discrete[i].couriers:
                    if c.state == 'active':
                        courier_count += 1
                        courier_distance_per_episode += c.travel_distance
                courier_distance_per_episode /= courier_count

                for o in self.envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        count_dropped_orders += 1
                        if o.is_late == 1:
                            late_orders += 1
                        else:
                            ETA_usage += o.ETA_usage   
                            
            episode_rewards.append(episode_reward_sum)
            print(f"Total Reward for Episode {episode+1}: {episode_reward_sum}")
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
 
                                                     
            courier_distance_per_episode /= self.envs.num_envs
            distance.append(courier_distance_per_episode)
            print(f"Average Travel Distance per Courier: {courier_distance_per_episode}")
            self.writter.add_scalar('Total Distance', courier_distance_per_episode, episode + 1)

            overspeed = count_overspeed / num_active_couriers
            print(f"Rate of Overspeed for Episode {episode+1}: {overspeed}")
            rate_of_overspeed.append(overspeed)
            self.writter.add_scalar('Overspeed Rate', overspeed, episode + 1)


            message = f"Average Courier's Travel Distance for Episode {episode+1}: {courier_distance_per_episode}\n" + f"Total Reward for Episode {episode+1}: {episode_reward_sum}\n" + f"Rate of Overspeed for Episode {episode+1}: {overspeed}\n"
            logger.info(message)

            if count_dropped_orders == 0:
                print("No order is dropped in this episode")
                rate_of_late_order.append(-1)
                rate_of_ETA_usage.append(-1)
                logger.info("No order is dropped in this episode\n")
                self.writter.add_scalar('Late Orders Rate', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate', -1, episode + 1)
            else:
                late_rate = late_orders / count_dropped_orders
                print(f"Rate of Late Orders for Episode {episode+1}: {late_rate}")
                rate_of_late_order.append(late_rate)
                logger.info(f"Rate of Late Orders for Episode {episode+1}: {late_rate}\n")

                ETA_usage_rate = ETA_usage / count_dropped_orders
                print(f"Rate of ETA Usage for Episode {episode+1}: {ETA_usage_rate}")
                rate_of_ETA_usage.append(ETA_usage_rate)
                logger.info(f"Rate of ETA Usage for Episode {episode+1}: {ETA_usage_rate}\n")
                
                self.writter.add_scalar('Late Orders Rate', late_rate, episode + 1)
                self.writter.add_scalar('ETA Usage Rate', ETA_usage_rate, episode + 1)

            print("\n")

            

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        
        self.writter.close()
        

        # draw the evaluation graph
        plt.figure(figsize=(10, 6))
        plt.plot(distance)
        plt.xlabel('Episodes')
        plt.ylabel('Total Distances')
        plt.title('Distance over Episodes')
        plt.grid(True)
        plt.savefig('Distance.png')

        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Reward over Episodes')
        plt.grid(True)
        plt.savefig('reward_curve.png')

        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_overspeed)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Overspeed')
        plt.title('rate of overspeed over Episodes')
        plt.grid(True)
        plt.savefig('rate_of_overspeed.png')

        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_late_order)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Late Orders')
        plt.title('rate of late orders over Episodes')
        plt.grid(True)
        plt.savefig('rate_of_late_orders.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_ETA_usage)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of ETA Usage')
        plt.title('rate of ETA usage over Episodes')
        plt.grid(True)
        plt.savefig('rate_of_ETA_usage.png')

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                # self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            # share_obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        for agent_id in range(self.num_agents):
            # if not self.use_centralized_V:
            #     share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                # share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id].reshape(-1,1),
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)
    
    # def game_success(self, step, map_env):
    #     flag = True
    #     if step <= 10:
    #         flag = False
    #     else:
    #         for order in map_env.orders:
    #             if order.status != 'dropped':
    #                 flag = False
        
    #     return flag
