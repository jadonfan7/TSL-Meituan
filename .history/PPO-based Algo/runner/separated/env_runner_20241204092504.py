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
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        self.eval_num = 0

        distance = []
        episode_rewards = []
        rate_of_overspeed = []
        rate_of_late_order = []
        rate_of_ETA_usage = []
        reject_rate = []
        order_price = []
        
        algo1_distance = []
        algo2_distance = []
        algo1_eval_episode_rewards = []
        algo2_eval_episode_rewards = []
        algo1_rate_of_overspeed = []
        algo2_rate_of_overspeed = []
        algo1_rate_of_late_order = []
        algo2_rate_of_late_order = []
        algo1_rate_of_ETA_usage = []
        algo2_rate_of_ETA_usage = []

        for episode in range(episodes):
            print(f"THE START OF EPISODE {episode+1}")

            courier0_distance_per_episode = 0
            courier1_distance_per_episode = 0

            episode_reward_sum = 0

            count_overspeed0 = 0
            num_active_courier0 = 0
            count_overspeed1 = 0
            num_active_courier1 = 0

            count_reject_orders = 0
            max_reject_num = 0

            late_orders = 0
            ETA_usage = 0
            count_dropped_orders = 0
            
            order0_price = 0
            order1_price = 0


            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            
            obs = self.envs.reset(episode % 10)
            self.num_agents = self.envs.envs_discrete[0].num_couriers

            for step in range(self.episode_length):
                # print("-"*25)
                # print(f"THIS IS STEP {step}")
            
                # dead_count = 0 # end the code

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

                # if dead_count == 5:
                #     break
                
                available_actions = self.envs.get_available_actions()
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step, available_actions)
                # print(actions)

                # Obser reward and next obs
                # obs, rewards, dones, infos, share_obs = self.envs.step(actions_env)
                obs, rewards, dones, infos = self.envs.step(actions_env)

                episode_reward_sum += rewards.sum() / self.envs.num_envs

                for i in range(self.envs.num_envs):
                    for c in self.envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                num_active_courier0 += 1
                                if c.speed > 4:
                                    count_overspeed0 += 1
                            else:
                                num_active_courier1 += 1
                                if c.speed > 4:
                                    count_overspeed1 += 1

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
            
            # Train over periods
            for i in range(self.envs.num_envs):
                courier0_count = 0
                courier1_count = 0
                for c in self.envs.envs_discrete[i].couriers:
                    if c.travel_distance != 0:
                        
                        if c.courier_type == 0:
                            courier0_count += 1
                            courier0_distance_per_episode += c.travel_distance
                        else:
                            courier1_count += 1
                            courier1_distance_per_episode += c.travel_distance
                
                courier0_distance_per_episode /= courier0_count
                courier1_distance_per_episode /= courier1_count

                for o in self.envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        count_dropped_orders += 1
                        if o.is_late == 1:
                            late_orders += 1
                        else:
                            ETA_usage += o.ETA_usage   
                        
                    if o.reject_count > 0:
                        count_reject_orders += 1
                        if max_reject_num <= o.reject_count:
                            max_reject_num = o.reject_count
                    
                    if o.pair_courier == 
                    
                            
                            
            episode_rewards.append(episode_reward_sum)
            print(f"Total Reward for Episode {episode+1}: {episode_reward_sum}")
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
                                    
            courier0_distance_per_episode /= self.envs.num_envs
            courier1_distance_per_episode /= self.envs.num_envs
            distance.append([courier0_distance_per_episode, courier1_distance_per_episode, courier0_distance_per_episode + courier1_distance_per_episode])
            print(f"Average Travel Distance per Courier0: {courier0_distance_per_episode}")
            print(f"Average Travel Distance per Courier1: {courier1_distance_per_episode}")
            print(f"Average Travel Distance per Courier: {courier0_distance_per_episode + courier1_distance_per_episode}")
            self.writter.add_scalar('Total Distance/Courier0', courier0_distance_per_episode, episode + 1)
            self.writter.add_scalar('Total Distance/Courier1', courier1_distance_per_episode, episode + 1)
            self.writter.add_scalar('Total Distance/Courier', courier0_distance_per_episode + courier1_distance_per_episode, episode + 1)
            
            overspeed = (count_overspeed0 + count_overspeed1) / (num_active_courier0 + num_active_courier1)
            overspeed0 = count_overspeed0 / num_active_courier0
            overspeed1 = count_overspeed1 / num_active_courier1
            print(f"Rate of Overspeed for Episode {episode+1}: Total rate - {overspeed}, Courier0 - {overspeed0}, Courier1 - {overspeed1}")
            rate_of_overspeed.append([overspeed, overspeed0, overspeed1])
            self.writter.add_scalar('Overspeed Rate/Total rate', overspeed, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Courier0', overspeed0, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Courier1', overspeed1, episode + 1)
            
            reject_rate_per_episode = count_reject_orders / len(self.envs.envs_discrete[0].orders)
            reject_rate.append(reject_rate_per_episode)
            print(f"The rejection rate is {reject_rate_per_episode}")
            self.writter.add_scalar('Reject rate', reject_rate_per_episode, episode + 1)
            
            

            message = f"\nAverage Travel Distance for Episode {episode+1}: Courier0 - {courier0_distance_per_episode}, Courier1 - {courier1_distance_per_episode}\n" + f"Total Reward for Episode {episode+1}: {episode_reward_sum}\n" + f"Rate of Overspeed for Episode {episode+1}: {overspeed}\n" + f"The rejection rate for Episode {episode+1}: {reject_rate_per_episode}\n"

            if count_dropped_orders == 0:
                print("No order is dropped in this episode")
                rate_of_late_order.append(-1)
                rate_of_ETA_usage.append(-1)
                message += "No order is dropped in this episode\n"
                logger.success(message)
                self.writter.add_scalar('Late Orders Rate', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate', -1, episode + 1)
            else:
                late_rate = late_orders / count_dropped_orders
                print(f"Rate of Late Orders for Episode {episode+1}: {late_rate}")
                rate_of_late_order.append(late_rate)

                ETA_usage_rate = ETA_usage / count_dropped_orders
                print(f"Rate of ETA Usage for Episode {episode+1}: {ETA_usage_rate}")
                rate_of_ETA_usage.append(ETA_usage_rate)
                
                message += f"Rate of Late Orders for Episode {episode+1}: {late_rate}\n" + f"Rate of ETA Usage for Episode {episode+1}: {ETA_usage_rate}\n"
                logger.success(message)
                
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
                info = "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                )
                print(info)
                logger.success(info)

                # self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval_num += 1
                
                (   
                    algo1_eval_episode_rewards_sum,
                    algo2_eval_episode_rewards_sum,
                    algo1_courier_distance_per_episode,
                    algo2_courier_distance_per_episode,
                    algo1_overspeed,
                    algo2_overspeed,
                    algo1_late_rate,
                    algo2_late_rate,
                    algo1_ETA_usage_rate,
                    algo2_ETA_usage_rate,
                ) = self.eval(total_num_steps)

                algo1_distance.append(algo1_courier_distance_per_episode)
                algo2_distance.append(algo2_courier_distance_per_episode)
                algo1_eval_episode_rewards.append(algo1_eval_episode_rewards_sum)
                algo2_eval_episode_rewards.append(algo2_eval_episode_rewards_sum)
                algo1_rate_of_overspeed.append(algo1_overspeed)
                algo2_rate_of_overspeed.append(algo2_overspeed)
                algo1_rate_of_late_order.append(algo1_late_rate)
                algo2_rate_of_late_order.append(algo2_late_rate)
                algo1_rate_of_ETA_usage.append(algo1_ETA_usage_rate)
                algo2_rate_of_ETA_usage.append(algo2_ETA_usage_rate)
        
        self.writter.close()
        

        # draw the Train graph
        courier0_distances = [d[0] for d in distance]
        courier1_distances = [d[1] for d in distance]
        courier_distances = [d[2] for d in distance]
        plt.plot(courier0_distances, label="Courier 0", color='blue')
        plt.plot(courier1_distances, label="Courier 1", color='orange')
        plt.plot(courier_distances, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Total Distances')
        plt.title('Train: Distance over Episodes')
        plt.grid(True)
        plt.savefig('Train_Distance.png')

        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Train: Reward over Episodes')
        plt.grid(True)
        plt.savefig('Train_reward_curve.png')
        
        courier_overspeed = [r[0] for r in rate_of_overspeed]
        courier0_overspeed = [r[1] for r in rate_of_overspeed]
        courier1_overspeed = [r[2] for r in rate_of_overspeed]
        plt.figure(figsize=(10, 6))
        plt.plot(courier0_overspeed, label="Courier 0", color='blue')
        plt.plot(courier1_overspeed, label="Courier 1", color='orange')
        plt.plot(courier_overspeed, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Overspeed')
        plt.title('Train: rate of overspeed over Episodes')
        plt.grid(True)
        plt.savefig('Train_rate_of_overspeed.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(reject_rate)
        plt.xlabel('Episodes')
        plt.ylabel('Order Reject Rate')
        plt.title('Train: order reject rate over Episodes')
        plt.grid(True)
        plt.savefig('Train_order_reject_rate.png')

        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_late_order)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Late Orders')
        plt.title('Train: rate of late orders over Episodes')
        plt.grid(True)
        plt.savefig('Train_rate_of_late_orders.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(rate_of_ETA_usage)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of ETA Usage')
        plt.title('Train: rate of ETA usage over Episodes')
        plt.grid(True)
        plt.savefig('Train_rate_of_ETA_usage.png')
        
        # draw the Evaluation graph
        plt.figure(figsize=(10, 6))
        plt.plot(len(algo1_distance), algo1_distance, label='Algo1 Distance', color='blue', marker='o')
        plt.plot(len(algo2_distance), algo2_distance, label='Algo2 Distance', color='green', marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('Total Distances')
        plt.title('Eval: Distance between two algos')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Distance_Algo1_Algo2.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(len(algo1_eval_episode_rewards), algo1_eval_episode_rewards, label='Algo1 Reward', color='blue', marker='o')
        plt.plot(len(algo2_eval_episode_rewards), algo2_eval_episode_rewards, label='Algo2 Reward', color='green', marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Eval: Reward over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Reward_Algo1_Algo2.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(len(algo1_rate_of_overspeed), algo1_rate_of_overspeed, label='Algo1 Overspeed Rate', color='blue', marker='o')
        plt.plot(len(algo2_rate_of_overspeed), algo2_rate_of_overspeed, label='Algo2 Overspeed Rate', color='green', marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('Overspeed Rate')
        plt.title('Eval: Overspeed Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Overspeed_Rate_Algo1_Algo2.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(len(algo1_rate_of_late_order), algo1_rate_of_late_order, label='Algo1 Late Order Rate', color='blue', marker='o')
        plt.plot(len(algo2_rate_of_late_order), algo2_rate_of_late_order, label='Algo2 Late Order Rate', color='green', marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('Late Order Rate')
        plt.title('Eval: Late Order Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Late_Order_Rate_Algo1_Algo2.png')
        
        plt.figure(figsize=(10, 6))
        plt.plot(len(algo1_rate_of_ETA_usage), algo1_rate_of_ETA_usage, label='Algo1 ETA Usage Rate', color='blue', marker='o')
        plt.plot(len(algo2_rate_of_ETA_usage), algo2_rate_of_ETA_usage, label='Algo2 ETA Usage Rate', color='green', marker='x')
        plt.xlabel('Episodes')
        plt.ylabel('ETA Usage Rate')
        plt.title('Eval: ETA Usage Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_ETA_Usage_Rate_Algo1_Algo2.png')

    @torch.no_grad()
    def collect(self, step, available_actions):
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
                torch.tensor(available_actions[agent_id]),
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
        
        eval_obs = self.eval_envs.reset(10)
        
        algo1_eval_episode_rewards_sum = 0
        algo2_eval_episode_rewards_sum = 0        
        
        algo1_courier_distance_per_episode = 0
        algo2_courier_distance_per_episode = 0

        algo1_count_overspeed = 0
        algo2_count_overspeed = 0
        
        algo1_num_active_couriers = 0
        algo2_num_active_couriers = 0

        algo1_late_orders = 0
        algo2_late_orders = 0
        
        algo1_ETA_usage = 0
        algo2_ETA_usage = 0
        
        algo1_count_dropped_orders = 0
        algo2_count_dropped_orders = 0
        
        self.eval_num_agents = self.eval_envs.envs_discrete[0].num_couriers

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.eval_num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.eval_num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            if self.eval_num_agents > self.num_agents:
                break
            
            print("-"*25)
            print(f"THIS IS STEP {eval_step}")

            for i in range(self.eval_envs.num_envs):
                
                print(f"ENVIRONMENT {i+1}")

                print("Couriers:")
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.state == 'active':
                        print(c)
                print("Orders:")
                for o in self.eval_envs.envs_discrete[i].orders:
                    print(o)  
                print("\n")
                
                self.log_env(1, eval_step, i)
                
            eval_temp_actions_env = []
            
            for agent_id in range(self.eval_num_agents):
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
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[eval_action[:, i]]
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
            
            algo1_eval_episode_rewards_sum += sum(eval_rewards[0])
            algo2_eval_episode_rewards_sum += sum(eval_rewards[1])

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.eval_num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            for i in range(self.eval_envs.num_envs):
                if i == 0:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            algo1_num_active_couriers += 1
                            if c.speed > 4:
                                algo1_count_overspeed += 1
                elif i == 1:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            algo2_num_active_couriers += 1
                            if c.speed > 4:
                                algo2_count_overspeed += 1
                                
            eval_obs = self.eval_envs.eval_env_step()
            
            add_courier_num = self.eval_envs.envs_discrete[0].num_couriers - self.eval_num_agents

            new_eval_rnn_states = np.zeros(
                (
                    self.n_eval_rollout_threads,
                    add_courier_num,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            eval_rnn_states = np.concatenate((eval_rnn_states, new_eval_rnn_states), axis=1)
            new_eval_masks = np.ones((self.n_eval_rollout_threads, add_courier_num, 1), dtype=np.float32)
            eval_masks = np.concatenate((eval_masks, new_eval_masks), axis=1)
                            
            self.eval_num_agents = self.eval_envs.envs_discrete[0].num_couriers

        # Evaluation over periods
        for i in range(self.eval_envs.num_envs):
            courier_count = 0
            if i == 0:
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.travel_distance != 0:
                        courier_count += 1
                        algo1_courier_distance_per_episode += c.travel_distance
                algo1_courier_distance_per_episode /= courier_count
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        algo1_count_dropped_orders += 1
                        if o.is_late == 1:
                            algo1_late_orders += 1
                        else:
                            algo1_ETA_usage += o.ETA_usage 
            elif i == 1:
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.travel_distance != 0:
                        courier_count += 1
                        algo2_courier_distance_per_episode += c.travel_distance
                algo2_courier_distance_per_episode /= courier_count

                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        algo2_count_dropped_orders += 1
                        if o.is_late == 1:
                            algo2_late_orders += 1
                        else:
                            algo2_ETA_usage += o.ETA_usage 
        
        print(f"\nTotal Reward for Evaluation Between Two Algos: Algo1: {algo1_eval_episode_rewards_sum}, Algo2: {algo2_eval_episode_rewards_sum}")
        self.writter.add_scalar('Eval Reward/Algo1', algo1_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo2', algo2_eval_episode_rewards_sum, self.eval_num)
        
        print(f"Average Travel Distance per Courier Between Two Algos: Algo1: {algo1_courier_distance_per_episode}, Algo2: {algo2_courier_distance_per_episode}")
        self.writter.add_scalar('Eval Travel Distance/Algo1', algo1_courier_distance_per_episode, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2', algo2_courier_distance_per_episode, self.eval_num)
        
        algo1_overspeed = algo1_count_overspeed / algo1_num_active_couriers
        algo2_overspeed = algo2_count_overspeed / algo2_num_active_couriers
        print(f"Rate of Overspeed for Evaluation Between Two Algos: Algo1: {algo1_overspeed}, Algo2: {algo2_overspeed}")
        self.writter.add_scalar('Eval Overspeed Rate/Algo1', algo1_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2', algo2_overspeed, self.eval_num)

        message = f"Total Reward for Evaluation Between Two Algos: Algo1: {algo1_eval_episode_rewards_sum}, Algo2: {algo2_eval_episode_rewards_sum}\n" + f"Average Travel Distance per Courier Between Two Algos: Algo1: {algo1_courier_distance_per_episode}, Algo2: {algo2_courier_distance_per_episode}\n" + f"Rate of Overspeed for Evaluation Between Two Algos: Algo1: {algo1_overspeed}, Algo2: {algo2_overspeed}\n"
        
        if algo1_count_dropped_orders == 0:
            print("No order is dropped in Algo1")
            algo1_late_rate = -1
            algo1_ETA_usage_rate = -1
            
            self.writter.add_scalar('Eval Late Order Rate/Algo1', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1', algo1_ETA_usage_rate, self.eval_num)
            
            message += "No order is dropped in Algo1\n"
        else:
            algo1_late_rate = algo1_late_orders / algo1_count_dropped_orders
            print(f"Rate of Late Orders for Evaluation in Algo1: {algo1_late_rate}")

            algo1_ETA_usage_rate = algo1_ETA_usage / algo1_count_dropped_orders
            print(f"Rate of ETA Usage for Evaluation in Algo1: {algo1_ETA_usage_rate}")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo1', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1', algo1_ETA_usage_rate, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo1: {algo1_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo1: {algo1_ETA_usage_rate}\n"

        
        if algo2_count_dropped_orders == 0:
            print("No order is dropped in Algo2")
            algo2_late_rate = -1
            algo2_ETA_usage_rate = -1
            
            self.writter.add_scalar('Eval Late Order Rate/Algo2', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2', algo2_ETA_usage_rate, self.eval_num)

            message += "No order is dropped in Algo2\n"
        else:
            algo2_late_rate = algo1_late_orders / algo2_count_dropped_orders
            print(f"Rate of Late Orders for Evaluation in Algo2: {algo2_late_rate}")

            algo2_ETA_usage_rate = algo1_ETA_usage / algo2_count_dropped_orders
            print(f"Rate of ETA Usage for Evaluation in Algo2: {algo2_ETA_usage_rate}")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo2', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2', algo2_ETA_usage_rate, self.eval_num)
           
            message += f"Rate of Late Orders for Evaluation in Algo2: {algo2_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo2: {algo2_ETA_usage_rate}\n"
        
        logger.success(message)
            
        print("\n")
        
        return (
            algo1_eval_episode_rewards_sum,
            algo2_eval_episode_rewards_sum,
            algo1_courier_distance_per_episode,
            algo2_courier_distance_per_episode,
            algo1_overspeed,
            algo2_overspeed,
            algo1_late_rate,
            algo2_late_rate,
            algo1_ETA_usage_rate,
            algo2_ETA_usage_rate
        )
        

        # self.log_train(eval_train_infos, total_num_steps)
    
    # def game_success(self, step, map_env):
    #     flag = True
    #     if step <= 10:
    #         flag = False
    #     else:
    #         for order in map_env.orders:
    #             if order.status != 'dropped':
    #                 flag = False
        
    #     return flag
