import time
import numpy as np
import torch

from runner.base_runner import Runner

from loguru import logger
import numpy as np

def _t2n(x):
    return x.detach().cpu().numpy()

class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        self.eval_num = 0
        
        for episode in range(episodes):
            print(f"THE START OF EPISODE {episode+1}")
                        
            courier_num = 0
            order_num = 0
            
            Hired_distance_per_episode = []
            Crowdsourced_distance_per_episode = []
            Hired_num = 0
            Crowdsourced_num = 0
            Crowdsourced_on = 0

            episode_reward_sum = 0
            
            overspeed0_step = []
            overspeed1_step = []
            
            count_reject_orders = 0
            max_reject_num = 0
            
            count_dropped_orders = 0
            count_unfinished_orders = 0
            late_orders = 0
            unfinished_late_orders = 0
            order_wait_pair = 0
            ETA_usage = []
            
            order_waiting_time = []

            # order_price = []
                     
            Hired_reject_num = []
            Crowdsourced_reject_num = []
            
            Hired_finished_num = []
            Crowdsourced_finished_num = []
            
            Hired_unfinished_num = []
            Crowdsourced_unfinished_num = []
            
            Hired_leisure_time = []
            Crowdsourced_leisure_time = []
            
            Hired_running_time = []
            Crowdsourced_running_time = []

            Hired_waiting_time = []
            Crowdsourced_waiting_time = []
            
            Hired_actual_speed = []
            Crowdsourced_actual_speed = []
            
            if self.use_linear_lr_decay:
                self.trainer1.policy.lr_decay(episode, episodes)
                self.trainer2.policy.lr_decay(episode, episodes)
            
            obs = self.envs.reset(episode % 4)
            self.num_agents = self.envs.envs_map[0].num_couriers

            for step in range(self.episode_length):
                print(f"THIS IS STEP {step}")

                for i in range(self.envs.num_envs):
                    self.log_env(episode, step, i)
                    
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)
                
                # Obser reward and next obs
                obs, rewards, dones, share_obs = self.envs.step(actions_env)

                episode_reward_sum += rewards.sum() / self.envs.num_envs

                num0 = 0
                num1 = 0
                count0 = 0
                count1 = 0
                for c in self.envs.envs_map[0].active_couriers:
                    if c.state == 'active':
                        if c.courier_type == 0:
                            num0 += 1
                            if c.speed > 4:
                                count0 += 1
                        else:
                            num1 += 1
                            if c.speed > 4:
                                count1 += 1
                overspeed0_step.append(count0 / num0)
                overspeed1_step.append(count1 / num1)

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                
                # insert data into buffer
                self.insert(data)
                                    
            # Train over periods            
            for c in self.envs.envs_map[0].couriers:
                if c.courier_type == 0:
                    Hired_num += 1
                    if c.travel_distance > 0:
                        Hired_distance_per_episode.append(c.travel_distance)
                        Hired_finished_num.append(c.finish_order_num)
                        Hired_unfinished_num.append(len(c.waybill)+len(c.wait_to_pick))
                        Hired_reject_num.append(c.reject_order_num)
                        Hired_leisure_time.append(c.total_leisure_time)
                        Hired_running_time.append(c.total_running_time)
                        Hired_waiting_time.append(c.total_waiting_time)
                    if c.actual_speed > 0:
                        Hired_actual_speed.append(c.actual_speed)
                else:
                    Crowdsourced_num += 1
                    if c.travel_distance > 0:
                        Crowdsourced_distance_per_episode.append(c.travel_distance)
                        Crowdsourced_finished_num.append(c.finish_order_num)
                        Crowdsourced_unfinished_num.append(len(c.waybill)+len(c.wait_to_pick))
                        Crowdsourced_reject_num.append(c.reject_order_num)
                        Crowdsourced_leisure_time.append(c.total_leisure_time)
                        Crowdsourced_running_time.append(c.total_running_time)
                        Crowdsourced_waiting_time.append(c.total_waiting_time)
                    if c.actual_speed > 0:
                        Crowdsourced_actual_speed.append(c.actual_speed)
                    if c.state == 'active':
                        Crowdsourced_on += 1
            
            courier_num = len(self.envs.envs_map[0].couriers)
                                
            for o in self.envs.envs_map[0].orders:
                if o.status in {'wait_pair', 'wait_pick', 'picked_up'}:
                    count_unfinished_orders += 1
                    if o.ETA <= self.envs.envs_map[0].clock:
                        unfinished_late_orders += 1
                        
                    if o.status == 'wait_pair':
                        order_wait_pair += 1
                    elif o.status == 'picked_up':
                        order_waiting_time.append(o.wait_time)
                        
                else:
                    count_dropped_orders += 1
                    if o.is_late:
                        late_orders += 1
                    else:
                        ETA_usage.append(o.ETA_usage)
                    order_waiting_time.append(o.wait_time)
                
                if o.reject_count > 0:
                    count_reject_orders += 1
                    if max_reject_num <= o.reject_count:
                        max_reject_num = o.reject_count
                                        
            order_num = len(self.envs.envs_map[0].orders)
                            
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
            
            # ---------------------
            # distance
            distance0 = np.mean(Hired_distance_per_episode) / 1000
            distance_var0 = np.var(Hired_distance_per_episode) / 1000000
            distance1 = np.mean(Crowdsourced_distance_per_episode) / 1000
            distance_var1 = np.var(Crowdsourced_distance_per_episode) / 1000000
            distance = np.mean(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000
            distance_var = np.var(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000000
            self.writter.add_scalar('Train/Total Distance/Total', distance, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Total_Var', distance_var, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Hired', distance0, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Hired_Var', distance_var0, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Crowdsourced', distance1, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Crowdsourced_Var', distance_var1, episode + 1)
            
            message = (
                f"\nThis is Train Episode {episode+1}\n"
                
                f"There are {courier_num} couriers ({Hired_num} Hired, {Crowdsourced_num} Crowdsourced with {Crowdsourced_on} ({round(100 * Crowdsourced_on / Crowdsourced_num, 2)}%) on), and {order_num} Orders ({count_dropped_orders} dropped, {count_unfinished_orders} unfinished), {order_wait_pair} ({round(100 * order_wait_pair / order_num, 2)}%) waiting to be paired\n"
                
                f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}\n"
                 
                f"Average Travel Distance for Episode {episode+1}: Hired ({len(Hired_distance_per_episode)}) - {distance0} km (Var: {distance_var0}), Crowdsourced ({len(Crowdsourced_distance_per_episode)}) - {distance1} km (Var: {distance_var1}), Total ({len(Hired_distance_per_episode+Crowdsourced_distance_per_episode)}) - {distance} km (Var: {distance_var})\n"
            )

            if count_dropped_orders == 0:
                print("No order is dropped in this episode")
                message += "No order is dropped in this episode\n"
                self.writter.add_scalar('Train/Late Orders Rate', -1, episode + 1)
                self.writter.add_scalar('Train/ETA Usage Rate', -1, episode + 1)
                self.writter.add_scalar('Train/ETA Usage Rate Var', 0, episode + 1)
            else:
                late_rate = late_orders / count_dropped_orders
                ETA_usage_rate = np.mean(ETA_usage)
                Var_ETA = np.var(ETA_usage)
                print(f"Rate of Late Orders for Episode {episode+1}: {late_rate} out of {count_dropped_orders} orders")
                print(f"Rate of ETA Usage for Episode {episode+1}: {ETA_usage_rate} (Var: {Var_ETA})")
                message += f"Rate of Late Orders for Episode {episode+1}: Total - {late_rate} out of {count_dropped_orders} orders\n" + f"Rate of ETA Usage for Episode {episode+1}: Total - {ETA_usage_rate} (Var: {Var_ETA})\n"
                self.writter.add_scalar('Train/Late Orders Rate', late_rate, episode + 1)
                self.writter.add_scalar('Train/ETA Usage Rate', ETA_usage_rate, episode + 1)
                self.writter.add_scalar('Train/ETA Usage Rate Var', Var_ETA, episode + 1)
                
            if count_unfinished_orders == 0:
                print("No order is unfinished in this episode")
                message += "No order is unfinished in this episode\n"
                logger.success(message)
                self.writter.add_scalar('Train/Unfinished Orders Rate', 0, episode + 1)
                self.writter.add_scalar('Train/Unfinished Late Rate', 0, episode + 1)
            else:
                unfinished = count_unfinished_orders / order_num
                unfinished_late_rate = unfinished_late_orders / count_unfinished_orders
                print(f"Unfinished Orders for Episode {episode+1} is {count_unfinished_orders} out of {order_num} orders ({unfinished}), with {unfinished_late_rate} being late")
                
                message += f"Unfinished Orders for Episode {episode+1} is {count_unfinished_orders} out of {order_num} orders ({unfinished}), with {unfinished_late_rate} being late\n"
                logger.success(message)
                self.writter.add_scalar('Train/Unfinished Orders Rate', unfinished, episode + 1)
                self.writter.add_scalar('Train/Unfinished Late Rate', unfinished_late_rate, episode + 1)
        
            print("\n")      

            # compute return and update nrk
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode+1) % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if (episode+1) % self.log_interval == 0:
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

                self.log_train(train_infos, total_num_steps)

            # eval
            if (episode+1) % self.eval_interval == 0 and self.use_eval:
                self.eval_num += 1
                                
                self.eval(total_num_steps)


        self.writter.close()
        
    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        self.trainer1.prep_rollout()
        self.trainer2.prep_rollout()
        for agent_id, agent in enumerate(self.agents):
            if agent.courier_type == 0:
                value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer1.policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step],
                    # torch.tensor(available_actions[agent_id]),
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
            else:
                value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer2.policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step],
                    # torch.tensor(available_actions[agent_id]),
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
            share_obs,
            rewards,
            dones,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        share_obs = np.array(share_obs)
        
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
        
        share_obs1 = []
        share_obs2 = []
        obs1 = []
        obs2 = []
        rnn_states1 = []
        rnn_states2 = []
        rnn_states_critic1 = []
        rnn_states_critic2 = []
        actions1 = []
        actions2 = []
        action_log_probs1 = []
        action_log_probs2 = []
        values1 = []
        values2 = []
        rewards1 = []
        rewards2 = []
        masks1 = []
        masks2 = []
        
        for agent_id, agent in enumerate(self.agents):

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id].reshape(-1,1),
                masks[:, agent_id],
            )
            
            if agent.courier_type == 0:
                share_obs1.append(share_obs)
                obs1.append(np.array(list(obs[:, agent_id])))
                rnn_states1.append(rnn_states[:, agent_id])
                rnn_states_critic1.append(rnn_states_critic[:, agent_id])
                actions1.append(actions[:, agent_id])
                action_log_probs1.append(action_log_probs[:, agent_id])
                values1.append(values[:, agent_id])
                rewards1.append(rewards[:, agent_id].reshape(-1,1))
                masks1.append(masks[:, agent_id])
            else:
                share_obs2.append(share_obs)
                obs2.append(np.array(list(obs[:, agent_id])))
                rnn_states2.append(rnn_states[:, agent_id])
                rnn_states_critic2.append(rnn_states_critic[:, agent_id])
                actions2.append(actions[:, agent_id])
                action_log_probs2.append(action_log_probs[:, agent_id])
                values2.append(values[:, agent_id])
                rewards2.append(rewards[:, agent_id].reshape(-1,1))
                masks2.append(masks[:, agent_id])
                
        self.buffer1.insert(
            np.stack(share_obs1, axis=1),
            np.stack(obs1, axis=1),
            np.stack(rnn_states1, axis=1),
            np.stack(rnn_states_critic1, axis=1),
            np.stack(actions1, axis=1),
            np.stack(action_log_probs1, axis=1),
            np.stack(values1, axis=1),
            np.stack(rewards1, axis=1),
            np.stack(masks1, axis=1)
        )
        
        self.buffer2.insert(
            np.stack(share_obs2, axis=1),
            np.stack(obs2, axis=1),
            np.stack(rnn_states2, axis=1),
            np.stack(rnn_states_critic2, axis=1),
            np.stack(actions2, axis=1),
            np.stack(action_log_probs2, axis=1),
            np.stack(values2, axis=1),
            np.stack(rewards2, axis=1),
            np.stack(masks2, axis=1)
        )
            

    @torch.no_grad()
    def eval(self, total_num_steps):
        
        eval_obs = self.eval_envs.reset(4, eval=True)
        
        # eval info
        stats = {i: {
            # platform side
            "platform_cost": 0,
            "Hired_finish_num": [],
            "Hired_unfinish_num": [],
            "Hired_reject_num": [],
            "Hired_leisure_time": [],
            "Hired_running_time": [],
            "Hired_waiting_time": [],
            "Crowdsourced_finish_num": [],
            "Crowdsourced_unfinish_num": [],
            "Crowdsourced_reject_num": [],
            "Crowdsourced_leisure_time": [],
            "Crowdsourced_running_time": [],
            "Crowdsourced_waiting_time": [],

            # courier side
            "courier_num": 0,
            "Hired_num": 0,
            "Crowdsourced_num": 0,
            "Crowdsourced_on": 0,
            "Hired_distance_per_episode": [],
            "Crowdsourced_distance_per_episode": [],
            "Hired_actual_speed": [],
            "Crowdsourced_actual_speed": [],
            "overspeed_step": {"ratio0": [], "ratio1": []}, 
            
            # order side
            "order_num": 0,
            "count_dropped_orders": 0,
            "count_unfinished_orders": 0,
            "unfinished_late_orders": 0,
            "count_reject_orders": 0,
            "max_reject_num": 0,
            "late_orders": 0,
            "ETA_usage": [],
            # "order_price": [],
            "order_wait_pair": 0,
            "order_waiting_time": [],
        } for i in range(self.eval_envs.num_envs)}

        self.eval_num_agents = self.eval_envs.envs_map[0].num_couriers
        self.eval_agents = self.eval_envs.envs_map[0].couriers

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

        for eval_step in range(self.eval_episodes_length):
            
            # print("-"*25)
            print(f"THIS IS EVAL STEP {eval_step}")

            for i in range(self.eval_envs.num_envs):
                
                self.log_env(0, eval_step, i, eval=True)
                
            eval_temp_actions_env = []
            
            for agent_id, agent in enumerate(self.eval_agents):
                self.trainer1.prep_rollout()
                self.trainer2.prep_rollout()
                
                if agent.courier_type == 0:
                
                    eval_action, eval_rnn_state = self.trainer1.policy.act(
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
                else:
                    eval_action, eval_rnn_state = self.trainer2.policy.act(
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
            eval_obs, eval_rewards, eval_dones, eval_share_obs = self.eval_envs.step(eval_actions_env)
            
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.eval_num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            algo_stats = {i: {"num0": 0, "num1": 0, "count0": 0, "count1": 0} for i in range(self.eval_envs.num_envs)}

            for i in range(self.eval_envs.num_envs):
                for c in self.eval_envs.envs_map[i].active_couriers:
                    if c.state == 'active':
                        if c.courier_type == 0:
                            algo_stats[i]["num0"] += 1
                            if c.speed > 4:
                                algo_stats[i]["count0"] += 1
                        else:
                            algo_stats[i]["num1"] += 1
                            if c.speed > 4:
                                algo_stats[i]["count1"] += 1

            for i in range(self.eval_envs.num_envs):
                overspeed_ratio0 = algo_stats[i]["count0"] / algo_stats[i]["num0"] if algo_stats[i]["num0"] > 0 else 0
                overspeed_ratio1 = algo_stats[i]["count1"] / algo_stats[i]["num1"] if algo_stats[i]["num1"] > 0 else 0

                stats[i]["overspeed_step"]["ratio0"].append(overspeed_ratio0)
                stats[i]["overspeed_step"]["ratio1"].append(overspeed_ratio1) 
                          
        # Evaluation over periods
        for i in range(self.eval_envs.num_envs):
            env = self.eval_envs.envs_map[i]
            stats[i]["platform_cost"] += env.platform_cost

            for c in env.couriers:
                category = "Hired" if c.courier_type == 0 else "Crowdsourced"
                stats[i][f"{category}_num"] += 1

                if c.travel_distance > 0:
                    stats[i][f"{category}_distance_per_episode"].append(c.travel_distance)

                    stats[i][f"{category}_finish_num"].append(c.finish_order_num)
                    stats[i][f"{category}_unfinish_num"].append(len(c.waybill) + len(c.wait_to_pick))
                    stats[i][f"{category}_reject_num"].append(c.reject_order_num)
                    stats[i][f"{category}_leisure_time"].append(c.total_leisure_time)
                    stats[i][f"{category}_running_time"].append(c.total_running_time)
                    stats[i][f"{category}_waiting_time"].append(c.total_waiting_time)

                if c.actual_speed > 0:
                    stats[i][f"{category}_actual_speed"].append(c.actual_speed)

                if category == "Crowdsourced" and c.state == "active":
                    stats[i]["Crowdsourced_on"] += 1

            stats[i]["courier_num"] = len(env.couriers)

            for o in env.orders:
                if o.status in {'wait_pair', 'wait_pick', 'picked_up'}:
                    stats[i]["count_unfinished_orders"] += 1
                    if o.ETA <= self.eval_envs.envs_map[0].clock:
                        stats[i]["unfinished_late_orders"] += 1
                        
                    if o.status == 'wait_pair':
                        stats[i]["order_wait_pair"] += 1
                    elif o.status == 'picked_up':
                        stats[i]["order_waiting_time"].append(o.wait_time)
                        
                else:
                    stats[i]["count_dropped_orders"] += 1
                    if o.is_late == 1:
                        stats[i]["late_orders"] += 1
                    else:
                        stats[i]["ETA_usage"].append(o.ETA_usage)
                    stats[i]["order_waiting_time"].append(o.wait_time)
                
                if o.reject_count > 0:
                    stats[i]["count_reject_orders"] += 1
                    if stats[i]["max_reject_num"] <= o.reject_count:
                        stats[i]["max_reject_num"] = o.reject_count

            stats[i]["order_num"] = len(env.orders)
        
        message = ''
        for algo_num in range(self.eval_envs.num_envs):
            data = stats[algo_num]
            
            # -----------------------
            # Distance
            hired_distance = np.mean(data["Hired_distance_per_episode"]) / 1000
            var_hired_distance = np.var(data["Hired_distance_per_episode"]) / 1000000
            crowdsourced_distance = np.mean(data["Crowdsourced_distance_per_episode"]) / 1000
            var_crowdsourced_distance = np.var(data["Crowdsourced_distance_per_episode"]) / 1000000
            total_distance = np.mean(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000
            var_total_distance = np.var(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000000
            total_courier_num = data['courier_num']

            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired', hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced', crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total', total_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired Var', var_hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced Var', var_crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total Var', var_total_distance, self.eval_num)
                                
            # -----------------------
            # Average Courier Leisure Time
            Hired_leisure_time = data['Hired_leisure_time']
            Crowdsourced_leisure_time = data['Crowdsourced_leisure_time']
            
            hired_leisure = np.mean(Hired_leisure_time) / 60
            hired_leisure_var = np.var(Hired_leisure_time) / 60**2
            Crowdsourced_leisure = np.mean(Crowdsourced_leisure_time) / 60
            Crowdsourced_leisure_var = np.var(Crowdsourced_leisure_time) / 60**2
            
            avg_leisure = np.mean(Hired_leisure_time + Crowdsourced_leisure_time) / 60
            avg_leisure_var = np.var(Hired_leisure_time + Crowdsourced_leisure_time) / 60**2

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Total', avg_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Hired', hired_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Crowdsourced', Crowdsourced_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Total Var', avg_leisure_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Hired Var', hired_leisure_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Crowdsourced Var', Crowdsourced_leisure_var, self.eval_num)

            # -----------------------
            # Average Courier running Time
            Hired_running_time = data['Hired_running_time']
            Crowdsourced_running_time = data['Crowdsourced_running_time']
            
            hired_running = np.mean(Hired_running_time) / 60
            hired_running_var = np.var(Hired_running_time) / 60**2
            Crowdsourced_running = np.mean(Crowdsourced_running_time) / 60
            Crowdsourced_running_var = np.var(Crowdsourced_running_time) / 60**2
            
            avg_running = np.mean(Hired_running_time + Crowdsourced_running_time) / 60
            avg_running_var = np.var(Hired_running_time + Crowdsourced_running_time) / 60**2

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total', avg_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired', hired_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced', Crowdsourced_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total Var', avg_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired Var', hired_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced Var', Crowdsourced_running_var, self.eval_num)
            
            # -----------------------
            # Overspeed
            Hired_overspeed = data["overspeed_step"]["ratio0"]
            Crowdsourced_overspeed = data["overspeed_step"]["ratio1"]
            hired_overspeed = np.mean(Hired_overspeed)
            crowdsourced_overspeed = np.mean(Crowdsourced_overspeed)
            total_overspeed = np.mean(Hired_overspeed + Crowdsourced_overspeed)

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Total', total_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Hired', hired_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Crowdsourced', crowdsourced_overspeed, self.eval_num)
            
            # ---------------------
            # Service Time per orders
            waiting_time_per_order = np.mean(data['order_waiting_time']) / 60
            var_waiting_time = np.var(data['order_waiting_time']) / 60**2
            print(f"The average waiting time for orders ({data['order_num'] - data['order_wait_pair']}) is {waiting_time_per_order} minutes (Var: {var_waiting_time})")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Order Waiting Time/Total', waiting_time_per_order, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Order Waiting Time/Total_Var', var_waiting_time, self.eval_num)
            
            message += (
                f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished), {data['order_wait_pair']} ({round(100 * data['order_wait_pair'] / data['order_num'], 2)}%) Orders waiting to be paired\n"
                                
                f"Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})\n"
                                                
                f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})\n"
                
                f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})\n"
                                
                f"Hired overspeed rate is {hired_overspeed}, Crowdsourced overspeed rate is {crowdsourced_overspeed}, Total overspeed rate is {total_overspeed}\n"     
            )

            if data['count_dropped_orders'] == 0:
                print(f"No order is dropped in Algo{algo_num+1}")
                self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', -1, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', -1, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var/Crowdsourced Var', 0, self.eval_num)
                
                message += f"No order is dropped in Algo{algo_num+1}\n"
            else:                
                late_rate = data['late_orders'] / data['count_dropped_orders']     
                ETA_usage_rate = np.mean(data['ETA_usage'])
                var_ETA = np.var(data['ETA_usage'])
                print(f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders")
                print(f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})")
                
                message += f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders\n" + f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})\n"

                self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', late_rate, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', ETA_usage_rate, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var', var_ETA, self.eval_num)
            
            if data['count_unfinished_orders'] == 0:
                print(f"No order is unfinished in Algo{algo_num+1}")
                message += f"No order is unfinished in Algo{algo_num+1}\n"
                self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', 0, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', 0, self.eval_num)
            else:
                unfinished = data['count_unfinished_orders'] / data['order_num']
                unfinished_late_rate = data['unfinished_late_orders'] / data['count_unfinished_orders']
                print(f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num']} orders ({unfinished}), with {unfinished_late_rate} being late")
                
                message += f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num']} orders ({unfinished}), with {unfinished_late_rate} being late\n"
                self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', unfinished, self.eval_num)
                self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', unfinished_late_rate, self.eval_num)
           
        logger.success(message)
            
        print("\n")