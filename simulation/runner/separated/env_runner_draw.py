import time
import numpy as np
import torch

from runner.separated.base_runner import Runner
import matplotlib.pyplot as plt

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

        distance_total = []
        episode_rewards = []
        rate_of_overspeed = []
        rate_of_late_order = []
        rate_of_ETA_usage = []
        # reject_rate = []
        order_price_total = []
        # courier_reject_num_total = []
        courier_finish_num_total = []
        leisure_time_total = []
        running_time_total = []
        avg_speed_total = []
        income_total = []
        
        algo1_eval_distance = []
        algo2_eval_distance = []
        algo3_eval_distance = []
        algo4_eval_distance = []
        algo1_eval_episode_rewards = []
        algo2_eval_episode_rewards = []
        algo3_eval_episode_rewards = []
        algo4_eval_episode_rewards = []
        algo1_eval_speed = []
        algo2_eval_speed = []
        algo3_eval_speed = []
        algo4_eval_speed = []
        algo1_eval_overspeed_rate = []
        algo2_eval_overspeed_rate = []
        algo3_eval_overspeed_rate = []
        algo4_eval_overspeed_rate = []
        # algo1_eval_reject_rate = []
        # algo2_eval_reject_rate = []
        # algo3_eval_reject_rate = []
        # algo4_eval_reject_rate = []
        # algo1_eval_reject = []
        # algo2_eval_reject = []
        # algo3_eval_reject = []
        # algo4_eval_reject = []
        algo1_eval_order_price = []
        algo2_eval_order_price = []
        algo3_eval_order_price = []
        algo4_eval_order_price = []
        algo1_eval_income = []
        algo2_eval_income = []
        algo3_eval_income = []
        algo4_eval_income = []
        algo1_eval_finish = []
        algo2_eval_finish = []
        algo3_eval_finish = []
        algo4_eval_finish = []
        algo1_eval_leisure = [] 
        algo2_eval_leisure = []
        algo3_eval_leisure = [] 
        algo4_eval_leisure = []
        algo1_eval_running = [] 
        algo2_eval_running = []
        algo3_eval_running = [] 
        algo4_eval_running = []
        algo1_rate_of_late_order = []
        algo2_rate_of_late_order = []
        algo3_rate_of_late_order = []
        algo4_rate_of_late_order = []
        algo1_rate_of_ETA_usage = []
        algo2_rate_of_ETA_usage = []
        algo3_rate_of_ETA_usage = []
        algo4_rate_of_ETA_usage = []

        for episode in range(episodes):
            print(f"THE START OF EPISODE {episode+1}")
            
            platform_cost_all = 0

            Hired_distance_per_episode = []
            Crowdsourced_distance_per_episode = []
            Hired_num = 0
            Crowdsourced_num = 0
            Crowdsourced_on = 0

            episode_reward_sum = 0

            count_overspeed0 = 0
            num_active_Hired = 0
            count_overspeed1 = 0
            num_active_Crowdsourced = 0

            # count_reject_orders = 0
            # max_reject_num = 0

            late_orders0 = 0
            late_orders1 = 0
            
            ETA_usage0 = []
            ETA_usage1 = []

            count_dropped_orders0 = 0
            count_dropped_orders1 = 0

            order0_price = []
            order1_price = []
            order0_num = 0
            order1_num = 0
            order_wait = 0

            # Hired_reject_num = []
            # Crowdsourced_reject_num = []
            
            Hired_finish_num = []
            Crowdsourced_finish_num = []
            
            Hired_leisure_time = []
            Crowdsourced_leisure_time = []
            
            Hired_running_time = []
            Crowdsourced_running_time = []
            
            Hired_avg_speed = []
            Crowdsourced_avg_speed = []
            
            Hired_income = []
            Crowdsourced_income = []

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            
            obs = self.envs.reset(episode % 5)
            # obs = self.envs.reset(1)
            self.reset_courier_num(self.envs.envs_discrete[0].num_couriers)
            self.num_agents = self.envs.envs_discrete[0].num_couriers

            for step in range(self.episode_length):
                # print("-"*25)
                print(f"THIS IS STEP {step}")
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
                                num_active_Hired += 1
                                if c.speed > 4:
                                    count_overspeed0 += 1
                            else:
                                num_active_Crowdsourced += 1
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
                platform_cost_all += self.envs.envs_discrete[i].platform_cost
                
                for c in self.envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        Hired_num += 1
                        Hired_distance_per_episode.append(c.travel_distance)
                        # Hired_reject_num.append(c.reject_order_num)
                        Hired_finish_num.append(c.finish_order_num)
                        Hired_leisure_time.append(c.total_leisure_time)
                        Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            Hired_avg_speed.append(c.avg_speed)
                        Hired_income.append(c.income)
                    else:
                        Crowdsourced_num += 1
                        Crowdsourced_distance_per_episode.append(c.travel_distance)
                        # Crowdsourced_reject_num.append(c.reject_order_num)
                        Crowdsourced_finish_num.append(c.finish_order_num)
                        Crowdsourced_leisure_time.append(c.total_leisure_time)
                        Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            Crowdsourced_avg_speed.append(c.avg_speed)
                        Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            Crowdsourced_on += 1
                
                for o in self.envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            count_dropped_orders0 += 1
                            if o.is_late == 1:
                                late_orders0 += 1
                            else:
                                ETA_usage0.append(o.ETA_usage)
                        else:
                            count_dropped_orders1 += 1
                            if o.is_late == 1:
                                late_orders1 += 1
                            else:
                                ETA_usage1.append(o.ETA_usage)
                        
                    # if o.reject_count > 0:
                    #     count_reject_orders += 1
                    #     if max_reject_num <= o.reject_count:
                    #         max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            order0_price.append(o.price)
                            order0_num += 1
                        else:
                            order1_price.append(o.price)
                            order1_num += 1              
                            
            print(f"\nThis is Episode {episode+1}\n")                
            print(f"There are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced with {Crowdsourced_on / self.envs.num_envs} on, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1, {order_wait / self.envs.num_envs} ({round(100 * order_wait / (order_wait + order0_num + order1_num), 2)}%) Orders waiting to be paired")                
            episode_rewards.append(episode_reward_sum)
            print(f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}")
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
            
            # ---------------------
            # distance
            distance0 = round(np.mean(Hired_distance_per_episode) / 1000, 2)
            distance_var0 = round(np.var(Hired_distance_per_episode) / 1000000, 2)
            distance1 = round(np.mean(Crowdsourced_distance_per_episode) / 1000, 2)
            distance_var1 = round(np.var(Crowdsourced_distance_per_episode) / 1000000, 2)
            distance = round(np.mean(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000, 2)
            distance_var = round(np.var(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000000, 2)
            distance_total.append([distance0, distance_var0, distance1, distance_var1, distance, distance_var])
            print(
                f"Average Travel Distance per Hired: {distance0} km (Var: {distance_var0}), "
                f"Crowdsourced: {distance1} km (Var: {distance_var1}), "
                f"Total: {distance} km (Var: {distance_var})"
            )
            self.writter.add_scalar('Total Distance/Total', distance, episode + 1)
            self.writter.add_scalar('Total Distance/Total_Var', distance_var, episode + 1)
            self.writter.add_scalar('Total Distance/Hired', distance0, episode + 1)
            self.writter.add_scalar('Total Distance/Hired_Var', distance_var0, episode + 1)
            self.writter.add_scalar('Total Distance/Crowdsourced', distance1, episode + 1)
            self.writter.add_scalar('Total Distance/Crowdsourced_Var', distance_var1, episode + 1)

            # ---------------------
            # average speed
            avg0_speed = round(np.mean(Hired_avg_speed), 2)
            var0_speed = round(np.var(Hired_avg_speed), 2)
            avg1_speed = round(np.mean(Crowdsourced_avg_speed), 2)
            var1_speed = round(np.var(Crowdsourced_avg_speed), 2)
            avg_speed = round(np.mean(Hired_avg_speed + Crowdsourced_avg_speed), 2)
            var_speed = round(np.var(Hired_avg_speed + Crowdsourced_avg_speed), 2)
            avg_speed_total.append([avg0_speed, var0_speed, avg1_speed, var1_speed, avg_speed, var_speed])    
            print(
                f"Hired average speed is {avg0_speed} m/s (Var: {var0_speed}), "
                f"Crowdsourced average speed is {avg1_speed} m/s (Var: {var1_speed}), "
                f"Average speed per courier is {avg_speed} m/s (Var: {var_speed})"
            )
            self.writter.add_scalar('Average Speed/Total', avg_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Total_Var', var_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Hired', avg0_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Hired_Var', var0_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Crowdsourced', avg1_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Crowdsourced_Var', var1_speed, episode + 1)

            # ---------------------
            # overspeed rate
            overspeed = (count_overspeed0 + count_overspeed1) / (num_active_Hired + num_active_Crowdsourced)
            overspeed = round(overspeed, 2)
            overspeed0 = round(count_overspeed0 / num_active_Hired, 2)
            overspeed1 = round(count_overspeed1 / num_active_Crowdsourced, 2)
            print(f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total rate - {overspeed}")
            rate_of_overspeed.append([overspeed0, overspeed1, overspeed])
            self.writter.add_scalar('Overspeed Rate/Total rate', overspeed, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Hired', overspeed0, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Crowdsourced', overspeed1, episode + 1)
            
            # ---------------------
            # order reject rate
            # reject_rate_per_episode = round(count_reject_orders / (order0_num + order1_num), 2) # reject once or twice or more
            # reject_rate.append(reject_rate_per_episode)
            # print(f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most")
            # self.writter.add_scalar('Reject rate', reject_rate_per_episode, episode + 1)
            
            # ---------------------
            # courier reject number
            # avg_reject0 = round(np.mean(Hired_reject_num), 2)
            # var_reject0 = round(np.var(Hired_reject_num), 2)
            # avg_reject1 = round(np.mean(Crowdsourced_reject_num), 2)
            # var_reject1 = round(np.var(Crowdsourced_reject_num), 2)
            # avg_reject = round(np.mean(Hired_reject_num + Crowdsourced_reject_num), 2)
            # var_reject = round(np.var(Hired_reject_num + Crowdsourced_reject_num), 2)
            # courier_reject_num_total.append([avg_reject0, var_reject0, avg_reject1, var_reject1, avg_reject, var_reject])
            # print(
            #     f"The average rejection number for Episode {episode+1}: Hired - {avg_reject0} (Var: {var_reject0}), "
            #     f"Crowdsourced - {avg_reject1} (Var: {var_reject1}), "
            #     f"Total - {avg_reject} (Var: {var_reject})"
            # )
            # self.writter.add_scalar('Reject Rate/Total', avg_reject, episode + 1)
            # self.writter.add_scalar('Reject Rate/Total_Var', var_reject, episode + 1)
            # self.writter.add_scalar('Reject Rate/Hired', avg_reject0, episode + 1)
            # self.writter.add_scalar('Reject Rate/Hired_Var', var_reject0, episode + 1)
            # self.writter.add_scalar('Reject Rate/Crowdsourced', avg_reject1, episode + 1)
            # self.writter.add_scalar('Reject Rate/Crowdsourced_Var', var_reject1, episode + 1)
            
            # ---------------------
            # average order price for courier
            price_per_order0 = round(np.mean(order0_price), 2)
            var_price0 = round(np.var(order0_price), 2)
            price_per_order1 = round(np.mean(order1_price), 2)
            var_price1 = round(np.var(order0_price), 2)
            price_per_order = round(np.mean(order0_price + order1_price), 2)
            var_price = round(np.var(order0_price + order1_price), 2)
            order_price_total.append([price_per_order0, var_price0, price_per_order1, var_price1, price_per_order, var_price])
            print(f"The average price of Hired's order is {price_per_order0} dollar (Var: {var_price0}) with {order0_num} orders, Crowdsourced's is {price_per_order1} dollar (Var: {var_price1}) with {order1_num} orders and for all is {price_per_order} dollar (Var: {var_price})")
            self.writter.add_scalar('Average Price/Total', price_per_order, episode + 1)
            self.writter.add_scalar('Average Price/Total_Var', var_price, episode + 1)
            self.writter.add_scalar('Average Price/Hired', price_per_order0, episode + 1)
            self.writter.add_scalar('Average Price/Hired_Var', var_price0, episode + 1)
            self.writter.add_scalar('Average Price/Crowdsourced', price_per_order1, episode + 1)
            self.writter.add_scalar('Average Price/Crowdsourced_Var', var_price1, episode + 1)

            # ---------------------
            # average courier income
            income0 = round(np.mean(Hired_income), 2)
            var_income0 = round(np.var(Hired_income), 2)
            income1 = round(np.mean(Crowdsourced_income), 2)
            var_income1 = round(np.var(Crowdsourced_income), 2)
            income = round(np.mean(Hired_income + Crowdsourced_income), 2)
            var_income = round(np.var(Hired_income + Crowdsourced_income), 2)
            platform_cost = round(platform_cost_all / self.envs.num_envs, 2)
            income_total.append([income0, var_income0, income1, var_income1, income, var_income, platform_cost])
            print(f"Hired's average income is {income0} dollar (Var: {var_income0}), Crowdsourced's average income is {income1} dollar (Var: {var_income1}) and Total income per courier is {income} dollar (Var: {var_income})")
            print(f"The platform total cost is {platform_cost} dollar")
            self.writter.add_scalar('Average Income/Total', income, episode + 1)
            self.writter.add_scalar('Average Income/Total_Var', var_income, episode + 1)
            self.writter.add_scalar('Average Income/Hired', income0, episode + 1)
            self.writter.add_scalar('Average Income/Hired_Var', var_income0, episode + 1)
            self.writter.add_scalar('Average Income/Crowdsourced', income1, episode + 1)
            self.writter.add_scalar('Average Income/Crowdsourced_Var', var_income1, episode + 1)
            self.writter.add_scalar('Platform Cost', platform_cost, episode + 1)
            
            # ---------------------
            # average courier finishing order number
            avg_finish0 = round(np.mean(Hired_finish_num), 2)
            var_finish0 = round(np.var(Hired_finish_num), 2)
            avg_finish1 = round(np.mean(Crowdsourced_finish_num), 2)
            var_finish1 = round(np.var(Crowdsourced_finish_num), 2)
            avg_finish = round(np.mean(Hired_finish_num + Crowdsourced_finish_num), 2)
            var_finish = round(np.var(Hired_finish_num + Crowdsourced_finish_num), 2)
            courier_finish_num_total.append([avg_finish0, var_finish0, avg_finish1, var_finish1, avg_finish, var_finish])
            print(
                f"Hired finishes average {avg_finish0} orders (Var: {var_finish0}), "
                f"Crowdsourced finishes average {avg_finish1} orders (Var: {var_finish1}), "
                f"Total finish number per courier is {avg_finish} (Var: {var_finish})"
            )
            self.writter.add_scalar('Average Finish/Total', avg_finish, episode + 1)
            self.writter.add_scalar('Average Finish/Total_Var', var_finish, episode + 1)
            self.writter.add_scalar('Average Finish/Hired', avg_finish0, episode + 1)
            self.writter.add_scalar('Average Finish/Hired_Var', var_finish0, episode + 1)
            self.writter.add_scalar('Average Finish/Crowdsourced', avg_finish1, episode + 1)
            self.writter.add_scalar('Average Finish/Crowdsourced_Var', var_finish1, episode + 1)
            
            # ---------------------
            # average courier leisure time
            avg0_leisure = round(np.mean(Hired_leisure_time) / 60, 2)
            var_leisure0 = round(np.var(Hired_leisure_time) / 60**2, 2)
            avg1_leisure = round(np.mean(Crowdsourced_leisure_time) / 60, 2)
            var_leisure1 = round(np.var(Crowdsourced_leisure_time) / 60**2, 2)
            avg_leisure = round(np.mean(Hired_leisure_time + Crowdsourced_leisure_time) / 60, 2)
            var_leisure = round(np.var(Hired_leisure_time + Crowdsourced_leisure_time) / 60**2, 2)
            leisure_time_total.append([avg0_leisure, var_leisure0, avg1_leisure, var_leisure1, avg_leisure, var_leisure])
            print(f"Hired leisure time is {avg0_leisure} minutes (Var: {var_leisure0}), Crowdsourced leisure time is {avg1_leisure} minutes (Var: {var_leisure1}) and Total leisure time per courier is {avg_leisure} minutes (Var: {var_leisure})")
            self.writter.add_scalar('Average Leisure Time/Total', avg_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Total_Var', var_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Hired', avg0_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Hired_Var', var_leisure0, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Crowdsourced', avg1_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Crowdsourced_Var', var_leisure1, episode + 1)
            
            # ---------------------
            # average courier utilization time
            avg0_running = round(np.mean(Hired_running_time) / 60, 2)
            var_running0 = round(np.var(Hired_running_time) / 60**2, 2)
            avg1_running = round(np.mean(Crowdsourced_running_time) / 60, 2)
            var_running1 = round(np.var(Crowdsourced_running_time) / 60**2, 2)
            avg_running = round(np.mean(Hired_running_time + Crowdsourced_running_time) / 60, 2)
            var_running = round(np.var(Hired_running_time + Crowdsourced_running_time) / 60**2, 2)
            running_time_total.append([avg0_running, var_running0, avg1_running, var_running1, avg_running, var_running])
            print(f"Hired running time is {avg0_running} minutes (Var: {var_running0}), Crowdsourced running time is {avg1_running} minutes (Var: {var_running1}) and Total running time per courier is {avg_running} minutes (Var: {var_running})")
            self.writter.add_scalar('Average running Time/Total', avg_running, episode + 1)
            self.writter.add_scalar('Average running Time/Total_Var', var_running, episode + 1)
            self.writter.add_scalar('Average running Time/Hired', avg0_running, episode + 1)
            self.writter.add_scalar('Average running Time/Hired_Var', var_running0, episode + 1)
            self.writter.add_scalar('Average running Time/Crowdsourced', avg1_running, episode + 1)
            self.writter.add_scalar('Average running Time/Crowdsourced_Var', var_running1, episode + 1)
            
            message = (
                f"\nThis is Train Episode {episode+1}\nThere are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1, {order_wait / self.envs.num_envs} ({round(100 * order_wait / (order_wait + order0_num + order1_num), 2)}%) Orders waiting to be paired\n"
                f"Average Travel Distance for Episode {episode+1}: Hired - {distance0} km (Var: {distance_var0}), Crowdsourced - {distance1} km (Var: {distance_var1}), Total - {distance} km (Var: {distance_var})\n"
                f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}\n"
                f"The average speed for Episode {episode+1}: Hired - {avg0_speed} m/s (Var: {var0_speed}), Crowdsourced - {avg1_speed} m/s (Var: {var1_speed}), Total - {avg_speed} m/s (Var: {var_speed})\n"
                f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total - {overspeed}\n"
                # f"Order rejection rate for Episode {episode+1}: {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most\n"
                # f"The average rejection number for Episode {episode+1}: Hired - {avg_reject0} (Var: {var_reject0}), Crowdsourced - {avg_reject1} (Var: {var_reject1}), Total - {avg_reject} (Var: {var_reject})\n"
                f"The average price for Episode {episode+1}: Hired - {price_per_order0} dollar (Var: {var_price0}) with {order0_num} orders, Crowdsourced - {price_per_order1} dollar (Var: {var_price1}) with {order1_num} orders, Total - {price_per_order} dollar (Var: {var_price})\n"
                f"The average income for Episode {episode+1}: Hired - {income0} dollar (Var: {var_income0}), Crowdsourced - {income1} dollar (Var: {var_income1}), Total - {income} dollar (Var: {var_income})\n"
                f"The platform total cost is {platform_cost} dollar\n"
                f"The average finish number for Episode {episode+1}: Hired - {avg_finish0} (Var: {var_finish0}), Crowdsourced - {avg_finish1} (Var: {var_finish1}), Total - {avg_finish} (Var: {var_finish})\n"
                f"The average leisure time for Episode {episode+1}: Hired - {avg0_leisure} minutes (Var: {var_leisure0}), Crowdsourced - {avg1_leisure} minutes (Var: {var_leisure1}), Total - {avg_leisure} minutes (Var: {var_leisure})\n"
                f"The average running time for Episode {episode+1}: Hired - {avg0_running} minutes (Var: {var_running0}), Crowdsourced - {avg1_running} minutes (Var: {var_running1}), Total - {avg_running} minutes (Var: {var_running})\n"
            )

            if count_dropped_orders0 + count_dropped_orders1 == 0:
                print("No order is dropped in this episode")
                rate_of_late_order.append([-1, -1, -1])
                rate_of_ETA_usage.append([-1, 0, -1, 0, -1, 0])
                message += "No order is dropped in this episode\n"
                logger.success(message)
                self.writter.add_scalar('Late Orders Rate/Total', -1, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Hired', -1, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Crowdsourced', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total_Var', 0, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired_Var', 0, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced_Var', 0, episode + 1)
            else:
                if count_dropped_orders0 != 0:
                    late_rate0 = round(late_orders0 / count_dropped_orders0, 2)
                    ETA_usage_rate0 = round(np.mean(ETA_usage0), 2)
                    Var_ETA0 = round(np.var(ETA_usage0), 2)
                else:
                    late_rate0 = -1
                    ETA_usage_rate0 = -1
                    Var_ETA0 = 0
                    
                if count_dropped_orders1 != 0:                    
                    late_rate1 = round(late_orders1 / count_dropped_orders1, 2)
                    ETA_usage_rate1 = round(np.mean(ETA_usage1), 2)
                    Var_ETA1 = round(np.var(ETA_usage1), 2)
                else:
                    late_rate1 = -1
                    ETA_usage_rate1 = -1
                    Var_ETA1 = 0
                    
                late_rate = round((late_orders0 + late_orders1) / (count_dropped_orders0 + count_dropped_orders1), 2)
                print(f"Rate of Late Orders for Episode {episode+1}: Hired - {late_rate0}, Crowdsourced - {late_rate1}, Total - {late_rate}")
                rate_of_late_order.append([late_rate0, late_rate1, late_rate])

                ETA_usage_rate = round(np.mean(ETA_usage0 + ETA_usage1), 2)
                Var_ETA = round(np.var(ETA_usage0 + ETA_usage1), 2)
                print(f"Rate of ETA Usage for Episode {episode+1}: Hired - {ETA_usage_rate0} (Var: {Var_ETA0}), Crowdsourced - {ETA_usage_rate1} (Var: {Var_ETA1}), Total - {ETA_usage_rate} (Var: {Var_ETA})")
                rate_of_ETA_usage.append([ETA_usage_rate0, Var_ETA0, ETA_usage_rate1, Var_ETA1, ETA_usage_rate, Var_ETA])
                
                message += f"Rate of Late Orders for Episode {episode+1}: Hired - {late_rate0}, Crowdsourced - {late_rate1}, Total - {late_rate}\n" + f"Rate of ETA Usage for Episode {episode+1}: Hired - {ETA_usage_rate0} (Var: {Var_ETA0}), Crowdsourced - {ETA_usage_rate1} (Var: {Var_ETA1}), Total - {ETA_usage_rate} (Var: {Var_ETA})\n"
                logger.success(message)
                
                self.writter.add_scalar('Late Orders Rate/Total', late_rate, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Hired', late_rate0, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Crowdsourced', late_rate1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total', ETA_usage_rate, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total_Var', Var_ETA, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired', ETA_usage_rate0, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired', Var_ETA0, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced', ETA_usage_rate1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced', Var_ETA1, episode + 1)
            
            print("\n")
                        

            # compute return and update nrk
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
                    algo3_eval_episode_rewards_sum,
                    algo4_eval_episode_rewards_sum,
                    
                    algo1_distance0,
                    algo1_distance1,
                    algo1_distance,
                    algo2_distance0,
                    algo2_distance1,
                    algo2_distance,
                    algo3_distance0,
                    algo3_distance1,
                    algo3_distance,
                    algo4_distance0,
                    algo4_distance1,
                    algo4_distance,
                    algo1_var0_distance,
                    algo1_var1_distance,
                    algo1_var_distance,
                    algo2_var0_distance,
                    algo2_var1_distance,
                    algo2_var_distance,
                    algo3_var0_distance,
                    algo3_var1_distance,
                    algo3_var_distance,
                    algo4_var0_distance,
                    algo4_var1_distance,
                    algo4_var_distance,

                    algo1_avg0_speed,
                    algo1_avg1_speed,
                    algo1_avg_speed,
                    algo2_avg0_speed,
                    algo2_avg1_speed,
                    algo2_avg_speed,
                    algo3_avg0_speed,
                    algo3_avg1_speed,
                    algo3_avg_speed,
                    algo4_avg0_speed,
                    algo4_avg1_speed,
                    algo4_avg_speed,
                    algo1_var0_speed,
                    algo1_var1_speed,
                    algo1_var_speed,
                    algo2_var0_speed,
                    algo2_var1_speed,
                    algo2_var_speed,
                    algo3_var0_speed,
                    algo3_var1_speed,
                    algo3_var_speed,
                    algo4_var0_speed,
                    algo4_var1_speed,
                    algo4_var_speed,
                    
                    algo1_overspeed0,
                    algo1_overspeed1,
                    algo1_overspeed,
                    algo2_overspeed0,
                    algo2_overspeed1,
                    algo2_overspeed,
                    algo3_overspeed0,
                    algo3_overspeed1,
                    algo3_overspeed,
                    algo4_overspeed0,
                    algo4_overspeed1,
                    algo4_overspeed,
                    
                    # algo1_reject_rate_per_episode,
                    # algo2_reject_rate_per_episode,
                    # algo3_reject_rate_per_episode,
                    # algo4_reject_rate_per_episode,
                    
                    # algo1_reject0,
                    # algo1_reject1,
                    # algo1_reject,
                    # algo2_reject0,
                    # algo2_reject1,
                    # algo2_reject,
                    # algo3_reject0,
                    # algo3_reject1,
                    # algo3_reject,
                    # algo4_reject0,
                    # algo4_reject1,
                    # algo4_reject,
                    # algo1_var0_reject,
                    # algo1_var1_reject,
                    # algo1_var_reject,
                    # algo2_var0_reject,
                    # algo2_var1_reject,
                    # algo2_var_reject,
                    # algo3_var0_reject,
                    # algo3_var1_reject,
                    # algo3_var_reject,
                    # algo4_var0_reject,
                    # algo4_var1_reject,
                    # algo4_var_reject,
                    
                    algo1_price_per_order0,
                    algo1_price_per_order1,
                    algo1_price_per_order,
                    algo2_price_per_order0,
                    algo2_price_per_order1,
                    algo2_price_per_order,
                    algo3_price_per_order0,
                    algo3_price_per_order1,
                    algo3_price_per_order,
                    algo4_price_per_order0,
                    algo4_price_per_order1,
                    algo4_price_per_order,
                    algo1_var0_price,
                    algo1_var1_price,
                    algo1_var_price,
                    algo2_var0_price,
                    algo2_var1_price,
                    algo2_var_price,
                    algo3_var0_price,
                    algo3_var1_price,
                    algo3_var_price,
                    algo4_var0_price,
                    algo4_var1_price,
                    algo4_var_price,
                    
                    algo1_income0,
                    algo1_income1,
                    algo1_income,
                    platform_cost1,
                    algo2_income0,
                    algo2_income1,
                    algo2_income,
                    platform_cost2,
                    algo3_income0,
                    algo3_income1,
                    algo3_income,
                    platform_cost3,
                    algo4_income0,
                    algo4_income1,
                    algo4_income,
                    platform_cost4,
                    algo1_var0_income,
                    algo1_var1_income,
                    algo1_var_income,
                    algo2_var0_income,
                    algo2_var1_income,
                    algo2_var_income,
                    algo3_var0_income,
                    algo3_var1_income,
                    algo3_var_income,
                    algo4_var0_income,
                    algo4_var1_income,
                    algo4_var_income,
                    
                    algo1_finish0,
                    algo1_finish1,
                    algo1_finish,
                    algo2_finish0,
                    algo2_finish1,
                    algo2_finish,
                    algo3_finish0,
                    algo3_finish1,
                    algo3_finish,
                    algo4_finish0,
                    algo4_finish1,
                    algo4_finish,
                    algo1_var0_finish,
                    algo1_var1_finish,
                    algo1_var_finish,
                    algo2_var0_finish,
                    algo2_var1_finish,
                    algo2_var_finish,
                    algo3_var0_finish,
                    algo3_var1_finish,
                    algo3_var_finish,
                    algo4_var0_finish,
                    algo4_var1_finish,
                    algo4_var_finish,
                    
                    algo1_avg0_leisure,
                    algo1_avg1_leisure,
                    algo1_avg_leisure,
                    algo2_avg0_leisure,
                    algo2_avg1_leisure,
                    algo2_avg_leisure,
                    algo3_avg0_leisure,
                    algo3_avg1_leisure,
                    algo3_avg_leisure,
                    algo4_avg0_leisure,
                    algo4_avg1_leisure,
                    algo4_avg_leisure,
                    algo1_var0_leisure,
                    algo1_var1_leisure,
                    algo1_var_leisure,
                    algo2_var0_leisure,
                    algo2_var1_leisure,
                    algo2_var_leisure,
                    algo3_var0_leisure,
                    algo3_var1_leisure,
                    algo3_var_leisure,
                    algo4_var0_leisure,
                    algo4_var1_leisure,
                    algo4_var_leisure,
                    
                    algo1_avg0_running,
                    algo1_avg1_running,
                    algo1_avg_running,
                    algo2_avg0_running,
                    algo2_avg1_running,
                    algo2_avg_running,
                    algo3_avg0_running,
                    algo3_avg1_running,
                    algo3_avg_running,
                    algo4_avg0_running,
                    algo4_avg1_running,
                    algo4_avg_running,
                    algo1_var0_running,
                    algo1_var1_running,
                    algo1_var_running,
                    algo2_var0_running,
                    algo2_var1_running,
                    algo2_var_running,
                    algo3_var0_running,
                    algo3_var1_running,
                    algo3_var_running,
                    algo4_var0_running,
                    algo4_var1_running,
                    algo4_var_running,
                    
                    algo1_late_rate0,
                    algo1_late_rate1,
                    algo1_late_rate,
                    algo2_late_rate0,
                    algo2_late_rate1,
                    algo2_late_rate,
                    algo3_late_rate0,
                    algo3_late_rate1,
                    algo3_late_rate,
                    algo4_late_rate0,
                    algo4_late_rate1,
                    algo4_late_rate,
                    
                    algo1_ETA_usage_rate0,
                    algo1_ETA_usage_rate1,
                    algo1_ETA_usage_rate,
                    algo2_ETA_usage_rate0,
                    algo2_ETA_usage_rate1,
                    algo2_ETA_usage_rate,
                    algo3_ETA_usage_rate0,
                    algo3_ETA_usage_rate1,
                    algo3_ETA_usage_rate,
                    algo4_ETA_usage_rate0,
                    algo4_ETA_usage_rate1,
                    algo4_ETA_usage_rate,
                    algo1_var0_ETA,
                    algo1_var1_ETA,
                    algo1_var_ETA,
                    algo2_var0_ETA,
                    algo2_var1_ETA,
                    algo2_var_ETA,
                    algo3_var0_ETA,
                    algo3_var1_ETA,
                    algo3_var_ETA,
                    algo4_var0_ETA,
                    algo4_var1_ETA,
                    algo4_var_ETA,                    
                ) = self.eval(total_num_steps)

                algo1_eval_distance.append([algo1_distance0, algo1_var0_distance, algo1_distance1, algo1_var1_distance, algo1_distance, algo1_var_distance])
                algo2_eval_distance.append([algo2_distance0,algo2_var0_distance, algo2_distance1, algo2_var1_distance, algo2_distance, algo2_var_distance])
                algo3_eval_distance.append([algo3_distance0, algo3_var0_distance, algo3_distance1, algo3_var1_distance, algo3_distance, algo3_var_distance])
                algo4_eval_distance.append([algo4_distance0, algo4_var0_distance, algo4_distance1, algo4_var1_distance, algo4_distance, algo4_var_distance])

                algo1_eval_episode_rewards.append(algo1_eval_episode_rewards_sum)
                algo2_eval_episode_rewards.append(algo2_eval_episode_rewards_sum)
                algo3_eval_episode_rewards.append(algo3_eval_episode_rewards_sum)
                algo4_eval_episode_rewards.append(algo4_eval_episode_rewards_sum)
                
                algo1_eval_speed.append([algo1_avg0_speed, algo1_var0_speed, algo1_avg1_speed, algo1_var1_speed, algo1_avg_speed, algo1_var_speed])
                algo2_eval_speed.append([algo2_avg0_speed, algo2_var0_speed, algo2_avg1_speed, algo2_var1_speed, algo2_avg_speed, algo2_var_speed])
                algo3_eval_speed.append([algo3_avg0_speed, algo3_var0_speed, algo3_avg1_speed, algo3_var1_speed, algo3_avg_speed, algo3_var_speed])
                algo4_eval_speed.append([algo4_avg0_speed, algo4_var0_speed, algo4_avg1_speed, algo4_var1_speed, algo4_avg_speed, algo4_var_speed])

                algo1_eval_overspeed_rate.append([algo1_overspeed0, algo1_overspeed1, algo1_overspeed])
                algo2_eval_overspeed_rate.append([algo2_overspeed0, algo2_overspeed1, algo2_overspeed])
                algo3_eval_overspeed_rate.append([algo3_overspeed0, algo3_overspeed1, algo3_overspeed])
                algo4_eval_overspeed_rate.append([algo4_overspeed0, algo4_overspeed1, algo4_overspeed])
                
                # algo1_eval_reject_rate.append(algo1_reject_rate_per_episode)
                # algo2_eval_reject_rate.append(algo2_reject_rate_per_episode)
                # algo3_eval_reject_rate.append(algo3_reject_rate_per_episode)
                # algo4_eval_reject_rate.append(algo4_reject_rate_per_episode)

                # algo1_eval_reject.append([algo1_reject0, algo1_var0_reject, algo1_reject1, algo1_var1_reject, algo1_reject, algo1_var_reject])
                # algo2_eval_reject.append([algo2_reject0, algo2_var0_reject, algo2_reject1, algo2_var1_reject, algo2_reject, algo2_var_reject])
                # algo3_eval_reject.append([algo3_reject0, algo3_var0_reject, algo3_reject1, algo3_var1_reject, algo3_reject, algo3_var_reject])
                # algo4_eval_reject.append([algo4_reject0, algo4_var0_reject, algo4_reject1, algo4_var1_reject, algo4_reject, algo4_var_reject])

                algo1_eval_order_price.append([algo1_price_per_order0, algo1_var0_price, algo1_price_per_order1, algo1_var1_price, algo1_price_per_order, algo1_var_price])
                algo2_eval_order_price.append([algo2_price_per_order0, algo2_var0_price, algo2_price_per_order1, algo2_var1_price, algo2_price_per_order, algo2_var_price])
                algo3_eval_order_price.append([algo3_price_per_order0, algo3_var0_price, algo3_price_per_order1, algo3_var1_price, algo3_price_per_order, algo3_var_price])
                algo4_eval_order_price.append([algo4_price_per_order0, algo4_var0_price, algo4_price_per_order1, algo4_var1_price, algo4_price_per_order, algo4_var_price])

                algo1_eval_income.append([algo1_income0, algo1_var0_income, algo1_income1, algo1_var1_income, algo1_income, algo1_var_income, platform_cost1])
                algo2_eval_income.append([algo2_income0, algo2_var0_income, algo2_income1, algo2_var1_income, algo2_income, algo2_var_income, platform_cost2])
                algo3_eval_income.append([algo3_income0, algo3_var0_income, algo3_income1, algo3_var1_income, algo3_income, algo3_var_income, platform_cost3])
                algo4_eval_income.append([algo4_income0, algo4_var0_income, algo4_income1, algo4_var1_income, algo4_income, algo4_var_income, platform_cost4])

                algo1_eval_finish.append([algo1_finish0, algo1_var0_finish, algo1_finish1, algo1_var1_finish, algo1_finish, algo1_var_finish])
                algo2_eval_finish.append([algo2_finish0, algo2_var0_finish, algo2_finish1, algo2_var1_finish, algo2_finish, algo2_var_finish])
                algo3_eval_finish.append([algo3_finish0, algo3_var0_finish, algo3_finish1, algo3_var1_finish, algo3_finish, algo3_var_finish])
                algo4_eval_finish.append([algo4_finish0, algo4_var0_finish, algo4_finish1, algo4_var1_finish, algo4_finish, algo4_var_finish])

                algo1_eval_leisure.append([algo1_avg0_leisure, algo1_var0_leisure, algo1_avg1_leisure, algo1_var1_leisure, algo1_avg_leisure, algo1_var_leisure])
                algo2_eval_leisure.append([algo2_avg0_leisure, algo2_var0_leisure, algo2_avg1_leisure, algo2_var1_leisure, algo2_avg_leisure, algo2_var_leisure])
                algo3_eval_leisure.append([algo3_avg0_leisure, algo3_var0_leisure, algo3_avg1_leisure, algo3_var1_leisure, algo3_avg_leisure, algo3_var_leisure])
                algo4_eval_leisure.append([algo4_avg0_leisure, algo4_var0_leisure, algo4_avg1_leisure, algo4_var1_leisure, algo4_avg_leisure, algo4_var_leisure])
                
                algo1_eval_running.append([algo1_avg0_running, algo1_var0_running, algo1_avg1_running, algo1_var1_running, algo1_avg_running, algo1_var_running])
                algo2_eval_running.append([algo2_avg0_running, algo2_var0_running, algo2_avg1_running, algo2_var1_running, algo2_avg_running, algo2_var_running])
                algo3_eval_running.append([algo3_avg0_running, algo3_var0_running, algo3_avg1_running, algo3_var1_running, algo3_avg_running, algo3_var_running])
                algo4_eval_running.append([algo4_avg0_running, algo4_var0_running, algo4_avg1_running, algo4_var1_running, algo4_avg_running, algo4_var_running])

                algo1_rate_of_late_order.append([algo1_late_rate0, algo1_late_rate1, algo1_late_rate])
                algo2_rate_of_late_order.append([algo2_late_rate0, algo2_late_rate1, algo2_late_rate])
                algo3_rate_of_late_order.append([algo3_late_rate0, algo3_late_rate1, algo3_late_rate])
                algo4_rate_of_late_order.append([algo4_late_rate0, algo4_late_rate1, algo4_late_rate])

                algo1_rate_of_ETA_usage.append([algo1_ETA_usage_rate0, algo1_var0_ETA, algo1_ETA_usage_rate1, algo1_var1_ETA, algo1_ETA_usage_rate, algo1_var_ETA])
                algo2_rate_of_ETA_usage.append([algo2_ETA_usage_rate0, algo2_var0_ETA, algo2_ETA_usage_rate1, algo2_var1_ETA, algo2_ETA_usage_rate, algo2_var_ETA])
                algo3_rate_of_ETA_usage.append([algo3_ETA_usage_rate0, algo3_var0_ETA, algo3_ETA_usage_rate1, algo3_var1_ETA, algo3_ETA_usage_rate, algo3_var_ETA])
                algo4_rate_of_ETA_usage.append([algo4_ETA_usage_rate0, algo4_var0_ETA, algo4_ETA_usage_rate1, algo4_var1_ETA, algo4_ETA_usage_rate, algo4_var_ETA])

        self.writter.close()
        
        # draw the Train graph
        Hired_distances = [d[0] for d in distance_total]
        var_Hired = [d[1] for d in distance_total]
        Crowdsourced_distances = [d[2] for d in distance_total]
        var_Crowdsourced = [d[3] for d in distance_total]
        courier_distances = [d[4] for d in distance_total]
        var_distance = [d[4] for d in distance_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_distances, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_distances)), np.array(Hired_distances) - np.array(var_Hired), np.array(Hired_distances) + np.array(var_Hired), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_distances, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_distances)), np.array(Crowdsourced_distances) - np.array(var_Crowdsourced), np.array(Crowdsourced_distances) + np.array(var_Crowdsourced), color='orange', alpha=0.2)
        plt.plot(courier_distances, label="Courier", color='green')
        plt.fill_between(range(len(courier_distances)), np.array(courier_distances) - np.array(var_distance), np.array(courier_distances) + np.array(var_distance), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Total Distances')
        plt.title('Train: Distance over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_Distance.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Train: Reward over Episodes')
        plt.grid(True)
        plt.savefig('Train_reward_curve.png')
        plt.close()
        
        Hired_speed = [s[0] for s in avg_speed_total]
        var_Hired_speed = [s[1] for s in avg_speed_total]
        Crowdsourced_speed = [s[2] for s in avg_speed_total]
        var_Crowdsourced_speed = [s[3] for s in avg_speed_total]
        courier_speed = [s[4] for s in avg_speed_total]
        var_speed = [s[5] for s in avg_speed_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_speed, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_speed)), np.array(Hired_speed) - np.array(var_Hired_speed), np.array(Hired_speed) + np.array(var_Hired_speed), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_speed, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_speed)), np.array(Crowdsourced_speed) - np.array(var_Crowdsourced_speed), np.array(Crowdsourced_speed) + np.array(var_Crowdsourced_speed), color='orange', alpha=0.2)
        plt.plot(courier_speed, label="Courier", color='green')
        plt.fill_between(range(len(courier_speed)), np.array(courier_speed) - np.array(var_speed), np.array(courier_speed) + np.array(var_speed), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Speed')
        plt.title('Train: Average Speed over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_speed.png')
        plt.close()
        
        Hired_overspeed = [r[0] for r in rate_of_overspeed]
        Crowdsourced_overspeed = [r[1] for r in rate_of_overspeed]
        courier_overspeed = [r[2] for r in rate_of_overspeed]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_overspeed, label="Hired", color='blue')
        plt.plot(Crowdsourced_overspeed, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_overspeed, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Overspeed')
        plt.title('Train: rate of overspeed over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_rate_of_overspeed.png')
        plt.close()
        
        # plt.figure(figsize=(12, 8))
        # plt.plot(reject_rate)
        # plt.xlabel('Episodes')
        # plt.ylabel('Order Reject Rate')
        # plt.title('Train: order reject rate over Episodes')
        # plt.grid(True)
        # plt.savefig('Train_order_reject_rate.png')
        # plt.close()
        
        # avg_reject0 = [r[0] for r in courier_reject_num_total]
        # var_reject0 = [r[1] for r in courier_reject_num_total]
        # avg_reject1 = [r[2] for r in courier_reject_num_total]
        # var_reject1 = [r[3] for r in courier_reject_num_total]
        # avg_reject = [r[4] for r in courier_reject_num_total]
        # var_reject = [r[5] for r in courier_reject_num_total]
        # plt.figure(figsize=(12, 8))
        # plt.plot(avg_reject0, label="Hired", color='blue')
        # plt.fill_between(range(len(avg_reject0)), np.array(avg_reject0) - np.array(np.sqrt(var_reject0)), np.array(avg_reject0) + np.array(np.sqrt(var_reject0)), color='blue', alpha=0.2)
        # plt.plot(avg_reject1, label="Crowdsourced Courier", color='orange')
        # plt.fill_between(range(len(avg_reject1)), np.array(avg_reject1) - np.array(np.sqrt(var_reject1)), np.array(avg_reject1) + np.array(np.sqrt(var_reject1)), color='orange', alpha=0.2)
        # plt.plot(avg_reject, label="Courier", color='green')
        # plt.fill_between(range(len(avg_reject)), np.array(avg_reject) - np.array(np.sqrt(var_reject)), np.array(avg_reject) + np.array(np.sqrt(var_reject)), color='green', alpha=0.2)
        # plt.xlabel('Episodes')
        # plt.ylabel('Average Rejection Number')
        # plt.title("Train: courier's average rejection number")
        # plt.grid(True)
        # plt.legend()
        # plt.savefig('Train_avg_rejection_num.png')
        # plt.close()
        
        price0 = [p[0] for p in order_price_total]
        var_price0 = [p[1] for p in order_price_total]
        price1 = [p[2] for p in order_price_total]
        var_price1 = [p[3] for p in order_price_total]
        price = [p[4] for p in order_price_total]
        var_price = [p[5] for p in order_price_total]
        plt.figure(figsize=(12, 8))
        plt.plot(price0, label="Hired", color='blue')
        plt.fill_between(range(len(price0)), np.array(price0) - np.array(np.sqrt(var_price0)), np.array(price0) + np.array(np.sqrt(var_price0)), color='blue', alpha=0.2)
        plt.plot(price1, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(price1)), np.array(price1) - np.array(np.sqrt(var_price1)), np.array(price1) + np.array(np.sqrt(var_price1)), color='orange', alpha=0.2)
        plt.plot(price, label="Courier", color='green')
        plt.fill_between(range(len(price)), np.array(price) - np.array(np.sqrt(var_price)), np.array(price) + np.array(np.sqrt(var_price)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Price of Order')
        plt.title('Train: average price of order')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_price_of_order.png')
        plt.close()
        
        Hired_income = [i[0] for i in income_total]
        var_Hired_income = [i[1] for i in income_total]
        Crowdsourced_income = [i[2] for i in income_total]
        var_Crowdsourced_income = [i[3] for i in income_total]
        courier_income = [i[4] for i in income_total]
        var_courier_income = [i[5] for i in income_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_income, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_income)), np.array(Hired_income) - np.array(np.sqrt(var_Hired_income)), np.array(Hired_income) + np.array(np.sqrt(var_Hired_income)), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_income, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_income)), np.array(Crowdsourced_income) - np.array(np.sqrt(var_Crowdsourced_income)), np.array(Crowdsourced_income) + np.array(np.sqrt(var_Crowdsourced_income)), color='orange', alpha=0.2)
        plt.plot(courier_income, label="Courier", color='green')
        plt.fill_between(range(len(courier_income)), np.array(courier_income) - np.array(np.sqrt(var_courier_income)), np.array(courier_income) + np.array(np.sqrt(var_courier_income)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Income per Courier')
        plt.title('Train: average income per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_income_per_courier.png')
        plt.close()
        
        platform_total_cost = [i[3] for i in income_total]
        plt.figure(figsize=(12, 8))
        plt.plot(platform_total_cost, label="Platform total cost", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Platform Total Cost')
        plt.title('Train: Platform Total Cost')
        plt.grid(True)
        plt.savefig('Train_platform_total_cost.png')
        plt.close()
        
        Hired_finish = [f[0] for f in courier_finish_num_total]
        var_Hired_finish = [f[1] for f in courier_finish_num_total]
        Crowdsourced_finish = [f[2] for f in courier_finish_num_total]
        var_Crowdsourced_finish = [f[3] for f in courier_finish_num_total]
        courier_finish = [f[4] for f in courier_finish_num_total]
        var_courier_finish = [f[5] for f in courier_finish_num_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_finish, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_finish)), np.array(Hired_finish) - np.array(np.sqrt(var_Hired_finish)), np.array(Hired_finish) + np.array(np.sqrt(var_Hired_finish)), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_finish, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_finish)), np.array(Crowdsourced_finish) - np.array(np.sqrt(var_Crowdsourced_finish)), np.array(Crowdsourced_finish) + np.array(np.sqrt(var_Crowdsourced_finish)), color='orange', alpha=0.2)
        plt.plot(courier_finish, label="Courier", color='green')
        plt.fill_between(range(len(courier_finish)), np.array(courier_finish) - np.array(np.sqrt(var_courier_finish)), np.array(courier_finish) + np.array(np.sqrt(var_courier_finish)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Finish per Courier')
        plt.title('Train: average finish per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_finish_per_courier.png')
        plt.close()
        
        Hired_leisure = [f[0] for f in leisure_time_total]
        var_Hired_leisure = [f[1] for f in leisure_time_total]
        Crowdsourced_leisure = [f[2] for f in leisure_time_total]
        var_Crowdsourced_leisure = [f[3] for f in leisure_time_total]
        courier_leisure = [f[4] for f in leisure_time_total]
        var_courier_leisure = [f[5] for f in leisure_time_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_leisure, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_leisure)), np.array(Hired_leisure) - np.array(np.sqrt(var_Hired_leisure)), np.array(Hired_leisure) + np.array(np.sqrt(var_Hired_leisure)), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_leisure, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_leisure)), np.array(Crowdsourced_leisure) - np.array(np.sqrt(var_Crowdsourced_leisure)), np.array(Crowdsourced_leisure) + np.array(np.sqrt(var_Crowdsourced_leisure)), color='orange', alpha=0.2)
        plt.plot(courier_leisure, label="Courier", color='green')
        plt.fill_between(range(len(courier_leisure)), np.array(courier_leisure) - np.array(np.sqrt(var_courier_leisure)), np.array(courier_leisure) + np.array(np.sqrt(var_courier_leisure)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Leisure Time per Courier')
        plt.title('Train: average leisure time per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_leisure_time_per_courier.png')
        plt.close()
        
        Hired_running = [f[0] for f in running_time_total]
        var_Hired_running = [f[1] for f in running_time_total]
        Crowdsourced_running = [f[2] for f in running_time_total]
        var_Crowdsourced_running = [f[3] for f in running_time_total]
        courier_running = [f[4] for f in running_time_total]
        var_courier_running = [f[5] for f in running_time_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_running, label="Hired", color='blue')
        plt.fill_between(range(len(Hired_running)), np.array(Hired_running) - np.array(np.sqrt(var_Hired_running)), np.array(Hired_running) + np.array(np.sqrt(var_Hired_running)), color='blue', alpha=0.2)
        plt.plot(Crowdsourced_running, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(Crowdsourced_running)), np.array(Crowdsourced_running) - np.array(np.sqrt(var_Crowdsourced_running)), np.array(Crowdsourced_running) + np.array(np.sqrt(var_Crowdsourced_running)), color='orange', alpha=0.2)
        plt.plot(courier_running, label="Courier", color='green')
        plt.fill_between(range(len(courier_running)), np.array(courier_running) - np.array(np.sqrt(var_courier_running)), np.array(courier_running) + np.array(np.sqrt(var_courier_running)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average running Time per Courier')
        plt.title('Train: average running time per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_running_time_per_courier.png')
        plt.close()
        
        order0_late = [l[0] for l in rate_of_late_order]
        order1_late = [l[1] for l in rate_of_late_order]
        order_late = [l[2] for l in rate_of_late_order]
        plt.figure(figsize=(12, 8))
        plt.plot(order0_late, label="Hired", color='blue')
        plt.plot(order1_late, label="Crowdsourced Courier", color='orange')
        plt.plot(order_late, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Rate of Late Orders')
        plt.title('Train: rate of late orders over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_rate_of_late_orders.png')
        plt.close()
        
        order0_ETA = [e[0] for e in rate_of_ETA_usage]
        var_order0_ETA = [e[1] for e in rate_of_ETA_usage]
        order1_ETA = [e[2] for e in rate_of_ETA_usage]
        var_order1_ETA = [e[3] for e in rate_of_ETA_usage]
        order_ETA = [e[4] for e in rate_of_ETA_usage]
        var_order_ETA = [e[5] for e in rate_of_ETA_usage]
        plt.figure(figsize=(12, 8))
        plt.plot(order0_ETA, label="Hired", color='blue')
        plt.fill_between(range(len(order0_ETA)), np.array(order0_ETA) - np.array(np.sqrt(var_order0_ETA)), np.array(order0_ETA) + np.array(np.sqrt(var_order0_ETA)), color='blue', alpha=0.2)
        plt.plot(order1_ETA, label="Crowdsourced Courier", color='orange')
        plt.fill_between(range(len(order1_ETA)), np.array(order1_ETA) - np.array(np.sqrt(var_order1_ETA)), np.array(order1_ETA) + np.array(np.sqrt(var_order1_ETA)), color='orange', alpha=0.2)
        plt.plot(order_ETA, label="Courier", color='green')
        plt.fill_between(range(len(order_ETA)), np.array(order_ETA) - np.array(np.sqrt(var_order_ETA)), np.array(order_ETA) + np.array(np.sqrt(var_order_ETA)), color='green', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Rate of ETA Usage')
        plt.title('Train: rate of ETA usage over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_rate_of_ETA_usage.png')
        plt.close()
        
        #--------------------------
        # draw the Evaluation graph
        
        episodes = list(range(1, len(algo1_eval_distance) + 1))
        x0 = [x[0] for x in algo1_eval_distance]
        x0_var = [x[1] for x in algo1_eval_distance]
        x1 = [x[2] for x in algo1_eval_distance]
        x1_var = [x[3] for x in algo1_eval_distance]
        x2 = [x[4] for x in algo1_eval_distance]
        x2_var = [x[5] for x in algo1_eval_distance]
        x3 = [x[0] for x in algo2_eval_distance]
        x3_var = [x[1] for x in algo2_eval_distance]
        x4 = [x[2] for x in algo2_eval_distance]
        x4_var = [x[3] for x in algo2_eval_distance]
        x5 = [x[4] for x in algo2_eval_distance]
        x5_var = [x[5] for x in algo2_eval_distance]
        x6 = [x[0] for x in algo3_eval_distance]
        x6_var = [x[1] for x in algo3_eval_distance]
        x7 = [x[2] for x in algo3_eval_distance]
        x7_var = [x[3] for x in algo3_eval_distance]
        x8 = [x[4] for x in algo3_eval_distance]
        x8_var = [x[5] for x in algo3_eval_distance]
        x9 = [x[0] for x in algo4_eval_distance]
        x9_var = [x[1] for x in algo4_eval_distance]
        x10 = [x[2] for x in algo4_eval_distance]
        x10_var = [x[3] for x in algo4_eval_distance]
        x11 = [x[4] for x in algo4_eval_distance]
        x11_var = [x[5] for x in algo4_eval_distance]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Total Distance', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Total Distance', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Total Distance', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Total Distance', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Distances')
        plt.title('Eval: Distance Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Distance.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, algo1_eval_episode_rewards, label='Algo1 Reward', color='blue', marker='o')
        plt.plot(episodes, algo2_eval_episode_rewards, label='Algo2 Reward', color='green', marker='o')
        plt.plot(episodes, algo3_eval_episode_rewards, label='Algo3 Reward', color='yellow', marker='o')
        plt.plot(episodes, algo4_eval_episode_rewards, label='Algo4 Reward', color='red', marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Eval: Reward over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Reward.png')
        plt.close()
                
        x0 = [x[0] for x in algo1_eval_speed]
        x0_var = [x[1] for x in algo1_eval_speed]
        x1 = [x[2] for x in algo1_eval_speed]
        x1_var = [x[3] for x in algo1_eval_speed]
        x2 = [x[4] for x in algo1_eval_speed]
        x2_var = [x[5] for x in algo1_eval_speed]
        x3 = [x[0] for x in algo2_eval_speed]
        x3_var = [x[1] for x in algo2_eval_speed]
        x4 = [x[2] for x in algo2_eval_speed]
        x4_var = [x[3] for x in algo2_eval_speed]
        x5 = [x[4] for x in algo2_eval_speed]
        x5_var = [x[5] for x in algo2_eval_speed]
        x6 = [x[0] for x in algo3_eval_speed]
        x6_var = [x[1] for x in algo3_eval_speed]
        x7 = [x[2] for x in algo3_eval_speed]
        x7_var = [x[3] for x in algo3_eval_speed]
        x8 = [x[4] for x in algo3_eval_speed]
        x8_var = [x[5] for x in algo3_eval_speed]
        x9 = [x[0] for x in algo4_eval_distance]
        x9_var = [x[1] for x in algo4_eval_distance]
        x10 = [x[2] for x in algo4_eval_distance]
        x10_var = [x[3] for x in algo4_eval_distance]
        x11 = [x[4] for x in algo4_eval_distance]
        x11_var = [x[5] for x in algo4_eval_distance]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 average speed', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 average speed', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 average speed', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 average speed', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Speed')
        plt.title('Eval: Average Speed Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Average_Speed.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_overspeed_rate], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_overspeed_rate], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_overspeed_rate], label='Algo1 Overspeed Rate', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_overspeed_rate], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_overspeed_rate], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_overspeed_rate], label='Algo2 Overspeed Rate', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_overspeed_rate], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_overspeed_rate], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_overspeed_rate], label='Algo3 Overspeed Rate', color='yellow', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo4_eval_overspeed_rate], label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo4_eval_overspeed_rate], label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo4_eval_overspeed_rate], label='Algo4 Overspeed Rate', color='red', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Overspeed Rate')
        plt.title('Eval: Overspeed Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Overspeed_Rate.png')
        plt.close()
        
        # plt.figure(figsize=(12, 8))
        # plt.plot(episodes, algo1_eval_reject_rate, label='Algo1', color='blue', linestyle='--', marker='o')
        # plt.plot(episodes, algo2_eval_reject_rate, label='Algo2', color='green', linestyle='--', marker='o')
        # plt.plot(episodes, algo3_eval_reject_rate, label='Algo3', color='yellow', linestyle='--', marker='o')
        # plt.plot(episodes, algo4_eval_reject_rate, label='Algo4', color='red', linestyle='--', marker='o')
        # plt.xlabel('Episodes')
        # plt.ylabel('Reject Rate')
        # plt.title('Eval: Reject Rate over Episodes')
        # plt.grid(True)
        # plt.legend()
        # plt.savefig('Eval_Reject_Rate.png')
        # plt.close()

        # x0 = [x[0] for x in algo1_eval_reject]
        # x0_var = [x[1] for x in algo1_eval_reject]
        # x1 = [x[2] for x in algo1_eval_reject]
        # x1_var = [x[3] for x in algo1_eval_reject]
        # x2 = [x[4] for x in algo1_eval_reject]
        # x2_var = [x[5] for x in algo1_eval_reject]
        # x3 = [x[0] for x in algo2_eval_reject]
        # x3_var = [x[1] for x in algo2_eval_reject]
        # x4 = [x[2] for x in algo2_eval_reject]
        # x4_var = [x[3] for x in algo2_eval_reject]
        # x5 = [x[4] for x in algo2_eval_reject]
        # x5_var = [x[5] for x in algo2_eval_reject]
        # x6 = [x[0] for x in algo3_eval_reject]
        # x6_var = [x[1] for x in algo3_eval_reject]
        # x7 = [x[2] for x in algo3_eval_reject]
        # x7_var = [x[3] for x in algo3_eval_reject]
        # x8 = [x[4] for x in algo3_eval_reject]
        # x8_var = [x[5] for x in algo3_eval_reject]
        # x9 = [x[0] for x in algo4_eval_reject]
        # x9_var = [x[1] for x in algo4_eval_reject]
        # x10 = [x[2] for x in algo4_eval_reject]
        # x10_var = [x[3] for x in algo4_eval_reject]
        # x11 = [x[4] for x in algo4_eval_reject]
        # x11_var = [x[5] for x in algo4_eval_reject]
        # plt.figure(figsize=(12, 8))
        # plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        # plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        # plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        # plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        # plt.plot(episodes, x2, label='Algo1 Courier Reject Number', color='blue', linestyle='-', marker='^')
        # plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        # plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        # plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        # plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        # plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        # plt.plot(episodes, x5, label='Algo2 Courier Reject Number', color='green', linestyle='-', marker='^')
        # plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        # plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        # plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        # plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        # plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        # plt.plot(episodes, x8, label='Algo3 Courier Reject Number', color='yellow', linestyle='-', marker='^')
        # plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        # plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        # plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        # plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        # plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        # plt.plot(episodes, x11, label='Algo4 Courier Reject Number', color='red', linestyle='-', marker='^')
        # plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        # plt.xlabel('Episodes')
        # plt.ylabel('Courier Reject Number')
        # plt.title('Eval: Courier Reject Number over Episodes')
        # plt.grid(True)
        # plt.legend()
        # plt.savefig('Eval_Courier_Reject_Number.png')
        # plt.close()
        
        x0 = [x[0] for x in algo1_eval_order_price]
        x0_var = [x[1] for x in algo1_eval_order_price]
        x1 = [x[2] for x in algo1_eval_order_price]
        x1_var = [x[3] for x in algo1_eval_order_price]
        x2 = [x[4] for x in algo1_eval_order_price]
        x2_var = [x[5] for x in algo1_eval_order_price]
        x3 = [x[0] for x in algo2_eval_order_price]
        x3_var = [x[1] for x in algo2_eval_order_price]
        x4 = [x[2] for x in algo2_eval_order_price]
        x4_var = [x[3] for x in algo2_eval_order_price]
        x5 = [x[4] for x in algo2_eval_order_price]
        x5_var = [x[5] for x in algo2_eval_order_price]
        x6 = [x[0] for x in algo3_eval_order_price]
        x6_var = [x[1] for x in algo3_eval_order_price]
        x7 = [x[2] for x in algo3_eval_order_price]
        x7_var = [x[3] for x in algo3_eval_order_price]
        x8 = [x[4] for x in algo3_eval_order_price]
        x8_var = [x[5] for x in algo3_eval_order_price]
        x9 = [x[0] for x in algo4_eval_order_price]
        x9_var = [x[1] for x in algo4_eval_order_price]
        x10 = [x[2] for x in algo4_eval_order_price]
        x10_var = [x[3] for x in algo4_eval_order_price]
        x11 = [x[4] for x in algo4_eval_order_price]
        x11_var = [x[5] for x in algo4_eval_order_price]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Average Order Price', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Average Order Price', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Average Order Price', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Average Order Price', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Order Price')
        plt.title('Eval: Average Order Price over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_order_price.png')
        plt.close()

        x0 = [x[0] for x in algo1_eval_income]
        x0_var = [x[1] for x in algo1_eval_income]
        x1 = [x[2] for x in algo1_eval_income]
        x1_var = [x[3] for x in algo1_eval_income]
        x2 = [x[4] for x in algo1_eval_income]
        x2_var = [x[5] for x in algo1_eval_income]
        x3 = [x[0] for x in algo2_eval_income]
        x3_var = [x[1] for x in algo2_eval_income]
        x4 = [x[2] for x in algo2_eval_income]
        x4_var = [x[3] for x in algo2_eval_income]
        x5 = [x[4] for x in algo2_eval_income]
        x5_var = [x[5] for x in algo2_eval_income]
        x6 = [x[0] for x in algo3_eval_income]
        x6_var = [x[1] for x in algo3_eval_income]
        x7 = [x[2] for x in algo3_eval_income]
        x7_var = [x[3] for x in algo3_eval_income]
        x8 = [x[4] for x in algo3_eval_income]
        x8_var = [x[5] for x in algo3_eval_income]
        x9 = [x[0] for x in algo4_eval_income]
        x9_var = [x[1] for x in algo4_eval_income]
        x10 = [x[2] for x in algo4_eval_income]
        x10_var = [x[3] for x in algo4_eval_income]
        x11 = [x[4] for x in algo4_eval_income]
        x11_var = [x[5] for x in algo4_eval_income]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Courier Average Income', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Courier Average Income', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Courier Average Income', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Courier Average Income', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Courier Average Income')
        plt.title('Eval: Courier Average Income over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_courier_avg_income.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[6] for x in algo1_eval_income], label='Algo1', color='blue', linestyle='-', marker='o')
        plt.plot(episodes, [x[6] for x in algo2_eval_income], label='Algo2', color='green', linestyle='-', marker='o')
        plt.plot(episodes, [x[6] for x in algo3_eval_income], label='Algo3', color='yellow', linestyle='-', marker='o')
        plt.plot(episodes, [x[6] for x in algo4_eval_income], label='Algo4', color='red', linestyle='-', marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Platform Total Cost')
        plt.title('Eval: Platform Total Cost over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_platform_total_cost.png')
        plt.close()
        
        x0 = [x[0] for x in algo1_eval_finish]
        x0_var = [x[1] for x in algo1_eval_finish]
        x1 = [x[2] for x in algo1_eval_finish]
        x1_var = [x[3] for x in algo1_eval_finish]
        x2 = [x[4] for x in algo1_eval_finish]
        x2_var = [x[5] for x in algo1_eval_finish]
        x3 = [x[0] for x in algo2_eval_finish]
        x3_var = [x[1] for x in algo2_eval_finish]
        x4 = [x[2] for x in algo2_eval_finish]
        x4_var = [x[3] for x in algo2_eval_finish]
        x5 = [x[4] for x in algo2_eval_finish]
        x5_var = [x[5] for x in algo2_eval_finish]
        x6 = [x[0] for x in algo3_eval_finish]
        x6_var = [x[1] for x in algo3_eval_finish]
        x7 = [x[2] for x in algo3_eval_finish]
        x7_var = [x[3] for x in algo3_eval_finish]
        x8 = [x[4] for x in algo3_eval_finish]
        x8_var = [x[5] for x in algo3_eval_finish]
        x9 = [x[0] for x in algo4_eval_finish]
        x9_var = [x[1] for x in algo4_eval_finish]
        x10 = [x[2] for x in algo4_eval_finish]
        x10_var = [x[3] for x in algo4_eval_finish]
        x11 = [x[4] for x in algo4_eval_finish]
        x11_var = [x[5] for x in algo4_eval_finish]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Average Finished Orders per Courier', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Average Finished Orders per Courier', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Average Finished Orders per Courier', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Average Finished Orders per Courier', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Finished Orders per Courier')
        plt.title('Eval: Average Finished Orders per Courier over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_order_finished.png')
        plt.close()

        x0 = [x[0] for x in algo1_eval_leisure]
        x0_var = [x[1] for x in algo1_eval_leisure]
        x1 = [x[2] for x in algo1_eval_leisure]
        x1_var = [x[3] for x in algo1_eval_leisure]
        x2 = [x[4] for x in algo1_eval_leisure]
        x2_var = [x[5] for x in algo1_eval_leisure]
        x3 = [x[0] for x in algo2_eval_leisure]
        x3_var = [x[1] for x in algo2_eval_leisure]
        x4 = [x[2] for x in algo2_eval_leisure]
        x4_var = [x[3] for x in algo2_eval_leisure]
        x5 = [x[4] for x in algo2_eval_leisure]
        x5_var = [x[5] for x in algo2_eval_leisure]
        x6 = [x[0] for x in algo3_eval_leisure]
        x6_var = [x[1] for x in algo3_eval_leisure]
        x7 = [x[2] for x in algo3_eval_leisure]
        x7_var = [x[3] for x in algo3_eval_leisure]
        x8 = [x[4] for x in algo3_eval_leisure]
        x8_var = [x[5] for x in algo3_eval_leisure]
        x9 = [x[0] for x in algo4_eval_leisure]
        x9_var = [x[1] for x in algo4_eval_leisure]
        x10 = [x[2] for x in algo4_eval_leisure]
        x10_var = [x[3] for x in algo4_eval_leisure]
        x11 = [x[4] for x in algo4_eval_leisure]
        x11_var = [x[5] for x in algo4_eval_leisure]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Average Leisure Time', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Average Leisure Time', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Average Leisure Time', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Average Leisure Time', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average Leisure Time')
        plt.title('Eval: Average Leisure Time over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_leisure_time.png')
        plt.close()
        
        x0 = [x[0] for x in algo1_eval_running]
        x0_var = [x[1] for x in algo1_eval_running]
        x1 = [x[2] for x in algo1_eval_running]
        x1_var = [x[3] for x in algo1_eval_running]
        x2 = [x[4] for x in algo1_eval_running]
        x2_var = [x[5] for x in algo1_eval_running]
        x3 = [x[0] for x in algo2_eval_running]
        x3_var = [x[1] for x in algo2_eval_running]
        x4 = [x[2] for x in algo2_eval_running]
        x4_var = [x[3] for x in algo2_eval_running]
        x5 = [x[4] for x in algo2_eval_running]
        x5_var = [x[5] for x in algo2_eval_running]
        x6 = [x[0] for x in algo3_eval_running]
        x6_var = [x[1] for x in algo3_eval_running]
        x7 = [x[2] for x in algo3_eval_running]
        x7_var = [x[3] for x in algo3_eval_running]
        x8 = [x[4] for x in algo3_eval_running]
        x8_var = [x[5] for x in algo3_eval_running]
        x9 = [x[0] for x in algo4_eval_running]
        x9_var = [x[1] for x in algo4_eval_running]
        x10 = [x[2] for x in algo4_eval_running]
        x10_var = [x[3] for x in algo4_eval_running]
        x11 = [x[4] for x in algo4_eval_running]
        x11_var = [x[5] for x in algo4_eval_running]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 Average running Time', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 Average running Time', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 Average running Time', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 Average running Time', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('Average running Time')
        plt.title('Eval: Average running Time over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_running_time.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_rate_of_late_order], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_rate_of_late_order], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_rate_of_late_order], label='Algo1 Late Order Rate', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_rate_of_late_order], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_rate_of_late_order], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_rate_of_late_order], label='Algo2 Late Order Rate', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_rate_of_late_order], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_rate_of_late_order], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_rate_of_late_order], label='Algo3 Late Order Rate', color='yellow', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo4_rate_of_late_order], label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo4_rate_of_late_order], label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo4_rate_of_late_order], label='Algo4 Late Order Rate', color='red', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Late Order Rate')
        plt.title('Eval: Late Order Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Late_Order_Rate.png')
        plt.close()
        
        x0 = [x[0] for x in algo1_rate_of_ETA_usage]
        x0_var = [x[1] for x in algo1_rate_of_ETA_usage]
        x1 = [x[2] for x in algo1_rate_of_ETA_usage]
        x1_var = [x[3] for x in algo1_rate_of_ETA_usage]
        x2 = [x[4] for x in algo1_rate_of_ETA_usage]
        x2_var = [x[5] for x in algo1_rate_of_ETA_usage]
        x3 = [x[0] for x in algo2_rate_of_ETA_usage]
        x3_var = [x[1] for x in algo2_rate_of_ETA_usage]
        x4 = [x[2] for x in algo2_rate_of_ETA_usage]
        x4_var = [x[3] for x in algo2_rate_of_ETA_usage]
        x5 = [x[4] for x in algo2_rate_of_ETA_usage]
        x5_var = [x[5] for x in algo2_rate_of_ETA_usage]
        x6 = [x[0] for x in algo3_rate_of_ETA_usage]
        x6_var = [x[1] for x in algo3_rate_of_ETA_usage]
        x7 = [x[2] for x in algo3_rate_of_ETA_usage]
        x7_var = [x[3] for x in algo3_rate_of_ETA_usage]
        x8 = [x[4] for x in algo3_rate_of_ETA_usage]
        x8_var = [x[5] for x in algo3_rate_of_ETA_usage]
        x9 = [x[0] for x in algo4_rate_of_ETA_usage]
        x9_var = [x[1] for x in algo4_rate_of_ETA_usage]
        x10 = [x[2] for x in algo4_rate_of_ETA_usage]
        x10_var = [x[3] for x in algo4_rate_of_ETA_usage]
        x11 = [x[4] for x in algo4_rate_of_ETA_usage]
        x11_var = [x[5] for x in algo4_rate_of_ETA_usage]
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x0, label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x0) - np.array(np.sqrt(x0_var)), np.array(x0) + np.array(np.sqrt(x0_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x1, label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x1) - np.array(np.sqrt(x1_var)), np.array(x1) + np.array(np.sqrt(x1_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x2, label='Algo1 ETA Usage', color='blue', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x2) - np.array(np.sqrt(x2_var)), np.array(x2) + np.array(np.sqrt(x2_var)), color='blue', alpha=0.2)
        plt.plot(episodes, x3, label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x3) - np.array(np.sqrt(x3_var)), np.array(x3) + np.array(np.sqrt(x3_var)), color='green', alpha=0.2)
        plt.plot(episodes, x4, label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x4) - np.array(np.sqrt(x4_var)), np.array(x4) + np.array(np.sqrt(x4_var)), color='green', alpha=0.2)
        plt.plot(episodes, x5, label='Algo2 ETA Usage', color='green', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x5) - np.array(np.sqrt(x5_var)), np.array(x5) + np.array(np.sqrt(x5_var)), color='green', alpha=0.2)
        plt.plot(episodes, x6, label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x6) - np.array(np.sqrt(x6_var)), np.array(x6) + np.array(np.sqrt(x6_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x7, label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x7) - np.array(np.sqrt(x7_var)), np.array(x7) + np.array(np.sqrt(x7_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x8, label='Algo3 ETA Usage', color='yellow', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x8) - np.array(np.sqrt(x8_var)), np.array(x8) + np.array(np.sqrt(x8_var)), color='yellow', alpha=0.2)
        plt.plot(episodes, x9, label='Algo4 Hired', color='red', linestyle='--', marker='o')
        plt.fill_between(episodes, np.array(x9) - np.array(np.sqrt(x9_var)), np.array(x9) + np.array(np.sqrt(x9_var)), color='red', alpha=0.2)
        plt.plot(episodes, x10, label='Algo4 Crowdsourced', color='red', linestyle='-.', marker='s')
        plt.fill_between(episodes, np.array(x10) - np.array(np.sqrt(x10_var)), np.array(x10) + np.array(np.sqrt(x10_var)), color='red', alpha=0.2)
        plt.plot(episodes, x11, label='Algo4 ETA Usage', color='red', linestyle='-', marker='^')
        plt.fill_between(episodes, np.array(x11) - np.array(np.sqrt(x11_var)), np.array(x11) + np.array(np.sqrt(x11_var)), color='red', alpha=0.2)
        plt.xlabel('Episodes')
        plt.ylabel('ETA Usage Rate')
        plt.title('Eval: ETA Usage Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_ETA_Usage_Rate.png')
        plt.close()

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
        
        # eval_obs = self.eval_envs.reset(4)
        eval_obs = self.eval_envs.reset(5)
        algo1_Hired_num = 0
        algo1_Crowdsourced_num = 0
        algo1_Crowdsourced_on = 0
        algo2_Hired_num = 0
        algo2_Crowdsourced_num = 0
        algo2_Crowdsourced_on = 0
        algo3_Hired_num = 0
        algo3_Crowdsourced_num = 0
        algo3_Crowdsourced_on = 0
        algo4_Hired_num = 0
        algo4_Crowdsourced_num = 0
        algo4_Crowdsourced_on = 0
        
        algo1_eval_episode_rewards_sum = 0
        algo2_eval_episode_rewards_sum = 0
        algo3_eval_episode_rewards_sum = 0
        algo4_eval_episode_rewards_sum = 0
        
        algo1_Hired_distance_per_episode = []
        algo1_Crowdsourced_distance_per_episode = []
        algo2_Hired_distance_per_episode = []
        algo2_Crowdsourced_distance_per_episode = []
        algo3_Hired_distance_per_episode = []
        algo3_Crowdsourced_distance_per_episode = []
        algo4_Hired_distance_per_episode = []
        algo4_Crowdsourced_distance_per_episode = []

        algo1_count_overspeed0 = 0
        algo1_count_overspeed1 = 0
        algo2_count_overspeed0 = 0
        algo2_count_overspeed1 = 0
        algo3_count_overspeed0 = 0
        algo3_count_overspeed1 = 0
        algo4_count_overspeed0 = 0
        algo4_count_overspeed1 = 0
        
        algo1_num_active_couriers0 = 0
        algo1_num_active_couriers1 = 0
        algo2_num_active_couriers0 = 0
        algo2_num_active_couriers1 = 0
        algo3_num_active_couriers0 = 0
        algo3_num_active_couriers1 = 0
        algo4_num_active_couriers0 = 0
        algo4_num_active_couriers1 = 0
        
        # algo1_count_reject_orders = 0
        # algo1_max_reject_num = 0
        # algo2_count_reject_orders = 0
        # algo2_max_reject_num = 0
        # algo3_count_reject_orders = 0
        # algo3_max_reject_num = 0
        # algo4_count_reject_orders = 0
        # algo4_max_reject_num = 0

        algo1_late_orders0 = 0
        algo1_late_orders1 = 0
        algo2_late_orders0 = 0
        algo2_late_orders1 = 0
        algo3_late_orders0 = 0
        algo3_late_orders1 = 0
        algo4_late_orders0 = 0
        algo4_late_orders1 = 0
        
        algo1_ETA_usage0 = []
        algo1_ETA_usage1 = []
        algo2_ETA_usage0 = []
        algo2_ETA_usage1 = []
        algo3_ETA_usage0 = []
        algo3_ETA_usage1 = []
        algo4_ETA_usage0 = []
        algo4_ETA_usage1 = []
        
        algo1_count_dropped_orders0 = 0
        algo1_count_dropped_orders1 = 0
        algo2_count_dropped_orders0 = 0
        algo2_count_dropped_orders1 = 0
        algo3_count_dropped_orders0 = 0
        algo3_count_dropped_orders1 = 0
        algo4_count_dropped_orders0 = 0
        algo4_count_dropped_orders1 = 0
        
        algo1_order0_price = []
        algo1_order1_price = []
        algo1_order0_num = 0
        algo1_order1_num = 0
        algo1_order_wait = 0
        algo2_order0_price = []
        algo2_order1_price = []
        algo2_order0_num = 0
        algo2_order1_num = 0
        algo2_order_wait = 0
        algo3_order0_price = []
        algo3_order1_price = []
        algo3_order0_num = 0
        algo3_order1_num = 0
        algo3_order_wait = 0
        algo4_order0_price = []
        algo4_order1_price = []
        algo4_order0_num = 0
        algo4_order1_num = 0
        algo4_order_wait = 0
        
        platform_cost1 = 0
        platform_cost2 = 0
        platform_cost3 = 0
        platform_cost4 = 0
        
        # algo1_Hired_reject_num = []
        # algo1_Crowdsourced_reject_num = []
        # algo2_Hired_reject_num = []
        # algo2_Crowdsourced_reject_num = []
        # algo3_Hired_reject_num = []
        # algo3_Crowdsourced_reject_num = []
        # algo4_Hired_reject_num = []
        # algo4_Crowdsourced_reject_num = []
        
        algo1_Hired_finish_num = []
        algo1_Crowdsourced_finish_num = []
        algo2_Hired_finish_num = []
        algo2_Crowdsourced_finish_num = []
        algo3_Hired_finish_num = []
        algo3_Crowdsourced_finish_num = []
        algo4_Hired_finish_num = []
        algo4_Crowdsourced_finish_num = []
        
        algo1_Hired_leisure_time = []
        algo1_Crowdsourced_leisure_time = []
        algo2_Hired_leisure_time = []
        algo2_Crowdsourced_leisure_time = []
        algo3_Hired_leisure_time = []
        algo3_Crowdsourced_leisure_time = []
        algo4_Hired_leisure_time = []
        algo4_Crowdsourced_leisure_time = []
        
        algo1_Hired_running_time = []
        algo1_Crowdsourced_running_time = []
        algo2_Hired_running_time = []
        algo2_Crowdsourced_running_time = []
        algo3_Hired_running_time = []
        algo3_Crowdsourced_running_time = []
        algo4_Hired_running_time = []
        algo4_Crowdsourced_running_time = []
        
        algo1_Hired_avg_speed = []
        algo1_Crowdsourced_avg_speed = []
        algo2_Hired_avg_speed = []
        algo2_Crowdsourced_avg_speed = []
        algo3_Hired_avg_speed = []
        algo3_Crowdsourced_avg_speed = []
        algo4_Hired_avg_speed = []
        algo4_Crowdsourced_avg_speed = []
        
        algo1_Hired_income = []
        algo1_Crowdsourced_income = []
        algo2_Hired_income = []
        algo2_Crowdsourced_income = []
        algo3_Hired_income = []
        algo3_Crowdsourced_income = []
        algo4_Hired_income = []
        algo4_Crowdsourced_income = []
        
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

        for eval_step in range(self.eval_episodes_length):
            if self.eval_num_agents > self.num_agents:
                print(self.eval_num_agents)
                print(self.num_agents)
                break
            
            print("-"*25)
            print(f"THIS IS EVAL STEP {eval_step}")

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
                
                self.log_env(1, eval_step, i, eval=True)
                
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
            algo3_eval_episode_rewards_sum += sum(eval_rewards[2])
            algo4_eval_episode_rewards_sum += sum(eval_rewards[3])

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
                            if c.courier_type == 0:
                                algo1_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo1_count_overspeed0 += 1
                            else:
                                algo1_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo1_count_overspeed1 += 1
                elif i == 1:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                algo2_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo2_count_overspeed0 += 1
                            else:
                                algo2_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo2_count_overspeed1 += 1
                elif i == 2:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                algo3_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo3_count_overspeed0 += 1
                            else:
                                algo3_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo3_count_overspeed1 += 1
                else:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                algo4_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo4_count_overspeed0 += 1
                            else:
                                algo4_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo4_count_overspeed1 += 1
                    
                                
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
            if i == 0:
                platform_cost1 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo1_Hired_num += 1
                        algo1_Hired_distance_per_episode.append(c.travel_distance)
                        # algo1_Hired_reject_num.append(c.reject_order_num)
                        algo1_Hired_finish_num.append(c.finish_order_num)
                        algo1_Hired_leisure_time.append(c.total_leisure_time)
                        algo1_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo1_Hired_avg_speed.append(c.avg_speed)
                        algo1_Hired_income.append(c.income)
                    else:
                        algo1_Crowdsourced_num += 1
                        algo1_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        # algo1_Crowdsourced_reject_num.append(c.reject_order_num)
                        algo1_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo1_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo1_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo1_Crowdsourced_avg_speed.append(c.avg_speed)
                        algo1_Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            algo1_Crowdsourced_on += 1

                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo1_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo1_late_orders0 += 1
                            else:
                                algo1_ETA_usage0.append(o.ETA_usage)
                        else:
                            algo1_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo1_late_orders1 += 1
                            else:
                                algo1_ETA_usage1.append(o.ETA_usage)
                            
                    if o.reject_count > 0:
                        algo1_count_reject_orders += 1
                        if algo1_max_reject_num <= o.reject_count:
                            algo1_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo1_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo1_order0_price.append(o.price)
                            algo1_order0_num += 1
                        else:
                            algo1_order1_price.append(o.price)
                            algo1_order1_num += 1             
                    
            elif i == 1:
                platform_cost2 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo2_Hired_num += 1
                        algo2_Hired_distance_per_episode.append(c.travel_distance)
                        # algo2_Hired_reject_num.append(c.reject_order_num)
                        algo2_Hired_finish_num.append(c.finish_order_num)
                        algo2_Hired_leisure_time.append(c.total_leisure_time)
                        algo2_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo2_Hired_avg_speed.append(c.avg_speed)
                        algo2_Hired_income.append(c.income)
                    else:
                        algo2_Crowdsourced_num += 1
                        algo2_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        # algo2_Crowdsourced_reject_num.append(c.reject_order_num)
                        algo2_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo2_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo2_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo2_Crowdsourced_avg_speed.append(c.avg_speed)
                        algo2_Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            algo2_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo2_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo2_late_orders0 += 1
                            else:
                                algo2_ETA_usage0.append(o.ETA_usage)
                        else:
                            algo2_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo2_late_orders1 += 1
                            else:
                                algo2_ETA_usage1.append(o.ETA_usage)
                            
                    if o.reject_count > 0:
                        algo2_count_reject_orders += 1
                        if algo2_max_reject_num <= o.reject_count:
                            algo2_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo2_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo2_order0_price.append(o.price)
                            algo2_order0_num += 1
                        else:
                            algo2_order1_price.append(o.price)
                            algo2_order1_num += 1
            elif i == 2:
                platform_cost3 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo3_Hired_num += 1
                        algo3_Hired_distance_per_episode.append(c.travel_distance)
                        # algo3_Hired_reject_num.append(c.reject_order_num)
                        algo3_Hired_finish_num.append(c.finish_order_num)
                        algo3_Hired_leisure_time.append(c.total_leisure_time)
                        algo3_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo3_Hired_avg_speed.append(c.avg_speed)
                        algo3_Hired_income.append(c.income)
                    else:
                        algo3_Crowdsourced_num += 1
                        algo3_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        # algo3_Crowdsourced_reject_num.append(c.reject_order_num)
                        algo3_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo3_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo3_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo3_Crowdsourced_avg_speed.append(c.avg_speed)
                        algo3_Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            algo3_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo3_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo3_late_orders0 += 1
                            else:
                                algo3_ETA_usage0.append(o.ETA_usage)
                        else:
                            algo3_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo3_late_orders1 += 1
                            else:
                                algo3_ETA_usage1.append(o.ETA_usage)
                            
                    if o.reject_count > 0:
                        algo3_count_reject_orders += 1
                        if algo3_max_reject_num <= o.reject_count:
                            algo3_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo3_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo3_order0_price.append(o.price)
                            algo3_order0_num += 1
                        else:
                            algo3_order1_price.append(o.price)
                            algo3_order1_num += 1  
            else:
                platform_cost4 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo4_Hired_num += 1
                        algo4_Hired_distance_per_episode.append(c.travel_distance)
                        # algo4_Hired_reject_num.append(c.reject_order_num)
                        algo4_Hired_finish_num.append(c.finish_order_num)
                        algo4_Hired_leisure_time.append(c.total_leisure_time)
                        algo4_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo4_Hired_avg_speed.append(c.avg_speed)
                        algo4_Hired_income.append(c.income)
                    else:
                        algo4_Crowdsourced_num += 1
                        algo4_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        # algo4_Crowdsourced_reject_num.append(c.reject_order_num)
                        algo4_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo4_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo4_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo4_Crowdsourced_avg_speed.append(c.avg_speed)
                        algo4_Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            algo4_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo4_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo4_late_orders0 += 1
                            else:
                                algo4_ETA_usage0.append(o.ETA_usage)
                        else:
                            algo4_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo4_late_orders1 += 1
                            else:
                                algo4_ETA_usage1.append(o.ETA_usage)
                            
                    # if o.reject_count > 0:
                    #     algo4_count_reject_orders += 1
                    #     if algo4_max_reject_num <= o.reject_count:
                    #         algo4_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo4_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo4_order0_price.append(o.price)
                            algo4_order0_num += 1
                        else:
                            algo4_order1_price.append(o.price)
                            algo4_order1_num += 1   
                            
        print(f"\nIn Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} on, {algo1_order0_num} Order0, {algo1_order1_num} Order1, {algo1_order_wait} ({round(100 * algo1_order_wait / (algo1_order_wait + algo1_order0_num + algo1_order1_num), 2)}%) Orders waiting to be paired")
        print(f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} on, {algo2_order0_num} Order0, {algo2_order1_num} Order1, {algo2_order_wait} ({round(100 * algo2_order_wait / (algo2_order_wait + algo2_order0_num + algo2_order1_num), 2)}%) Orders waiting to be paired")  
        print(f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} on, {algo3_order0_num} Order0, {algo3_order1_num} Order1, {algo3_order_wait} ({round(100 * algo3_order_wait / (algo3_order_wait + algo3_order0_num + algo3_order1_num), 2)}%) Orders waiting to be paired")       
        print(f"In Algo4 there are {algo4_Hired_num} Hired, {algo4_Crowdsourced_num} Crowdsourced with {algo4_Crowdsourced_on} on, {algo4_order0_num} Order0, {algo4_order1_num} Order1, {algo4_order_wait} ({round(100 * algo4_order_wait / (algo4_order_wait + algo4_order0_num + algo4_order1_num), 2)}%) Orders waiting to be paired")
        
        # -----------------------
        # Reward
        print(f"Total Reward for Evaluation Between Algos:\nAlgo1: {round(algo1_eval_episode_rewards_sum, 2)}\nAlgo2: {round(algo2_eval_episode_rewards_sum, 2)}\nAlgo3: {round(algo3_eval_episode_rewards_sum, 2)}\nAlgo4: {round(algo4_eval_episode_rewards_sum, 2)}")
        self.writter.add_scalar('Eval Reward/Algo1', algo1_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo2', algo2_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo3', algo3_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo4', algo4_eval_episode_rewards_sum, self.eval_num)

        # -----------------------
        # Distance
        algo1_distance0 = round(np.mean(algo1_Hired_distance_per_episode) / 1000, 2)
        algo1_var0_distance = round(np.var(algo1_Hired_distance_per_episode) / 1000000, 2)
        algo1_distance1 = round(np.mean(algo1_Crowdsourced_distance_per_episode) / 1000, 2)
        algo1_var1_distance = round(np.var(algo1_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo1_distance = round(np.mean(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000, 2)
        algo1_var_distance = round(np.var(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000000, 2)
        
        algo2_distance0 = round(np.mean(algo2_Hired_distance_per_episode) / 1000, 2)
        algo2_var0_distance = round(np.var(algo2_Hired_distance_per_episode) / 1000000, 2)
        algo2_distance1 = round(np.mean(algo2_Crowdsourced_distance_per_episode) / 1000, 2)
        algo2_var1_distance = round(np.var(algo2_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo2_distance = round(np.mean(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000, 2)
        algo2_var_distance = round(np.var(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000000, 2)

        algo3_distance0 = round(np.mean(algo3_Hired_distance_per_episode) / 1000, 2)
        algo3_var0_distance = round(np.var(algo3_Hired_distance_per_episode) / 1000000, 2)
        algo3_distance1 = round(np.mean(algo3_Crowdsourced_distance_per_episode) / 1000, 2)
        algo3_var1_distance = round(np.var(algo3_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo3_distance = round(np.mean(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000, 2)
        algo3_var_distance = round(np.var(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000000, 2)

        algo4_distance0 = round(np.mean(algo4_Hired_distance_per_episode) / 1000, 2)
        algo4_var0_distance = round(np.var(algo4_Hired_distance_per_episode) / 1000000, 2)
        algo4_distance1 = round(np.mean(algo4_Crowdsourced_distance_per_episode) / 1000, 2)
        algo4_var1_distance = round(np.var(algo4_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo4_distance = round(np.mean(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000, 2)
        algo4_var_distance = round(np.var(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000000, 2)

        print("Average Travel Distance and Var per Courier Between Algos:")
        print(f"Algo1: Hired - {algo1_distance0} km (Var: {algo1_var0_distance}), Crowdsourced - {algo1_distance1} km (Var: {algo1_var1_distance}), Total - {algo1_distance} km (Var: {algo1_var_distance})")
        print(f"Algo2: Hired - {algo2_distance0} km (Var: {algo2_var0_distance}), Crowdsourced - {algo2_distance1} km (Var: {algo2_var1_distance}), Total - {algo2_distance} km (Var: {algo2_var_distance})")
        print(f"Algo3: Hired - {algo3_distance0} km (Var: {algo3_var0_distance}), Crowdsourced - {algo3_distance1} km (Var: {algo3_var1_distance}), Total - {algo3_distance} km (Var: {algo3_var_distance})")
        print(f"Algo4: Hired - {algo4_distance0} km (Var: {algo4_var0_distance}), Crowdsourced - {algo4_distance1} km (Var: {algo4_var1_distance}), Total - {algo4_distance} km (Var: {algo4_var_distance})")
        
        self.writter.add_scalar('Eval Travel Distance/Algo1 Hired', algo1_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Crowdsourced', algo1_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Total', algo1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Hired Var', algo1_var0_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Crowdsourced Var', algo1_var1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Total Var', algo1_var_distance, self.eval_num)

        self.writter.add_scalar('Eval Travel Distance/Algo2 Hired', algo2_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Crowdsourced', algo2_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Total', algo2_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Hired Var', algo2_var0_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Crowdsourced Var', algo2_var1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Total Var', algo2_var_distance, self.eval_num)

        self.writter.add_scalar('Eval Travel Distance/Algo3 Hired', algo3_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Crowdsourced', algo3_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Total', algo3_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Hired Var', algo3_var0_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Crowdsourced Var', algo3_var1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Total Var', algo3_var_distance, self.eval_num)

        self.writter.add_scalar('Eval Travel Distance/Algo4 Hired', algo4_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo4 Crowdsourced', algo4_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo4 Total', algo4_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo4 Hired Var', algo4_var0_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo4 Crowdsourced Var', algo4_var1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo4 Total Var', algo4_var_distance, self.eval_num)
        
        # -----------------------
        # Average Speed
        algo1_avg0_speed = round(np.mean(algo1_Hired_avg_speed), 2)
        algo1_var0_speed = round(np.var(algo1_Hired_avg_speed), 2)
        algo1_avg1_speed = round(np.mean(algo1_Crowdsourced_avg_speed), 2)
        algo1_var1_speed = round(np.var(algo1_Crowdsourced_avg_speed), 2)
        algo1_avg_speed = round(np.mean(algo1_Hired_avg_speed + algo1_Crowdsourced_avg_speed), 2)
        algo1_var_speed = round(np.var(algo1_Hired_avg_speed + algo1_Crowdsourced_avg_speed), 2)

        algo2_avg0_speed = round(np.mean(algo2_Hired_avg_speed), 2)
        algo2_var0_speed = round(np.var(algo2_Hired_avg_speed), 2)
        algo2_avg1_speed = round(np.mean(algo2_Crowdsourced_avg_speed), 2)
        algo2_var1_speed = round(np.var(algo2_Crowdsourced_avg_speed), 2)
        algo2_avg_speed = round(np.mean(algo2_Hired_avg_speed + algo2_Crowdsourced_avg_speed), 2)
        algo2_var_speed = round(np.var(algo2_Hired_avg_speed + algo2_Crowdsourced_avg_speed), 2)

        algo3_avg0_speed = round(np.mean(algo3_Hired_avg_speed), 2)
        algo3_var0_speed = round(np.var(algo3_Hired_avg_speed), 2)
        algo3_avg1_speed = round(np.mean(algo3_Crowdsourced_avg_speed), 2)
        algo3_var1_speed = round(np.var(algo3_Crowdsourced_avg_speed), 2)
        algo3_avg_speed = round(np.mean(algo3_Hired_avg_speed + algo3_Crowdsourced_avg_speed), 2)
        algo3_var_speed = round(np.var(algo3_Hired_avg_speed + algo3_Crowdsourced_avg_speed), 2)

        algo4_avg0_speed = round(np.mean(algo4_Hired_avg_speed), 2)
        algo4_var0_speed = round(np.var(algo4_Hired_avg_speed), 2)
        algo4_avg1_speed = round(np.mean(algo4_Crowdsourced_avg_speed), 2)
        algo4_var1_speed = round(np.var(algo4_Crowdsourced_avg_speed), 2)
        algo4_avg_speed = round(np.mean(algo4_Hired_avg_speed + algo4_Crowdsourced_avg_speed), 2)
        algo4_var_speed = round(np.var(algo4_Hired_avg_speed + algo4_Crowdsourced_avg_speed), 2)

        print("Average Speed and Variance per Courier Between Algos:")
        print(f"Algo1: Hired average speed is {algo1_avg0_speed} m/s (Var: {algo1_var0_speed}), Crowdsourced average speed is {algo1_avg1_speed} m/s (Var: {algo1_var1_speed}), Total average speed is {algo1_avg_speed} m/s (Var: {algo1_var_speed})")
        print(f"Algo2: Hired average speed is {algo2_avg0_speed} m/s (Var: {algo2_var0_speed}), Crowdsourced average speed is {algo2_avg1_speed} m/s (Var: {algo2_var1_speed}), Total average speed is {algo2_avg_speed} m/s (Var: {algo2_var_speed})")
        print(f"Algo3: Hired average speed is {algo3_avg0_speed} m/s (Var: {algo3_var0_speed}), Crowdsourced average speed is {algo3_avg1_speed} m/s (Var: {algo3_var1_speed}), Total average speed is {algo3_avg_speed} m/s (Var: {algo3_var_speed})")
        print(f"Algo4: Hired average speed is {algo4_avg0_speed} m/s (Var: {algo4_var0_speed}), Crowdsourced average speed is {algo4_avg1_speed} m/s (Var: {algo4_var1_speed}), Total average speed is {algo4_avg_speed} m/s (Var: {algo4_var_speed})")

        self.writter.add_scalar('Eval Average Speed/Algo1 Total', algo1_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Hired', algo1_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Crowdsourced', algo1_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Total Var', algo1_var_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Hired Var', algo1_var0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Crowdsourced Var', algo1_var1_speed, self.eval_num)

        self.writter.add_scalar('Eval Average Speed/Algo2 Total', algo2_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Hired', algo2_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Crowdsourced', algo2_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Total Var', algo2_var_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Hired Var', algo2_var0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Crowdsourced Var', algo2_var1_speed, self.eval_num)

        self.writter.add_scalar('Eval Average Speed/Algo3 Total', algo3_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Hired', algo3_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Crowdsourced', algo3_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Total Var', algo3_var_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Hired Var', algo3_var0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Crowdsourced Var', algo3_var1_speed, self.eval_num)

        self.writter.add_scalar('Eval Average Speed/Algo4 Total', algo4_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo4 Hired', algo4_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo4 Crowdsourced', algo4_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo4 Total Var', algo4_var_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo4 Hired Var', algo4_var0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo4 Crowdsourced Var', algo4_var1_speed, self.eval_num)

        # -----------------------
        # Overspeed
        algo1_overspeed0 = round(algo1_count_overspeed0 / algo1_num_active_couriers0, 2)
        algo1_overspeed1 = round(algo1_count_overspeed1 / algo1_num_active_couriers1, 2)
        algo1_overspeed = round((algo1_count_overspeed0 + algo1_count_overspeed1) / (algo1_num_active_couriers0 + algo1_num_active_couriers1), 2)
        
        algo2_overspeed0 = round(algo2_count_overspeed0 / algo2_num_active_couriers0, 2)
        algo2_overspeed1 = round(algo2_count_overspeed1 / algo2_num_active_couriers1, 2)
        algo2_overspeed = round((algo2_count_overspeed0 + algo2_count_overspeed1) / (algo2_num_active_couriers0 + algo2_num_active_couriers1), 2)
        
        algo3_overspeed0 = round(algo3_count_overspeed0 / algo3_num_active_couriers0, 2)
        algo3_overspeed1 = round(algo3_count_overspeed1 / algo3_num_active_couriers1, 2)
        algo3_overspeed = round((algo3_count_overspeed0 + algo3_count_overspeed1) / (algo3_num_active_couriers0 + algo3_num_active_couriers1), 2)

        algo4_overspeed0 = round(algo4_count_overspeed0 / algo4_num_active_couriers0, 2)
        algo4_overspeed1 = round(algo4_count_overspeed1 / algo4_num_active_couriers1, 2)
        algo4_overspeed = round((algo4_count_overspeed0 + algo4_count_overspeed1) / (algo4_num_active_couriers0 + algo4_num_active_couriers1), 2)

        print("Rate of Overspeed for Evaluation Between Algos:")
        print(f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}")
        print(f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}")
        print(f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}")
        print(f"Algo4: Hired - {algo4_overspeed0}, Crowdsourced - {algo4_overspeed1}, Total rate - {algo4_overspeed}")
        
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Total', algo1_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Hired', algo1_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Crowdsourced', algo1_overspeed1, self.eval_num)
        
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Total', algo2_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Hired', algo2_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Crowdsourced', algo2_overspeed1, self.eval_num)
        
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Total', algo3_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Hired', algo3_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Crowdsourced', algo3_overspeed1, self.eval_num)
        
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Total', algo4_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Hired', algo4_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Crowdsourced', algo4_overspeed1, self.eval_num)
        
        # -----------------------
        # Order Reject Rate
        # algo1_reject_rate_per_episode = round(algo1_count_reject_orders / len(self.eval_envs.envs_discrete[0].orders), 2)
        # algo2_reject_rate_per_episode = round(algo2_count_reject_orders / len(self.eval_envs.envs_discrete[1].orders), 2)
        # algo3_reject_rate_per_episode = round(algo3_count_reject_orders / len(self.eval_envs.envs_discrete[2].orders), 2)
        
        # algo4_reject_rate_per_episode = round(algo4_count_reject_orders / len(self.eval_envs.envs_discrete[3].orders), 2)

        # print("Reject Rate for Evaluation Between Algos:")
        # print(f"Algo1: {algo1_reject_rate_per_episode} and the order is rejected by {algo1_max_reject_num} times at most")
        # print(f"Algo2: {algo2_reject_rate_per_episode} and the order is rejected by {algo2_max_reject_num} times at most")
        # print(f"Algo3: {algo3_reject_rate_per_episode} and the order is rejected by {algo3_max_reject_num} times at most")
        
        # print(f"Algo4: {algo4_reject_rate_per_episode} and the order is rejected by {algo4_max_reject_num} times at most")

        # self.writter.add_scalar('Eval Reject rate/Algo1', algo1_reject_rate_per_episode, self.eval_num)
        # self.writter.add_scalar('Eval Reject rate/Algo2', algo2_reject_rate_per_episode, self.eval_num)
        # self.writter.add_scalar('Eval Reject rate/Algo3', algo3_reject_rate_per_episode, self.eval_num)
        # self.writter.add_scalar('Eval Reject rate/Algo4', algo4_reject_rate_per_episode, self.eval_num)
        
        # -----------------------
        # Average Courier Reject Number
        # algo1_reject0 = round(np.mean(algo1_Hired_reject_num), 2)
        # algo1_var0_reject = round(np.var(algo1_Hired_reject_num), 2)
        # algo1_reject1 = round(np.mean(algo1_Crowdsourced_reject_num), 2)
        # algo1_var1_reject = round(np.var(algo1_Crowdsourced_reject_num), 2)
        # algo1_reject = round(np.mean(algo1_Hired_reject_num + algo1_Crowdsourced_reject_num), 2)
        # algo1_var_reject = round(np.var(algo1_Hired_reject_num + algo1_Crowdsourced_reject_num), 2)

        # algo2_reject0 = round(np.mean(algo2_Hired_reject_num), 2)
        # algo2_var0_reject = round(np.var(algo2_Hired_reject_num), 2)
        # algo2_reject1 = round(np.mean(algo2_Crowdsourced_reject_num), 2)
        # algo2_var1_reject = round(np.var(algo2_Crowdsourced_reject_num), 2)
        # algo2_reject = round(np.mean(algo2_Hired_reject_num + algo2_Crowdsourced_reject_num), 2)
        # algo2_var_reject = round(np.var(algo2_Hired_reject_num + algo2_Crowdsourced_reject_num), 2)

        # algo3_reject0 = round(np.mean(algo3_Hired_reject_num), 2)
        # algo3_var0_reject = round(np.var(algo3_Hired_reject_num), 2)
        # algo3_reject1 = round(np.mean(algo3_Crowdsourced_reject_num), 2)
        # algo3_var1_reject = round(np.var(algo3_Crowdsourced_reject_num), 2)
        # algo3_reject = round(np.mean(algo3_Hired_reject_num + algo3_Crowdsourced_reject_num), 2)
        # algo3_var_reject = round(np.var(algo3_Hired_reject_num + algo3_Crowdsourced_reject_num), 2)

        # algo4_reject0 = round(np.mean(algo4_Hired_reject_num), 2)
        # algo4_var0_reject = round(np.var(algo4_Hired_reject_num), 2)
        # algo4_reject1 = round(np.mean(algo4_Crowdsourced_reject_num), 2)
        # algo4_var1_reject = round(np.var(algo4_Crowdsourced_reject_num), 2)
        # algo4_reject = round(np.mean(algo4_Hired_reject_num + algo4_Crowdsourced_reject_num), 2)
        # algo4_var_reject = round(np.var(algo4_Hired_reject_num + algo4_Crowdsourced_reject_num), 2)

        # print("Average Reject Numbers per Courier for Evaluation Between Algos:")
        # print(f"Algo1: Hired rejects average {algo1_reject0} orders (Var: {algo1_var0_reject}), Crowdsourced rejects average {algo1_reject1} orders (Var: {algo1_var1_reject}), Total reject number per courier is {algo1_reject} orders (Var: {algo1_var_reject})")
        # print(f"Algo2: Hired rejects average {algo2_reject0} orders (Var: {algo2_var0_reject}), Crowdsourced rejects average {algo2_reject1} orders (Var: {algo2_var1_reject}), Total reject number per courier is {algo2_reject} orders (Var: {algo2_var_reject})")
        # print(f"Algo3: Hired rejects average {algo3_reject0} orders (Var: {algo3_var0_reject}), Crowdsourced rejects average {algo3_reject1} orders (Var: {algo3_var1_reject}), Total reject number per courier is {algo3_reject} orders (Var: {algo3_var_reject})")
        # print(f"Algo4: Hired rejects average {algo4_reject0} orders (Var: {algo4_var0_reject}), Crowdsourced rejects average {algo4_reject1} orders (Var: {algo4_var1_reject}), Total reject number per courier is {algo4_reject} orders (Var: {algo4_var_reject})")

        # self.writter.add_scalar('Eval Average Rejection/Algo1 Total', algo1_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo1 Hired', algo1_reject0, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo1 Crowdsourced', algo1_reject1, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo1 Total Var', algo1_var_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo1 Hired Var', algo1_var0_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo1 Crowdsourced Var', algo1_var1_reject, self.eval_num)

        # self.writter.add_scalar('Eval Average Rejection/Algo2 Total', algo2_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo2 Hired', algo2_reject0, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo2 Crowdsourced', algo2_reject1, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo2 Total Var', algo2_var_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo2 Hired Var', algo2_var0_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo2 Crowdsourced Var', algo2_var1_reject, self.eval_num)

        # self.writter.add_scalar('Eval Average Rejection/Algo3 Total', algo3_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo3 Hired', algo3_reject0, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo3 Crowdsourced', algo3_reject1, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo3 Total Var', algo3_var_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo3 Hired Var', algo3_var0_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo3 Crowdsourced Var', algo3_var1_reject, self.eval_num)

        # self.writter.add_scalar('Eval Average Rejection/Algo4 Total', algo4_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo4 Hired', algo4_reject0, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo4 Crowdsourced', algo4_reject1, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo4 Total Var', algo4_var_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo4 Hired Var', algo4_var0_reject, self.eval_num)
        # self.writter.add_scalar('Eval Average Rejection/Algo4 Crowdsourced Var', algo4_var1_reject, self.eval_num)
        
        # -----------------------
        # Average Order Price
        algo1_price_per_order0 = round(np.mean(algo1_order0_price), 2)
        algo1_var0_price = round(np.var(algo1_order0_price), 2)
        algo1_price_per_order1 = round(np.mean(algo1_order1_price), 2)
        algo1_var1_price = round(np.var(algo1_order1_price), 2)
        algo1_price_per_order = round(np.mean(algo1_order0_price + algo1_order1_price), 2)
        algo1_var_price = round(np.var(algo1_order0_price + algo1_order1_price), 2)

        algo2_price_per_order0 = round(np.mean(algo2_order0_price), 2)
        algo2_var0_price = round(np.var(algo2_order0_price), 2)
        algo2_price_per_order1 = round(np.mean(algo2_order1_price), 2)
        algo2_var1_price = round(np.var(algo2_order1_price), 2)
        algo2_price_per_order = round(np.mean((algo2_order0_price + algo2_order1_price)), 2)
        algo2_var_price = round(np.var(algo2_order0_price + algo2_order1_price), 2)

        algo3_price_per_order0 = round(np.mean(algo3_order0_price), 2)
        algo3_var0_price = round(np.var(algo3_order0_price), 2)
        algo3_price_per_order1 = round(np.mean(algo3_order1_price), 2)
        algo3_var1_price = round(np.var(algo3_order1_price), 2)
        algo3_price_per_order = round(np.mean(algo3_order0_price + algo3_order1_price), 2)
        algo3_var_price = round(np.var(algo3_order0_price + algo3_order1_price), 2)

        algo4_price_per_order0 = round(np.mean(algo4_order0_price), 2)
        algo4_var0_price = round(np.var(algo4_order0_price), 2)
        algo4_price_per_order1 = round(np.mean(algo4_order1_price), 2)
        algo4_var1_price = round(np.var(algo4_order1_price), 2)
        algo4_price_per_order = round(np.mean(algo4_order0_price + algo4_order1_price), 2)
        algo4_var_price = round(np.var(algo4_order0_price + algo4_order1_price), 2)

        print("Average Price per Order for Evaluation Between Algos:")
        print(f"Algo1: Hired average price per order is {algo1_price_per_order0} dollars (Var: {algo1_var0_price}), Crowdsourced is {algo1_price_per_order1} dollars (Var: {algo1_var1_price}), Total average is {algo1_price_per_order} dollars (Var: {algo1_var_price})")
        print(f"Algo2: Hired average price per order is {algo2_price_per_order0} dollars (Var: {algo2_var0_price}), Crowdsourced is {algo2_price_per_order1} dollars (Var: {algo2_var1_price}), Total average is {algo2_price_per_order} dollars (Var: {algo2_var_price})")
        print(f"Algo3: Hired average price per order is {algo3_price_per_order0} dollars (Var: {algo3_var0_price}), Crowdsourced is {algo3_price_per_order1} dollars (Var: {algo3_var1_price}), Total average is {algo3_price_per_order} dollars (Var: {algo3_var_price})")
        print(f"Algo4: Hired average price per order is {algo4_price_per_order0} dollars (Var: {algo4_var0_price}), Crowdsourced is {algo4_price_per_order1} dollars (Var: {algo4_var1_price}), Total average is {algo4_price_per_order} dollars (Var: {algo4_var_price})")

        self.writter.add_scalar('Eval Average Price/Algo1 Total', algo1_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Hired', algo1_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Crowdsourced', algo1_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Total Var', algo1_var_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Hired Var', algo1_var0_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Crowdsourced Var', algo1_var1_price, self.eval_num)

        self.writter.add_scalar('Eval Average Price/Algo2 Total', algo2_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Hired', algo2_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Crowdsourced', algo2_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Total Var', algo2_var_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Hired Var', algo2_var0_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Crowdsourced Var', algo2_var1_price, self.eval_num)

        self.writter.add_scalar('Eval Average Price/Algo3 Total', algo3_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Hired', algo3_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Crowdsourced', algo3_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Total Var', algo3_var_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Hired Var', algo3_var0_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Crowdsourced Var', algo3_var1_price, self.eval_num)

        self.writter.add_scalar('Eval Average Price/Algo4 Total', algo4_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo4 Hired', algo4_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo4 Crowdsourced', algo4_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo4 Total Var', algo4_var_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo4 Hired Var', algo4_var0_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo4 Crowdsourced Var', algo4_var1_price, self.eval_num)
                
        # -----------------------
        # Average Courier Income
        algo1_income0 = round(np.mean(algo1_Hired_income), 2)
        algo1_var0_income = round(np.var(algo1_Hired_income), 2)
        algo1_income1 = round(np.mean(algo1_Crowdsourced_income), 2)
        algo1_var1_income = round(np.var(algo1_Crowdsourced_income), 2)
        algo1_income = round(np.mean(algo1_Hired_income + algo1_Crowdsourced_income), 2)
        algo1_var_income = round(np.var(algo1_Hired_income + algo1_Crowdsourced_income), 2)

        algo2_income0 = round(np.mean(algo2_Hired_income), 2)
        algo2_var0_income = round(np.var(algo2_Hired_income), 2)
        algo2_income1 = round(np.mean(algo2_Crowdsourced_income), 2)
        algo2_var1_income = round(np.var(algo2_Crowdsourced_income), 2)
        algo2_income = round(np.mean(algo2_Hired_income + algo2_Crowdsourced_income), 2)
        algo2_var_income = round(np.var(algo2_Hired_income + algo2_Crowdsourced_income), 2)

        algo3_income0 = round(np.mean(algo3_Hired_income), 2)
        algo3_var0_income = round(np.var(algo3_Hired_income), 2)
        algo3_income1 = round(np.mean(algo3_Crowdsourced_income), 2)
        algo3_var1_income = round(np.var(algo3_Crowdsourced_income), 2)
        algo3_income = round(np.mean(algo3_Hired_income + algo3_Crowdsourced_income), 2)
        algo3_var_income = round(np.var(algo3_Hired_income + algo3_Crowdsourced_income), 2)

        algo4_income0 = round(np.mean(algo4_Hired_income), 2)
        algo4_var0_income = round(np.var(algo4_Hired_income), 2)
        algo4_income1 = round(np.mean(algo4_Crowdsourced_income), 2)
        algo4_var1_income = round(np.var(algo4_Crowdsourced_income), 2)
        algo4_income = round(np.mean(algo4_Hired_income + algo4_Crowdsourced_income), 2)
        algo4_var_income = round(np.var(algo4_Hired_income + algo4_Crowdsourced_income), 2)

        print("Average Income per Courier for Evaluation Between Algos:")
        print(f"Algo1: Hired's average income is {algo1_income0} dollars (Var: {algo1_var0_income}), Crowdsourced's average income is {algo1_income1} dollars (Var: {algo1_var1_income}), Total income per courier is {algo1_income} dollars (Var: {algo1_var_income})")
        print(f"Algo2: Hired's average income is {algo2_income0} dollars (Var: {algo2_var0_income}), Crowdsourced's average income is {algo2_income1} dollars (Var: {algo2_var1_income}), Total income per courier is {algo2_income} dollars (Var: {algo2_var_income})")
        print(f"Algo3: Hired's average income is {algo3_income0} dollars (Var: {algo3_var0_income}), Crowdsourced's average income is {algo3_income1} dollars (Var: {algo3_var1_income}), Total income per courier is {algo3_income} dollars (Var: {algo3_var_income})")
        print(f"Algo4: Hired's average income is {algo4_income0} dollars (Var: {algo4_var0_income}), Crowdsourced's average income is {algo4_income1} dollars (Var: {algo4_var1_income}), Total income per courier is {algo4_income} dollars (Var: {algo4_var_income})")

        self.writter.add_scalar('Eval Average Income/Algo1 Total', algo1_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Hired', algo1_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Crowdsourced', algo1_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Total Var', algo1_var_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Hired Var', algo1_var0_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Crowdsourced Var', algo1_var1_income, self.eval_num)
        self.writter.add_scalar('Eval Platform Cost/Algo1', platform_cost1, self.eval_num)


        self.writter.add_scalar('Eval Average Income/Algo2 Total', algo2_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Hired', algo2_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Crowdsourced', algo2_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Total Var', algo2_var_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Hired Var', algo2_var0_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Crowdsourced Var', algo2_var1_income, self.eval_num)
        self.writter.add_scalar('Eval Platform Cost/Algo2', platform_cost2, self.eval_num)

        self.writter.add_scalar('Eval Average Income/Algo3 Total', algo3_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Hired', algo3_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Crowdsourced', algo3_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Total Var', algo3_var_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Hired Var', algo3_var0_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Crowdsourced Var', algo3_var1_income, self.eval_num)
        self.writter.add_scalar('Eval Platform Cost/Algo3', platform_cost3, self.eval_num)

        self.writter.add_scalar('Eval Average Income/Algo4 Total', algo4_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo4 Hired', algo4_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo4 Crowdsourced', algo4_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo4 Total Var', algo4_var_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo4 Hired Var', algo4_var0_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo4 Crowdsourced Var', algo4_var1_income, self.eval_num)
        self.writter.add_scalar('Eval Platform Cost/Algo4', platform_cost4, self.eval_num)

        # -----------------------
        # Average Courier Finishing Number
        algo1_finish0 = round(np.mean(algo1_Hired_finish_num), 2)
        algo1_var0_finish = round(np.var(algo1_Hired_finish_num), 2)
        algo1_finish1 = round(np.mean(algo1_Crowdsourced_finish_num), 2)
        algo1_var1_finish = round(np.var(algo1_Crowdsourced_finish_num), 2)
        algo1_finish = round(np.mean(algo1_Hired_finish_num + algo1_Crowdsourced_finish_num), 2)
        algo1_var_finish = round(np.var(algo1_Hired_finish_num + algo1_Crowdsourced_finish_num), 2)

        algo2_finish0 = round(np.mean(algo2_Hired_finish_num), 2)
        algo2_var0_finish = round(np.var(algo2_Hired_finish_num), 2)
        algo2_finish1 = round(np.mean(algo2_Crowdsourced_finish_num), 2)
        algo2_var1_finish = round(np.var(algo2_Crowdsourced_finish_num), 2)
        algo2_finish = round(np.mean(algo2_Hired_finish_num + algo2_Crowdsourced_finish_num), 2)
        algo2_var_finish = round(np.var(algo2_Hired_finish_num + algo2_Crowdsourced_finish_num), 2)

        algo3_finish0 = round(np.mean(algo3_Hired_finish_num), 2)
        algo3_var0_finish = round(np.var(algo3_Hired_finish_num), 2)
        algo3_finish1 = round(np.mean(algo3_Crowdsourced_finish_num), 2)
        algo3_var1_finish = round(np.var(algo3_Crowdsourced_finish_num), 2)
        algo3_finish = round(np.mean(algo3_Hired_finish_num + algo3_Crowdsourced_finish_num), 2)
        algo3_var_finish = round(np.var(algo3_Hired_finish_num + algo3_Crowdsourced_finish_num), 2)

        algo4_finish0 = round(np.mean(algo4_Hired_finish_num), 2)
        algo4_var0_finish = round(np.var(algo4_Hired_finish_num), 2)
        algo4_finish1 = round(np.mean(algo4_Crowdsourced_finish_num), 2)
        algo4_var1_finish = round(np.var(algo4_Crowdsourced_finish_num), 2)
        algo4_finish = round(np.mean(algo4_Hired_finish_num + algo4_Crowdsourced_finish_num), 2)
        algo4_var_finish = round(np.var(algo4_Hired_finish_num + algo4_Crowdsourced_finish_num), 2)

        print("Average Finished Orders per Courier for Evaluation Between Algos:")
        print(f"Algo1: Hired finishes average {algo1_finish0} orders (Var: {algo1_var0_finish}), Crowdsourced finishes average {algo1_finish1} orders (Var: {algo1_var1_finish}), Total finish number per courier is {algo1_finish} orders (Var: {algo1_var_finish})")
        print(f"Algo2: Hired finishes average {algo2_finish0} orders (Var: {algo2_var0_finish}), Crowdsourced finishes average {algo2_finish1} orders (Var: {algo2_var1_finish}), Total finish number per courier is {algo2_finish} orders (Var: {algo2_var_finish})")
        print(f"Algo3: Hired finishes average {algo3_finish0} orders (Var: {algo3_var0_finish}), Crowdsourced finishes average {algo3_finish1} orders (Var: {algo3_var1_finish}), Total finish number per courier is {algo3_finish} orders (Var: {algo3_var_finish})")
        print(f"Algo4: Hired finishes average {algo4_finish0} orders (Var: {algo4_var0_finish}), Crowdsourced finishes average {algo4_finish1} orders (Var: {algo4_var1_finish}), Total finish number per courier is {algo4_finish} orders (Var: {algo4_var_finish})")

        self.writter.add_scalar('Eval Average Finish/Algo1 Total', algo1_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Hired', algo1_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Crowdsourced', algo1_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Total Var', algo1_var_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Hired Var', algo1_var0_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Crowdsourced Var', algo1_var1_finish, self.eval_num)

        self.writter.add_scalar('Eval Average Finish/Algo2 Total', algo2_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Hired', algo2_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Crowdsourced', algo2_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Total Var', algo2_var_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Hired Var', algo2_var0_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Crowdsourced Var', algo2_var1_finish, self.eval_num)

        self.writter.add_scalar('Eval Average Finish/Algo3 Total', algo3_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Hired', algo3_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Crowdsourced', algo3_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Total Var', algo3_var_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Hired Var', algo3_var0_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Crowdsourced Var', algo3_var1_finish, self.eval_num)

        self.writter.add_scalar('Eval Average Finish/Algo4 Total', algo4_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo4 Hired', algo4_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo4 Crowdsourced', algo4_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo4 Total Var', algo4_var_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo4 Hired Var', algo4_var0_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo4 Crowdsourced Var', algo4_var1_finish, self.eval_num)

        # -----------------------
        # Average Courier Leisure Time
        algo1_avg0_leisure = round(np.mean(algo1_Hired_leisure_time) / 60, 2)
        algo1_var0_leisure = round(np.var(algo1_Hired_leisure_time) / 60**2, 2)
        algo1_avg1_leisure = round(np.mean(algo1_Crowdsourced_leisure_time) / 60, 2)
        algo1_var1_leisure = round(np.var(algo1_Crowdsourced_leisure_time) / 60**2, 2)
        algo1_avg_leisure = round(np.mean(algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time) / 60, 2)
        algo1_var_leisure = round(np.var(algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time) / 60**2, 2)

        algo2_avg0_leisure = round(np.mean(algo2_Hired_leisure_time) / 60, 2)
        algo2_var0_leisure = round(np.var(algo2_Hired_leisure_time) / 60**2, 2)
        algo2_avg1_leisure = round(np.mean(algo2_Crowdsourced_leisure_time) / 60, 2)
        algo2_var1_leisure = round(np.var(algo2_Crowdsourced_leisure_time) / 60**2, 2)
        algo2_avg_leisure = round(np.mean(algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time) / 60, 2)
        algo2_var_leisure = round(np.var(algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time) / 60**2, 2)

        algo3_avg0_leisure = round(np.mean(algo3_Hired_leisure_time) / 60, 2)
        algo3_var0_leisure = round(np.var(algo3_Hired_leisure_time) / 60**2, 2)
        algo3_avg1_leisure = round(np.mean(algo3_Crowdsourced_leisure_time) / 60, 2)
        algo3_var1_leisure = round(np.var(algo3_Crowdsourced_leisure_time) / 60**2, 2)
        algo3_avg_leisure = round(np.mean(algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time) / 60, 2)
        algo3_var_leisure = round(np.var(algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time) / 60**2, 2)

        algo4_avg0_leisure = round(np.mean(algo4_Hired_leisure_time) / 60, 2)
        algo4_var0_leisure = round(np.var(algo4_Hired_leisure_time) / 60**2, 2)
        algo4_avg1_leisure = round(np.mean(algo4_Crowdsourced_leisure_time) / 60, 2)
        algo4_var1_leisure = round(np.var(algo4_Crowdsourced_leisure_time) / 60**2, 2)
        algo4_avg_leisure = round(np.mean(algo4_Hired_leisure_time + algo4_Crowdsourced_leisure_time) / 60, 2)
        algo4_var_leisure = round(np.var(algo4_Hired_leisure_time + algo4_Crowdsourced_leisure_time) / 60**2, 2)

        print("Average leisure time per courier for Evaluation Between Algos:")
        print(f"Algo1: Hired leisure time is {algo1_avg0_leisure} minutes (Var: {algo1_var0_leisure}), Crowdsourced leisure time is {algo1_avg1_leisure} minutes (Var: {algo1_var1_leisure}), Total leisure time per courier is {algo1_avg_leisure} minutes (Var: {algo1_var_leisure})")
        print(f"Algo2: Hired leisure time is {algo2_avg0_leisure} minutes (Var: {algo2_var0_leisure}), Crowdsourced leisure time is {algo2_avg1_leisure} minutes (Var: {algo2_var1_leisure}), Total leisure time per courier is {algo2_avg_leisure} minutes (Var: {algo2_var_leisure})")
        print(f"Algo3: Hired leisure time is {algo3_avg0_leisure} minutes (Var: {algo3_var0_leisure}), Crowdsourced leisure time is {algo3_avg1_leisure} minutes (Var: {algo3_var1_leisure}), Total leisure time per courier is {algo3_avg_leisure} minutes (Var: {algo3_var_leisure})")
        print(f"Algo4: Hired leisure time is {algo4_avg0_leisure} minutes (Var: {algo4_var0_leisure}), Crowdsourced leisure time is {algo4_avg1_leisure} minutes (Var: {algo4_var1_leisure}), Total leisure time per courier is {algo4_avg_leisure} minutes (Var: {algo4_var_leisure})")

        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Total', algo1_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Hired', algo1_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Crowdsourced', algo1_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Total Var', algo1_var_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Hired Var', algo1_var0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Crowdsourced Var', algo1_var1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Total', algo2_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Hired', algo2_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Crowdsourced', algo2_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Total Var', algo2_var_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Hired Var', algo2_var0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Crowdsourced Var', algo2_var1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Total', algo3_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Hired', algo3_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Crowdsourced', algo3_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Total Var', algo3_var_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Hired Var', algo3_var0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Crowdsourced Var', algo3_var1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Total', algo4_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Hired', algo4_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Crowdsourced', algo4_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Total Var', algo4_var_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Hired Var', algo4_var0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo4 Crowdsourced Var', algo4_var1_leisure, self.eval_num)
        
        # -----------------------
        # Average Courier running Time
        algo1_avg0_running = round(np.mean(algo1_Hired_running_time) / 60, 2)
        algo1_var0_running = round(np.var(algo1_Hired_running_time) / 60**2, 2)
        algo1_avg1_running = round(np.mean(algo1_Crowdsourced_running_time) / 60, 2)
        algo1_var1_running = round(np.var(algo1_Crowdsourced_running_time) / 60**2, 2)
        algo1_avg_running = round(np.mean(algo1_Hired_running_time + algo1_Crowdsourced_running_time) / 60, 2)
        algo1_var_running = round(np.var(algo1_Hired_running_time + algo1_Crowdsourced_running_time) / 60**2, 2)

        algo2_avg0_running = round(np.mean(algo2_Hired_running_time) / 60, 2)
        algo2_var0_running = round(np.var(algo2_Hired_running_time) / 60**2, 2)
        algo2_avg1_running = round(np.mean(algo2_Crowdsourced_running_time) / 60, 2)
        algo2_var1_running = round(np.var(algo2_Crowdsourced_running_time) / 60**2, 2)
        algo2_avg_running = round(np.mean(algo2_Hired_running_time + algo2_Crowdsourced_running_time) / 60, 2)
        algo2_var_running = round(np.var(algo2_Hired_running_time + algo2_Crowdsourced_running_time) / 60**2, 2)

        algo3_avg0_running = round(np.mean(algo3_Hired_running_time) / 60, 2)
        algo3_var0_running = round(np.var(algo3_Hired_running_time) / 60**2, 2)
        algo3_avg1_running = round(np.mean(algo3_Crowdsourced_running_time) / 60, 2)
        algo3_var1_running = round(np.var(algo3_Crowdsourced_running_time) / 60**2, 2)
        algo3_avg_running = round(np.mean(algo3_Hired_running_time + algo3_Crowdsourced_running_time) / 60, 2)
        algo3_var_running = round(np.var(algo3_Hired_running_time + algo3_Crowdsourced_running_time) / 60**2, 2)

        algo4_avg0_running = round(np.mean(algo4_Hired_running_time) / 60, 2)
        algo4_var0_running = round(np.var(algo4_Hired_running_time) / 60**2, 2)
        algo4_avg1_running = round(np.mean(algo4_Crowdsourced_running_time) / 60, 2)
        algo4_var1_running = round(np.var(algo4_Crowdsourced_running_time) / 60**2, 2)
        algo4_avg_running = round(np.mean(algo4_Hired_running_time + algo4_Crowdsourced_running_time) / 60, 2)
        algo4_var_running = round(np.var(algo4_Hired_running_time + algo4_Crowdsourced_running_time) / 60**2, 2)

        print("Average running time per courier for Evaluation Between Algos:")
        print(f"Algo1: Hired running time is {algo1_avg0_running} minutes (Var: {algo1_var0_running}), Crowdsourced running time is {algo1_avg1_running} minutes (Var: {algo1_var1_running}), Total running time per courier is {algo1_avg_running} minutes (Var: {algo1_var_running})")
        print(f"Algo2: Hired running time is {algo2_avg0_running} minutes (Var: {algo2_var0_running}), Crowdsourced running time is {algo2_avg1_running} minutes (Var: {algo2_var1_running}), Total running time per courier is {algo2_avg_running} minutes (Var: {algo2_var_running})")
        print(f"Algo3: Hired running time is {algo3_avg0_running} minutes (Var: {algo3_var0_running}), Crowdsourced running time is {algo3_avg1_running} minutes (Var: {algo3_var1_running}), Total running time per courier is {algo3_avg_running} minutes (Var: {algo3_var_running})")
        print(f"Algo4: Hired running time is {algo4_avg0_running} minutes (Var: {algo4_var0_running}), Crowdsourced running time is {algo4_avg1_running} minutes (Var: {algo4_var1_running}), Total running time per courier is {algo4_avg_running} minutes (Var: {algo4_var_running})")

        self.writter.add_scalar('Eval Average running Time/Algo1 Total', algo1_avg_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo1 Hired', algo1_avg0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo1 Crowdsourced', algo1_avg1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo1 Total Var', algo1_var_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo1 Hired Var', algo1_var0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo1 Crowdsourced Var', algo1_var1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Total', algo2_avg_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Hired', algo2_avg0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Crowdsourced', algo2_avg1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Total Var', algo2_var_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Hired Var', algo2_var0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo2 Crowdsourced Var', algo2_var1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Total', algo3_avg_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Hired', algo3_avg0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Crowdsourced', algo3_avg1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Total Var', algo3_var_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Hired Var', algo3_var0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo3 Crowdsourced Var', algo3_var1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Total', algo4_avg_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Hired', algo4_avg0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Crowdsourced', algo4_avg1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Total Var', algo4_var_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Hired Var', algo4_var0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo4 Crowdsourced Var', algo4_var1_running, self.eval_num)

        message = (
            f"\nIn Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} on, {algo1_order0_num} Order0, {algo1_order1_num} Order1, {algo1_order_wait} ({round(100 * algo1_order_wait / (algo1_order_wait + algo1_order0_num + algo1_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} on, {algo2_order0_num} Order0, {algo2_order1_num} Order1, {algo2_order_wait} ({round(100 * algo2_order_wait / (algo2_order_wait + algo2_order0_num + algo2_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} on, {algo3_order0_num} Order0, {algo3_order1_num} Order1, {algo3_order_wait} ({round(100 * algo3_order_wait / (algo3_order_wait + algo3_order0_num + algo3_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo4 there are {algo4_Hired_num} Hired, {algo4_Crowdsourced_num} Crowdsourced with {algo4_Crowdsourced_on} on, {algo4_order0_num} Order0, {algo4_order1_num} Order1, {algo4_order_wait} ({round(100 * algo4_order_wait / (algo4_order_wait + algo4_order0_num + algo4_order1_num), 2)}%) Orders waiting to be paired\n"
            f"Total Reward for Evaluation Between Algos:\n"
            f"Algo1: {round(algo1_eval_episode_rewards_sum, 2)}\n"
            f"Algo2: {round(algo2_eval_episode_rewards_sum, 2)}\n"
            f"Algo3: {round(algo3_eval_episode_rewards_sum, 2)}\n"
            f"Algo4: {round(algo4_eval_episode_rewards_sum, 2)}\n"
            f"Average Travel Distance per Courier Between Algos:\n"
            f"Algo1: Hired - {algo1_distance0} km (Var: {algo1_var0_distance}), Crowdsourced - {algo1_distance1} km (Var: {algo1_var1_distance}), Total - {algo1_distance} km (Var: {algo1_var_distance})\n"
            f"Algo2: Hired - {algo2_distance0} km (Var: {algo2_var0_distance}), Crowdsourced - {algo2_distance1} km (Var: {algo2_var1_distance}), Total - {algo2_distance} km (Var: {algo2_var_distance})\n"
            f"Algo3: Hired - {algo3_distance0} km (Var: {algo3_var0_distance}), Crowdsourced - {algo3_distance1} km (Var: {algo3_var1_distance}), Total - {algo3_distance} km (Var: {algo3_var_distance})\n"
            f"Algo4: Hired - {algo4_distance0} km (Var: {algo4_var0_distance}), Crowdsourced - {algo4_distance1} km (Var: {algo4_var1_distance}), Total - {algo4_distance} km (Var: {algo4_var_distance})\n"
            "Average Speed per Courier Between Algos:\n"
            f"Algo1: Hired average speed is {algo1_avg0_speed} m/s (Var: {algo1_var0_speed}), Crowdsourced average speed is {algo1_avg1_speed} m/s (Var: {algo1_var1_speed}) and average speed per courier is {algo1_avg_speed} m/s (Var: {algo1_var_speed})\n"
            f"Algo2: Hired average speed is {algo2_avg0_speed} m/s (Var: {algo2_var0_speed}), Crowdsourced average speed is {algo2_avg1_speed} m/s (Var: {algo2_var1_speed}) and average speed per courier is {algo2_avg_speed} m/s (Var: {algo2_var_speed})\n"
            f"Algo3: Hired average speed is {algo3_avg0_speed} m/s (Var: {algo3_var0_speed}), Crowdsourced average speed is {algo3_avg1_speed} m/s (Var: {algo3_var1_speed}) and average speed per courier is {algo3_avg_speed} m/s (Var: {algo3_var_speed})\n"
            f"Algo4: Hired average speed is {algo4_avg0_speed} m/s (Var: {algo4_var0_speed}), Crowdsourced average speed is {algo4_avg1_speed} m/s (Var: {algo4_var1_speed}) and average speed per courier is {algo4_avg_speed} m/s (Var: {algo4_var_speed})\n"
            "Rate of Overspeed for Evaluation Between Algos:\n"
            f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}\n"
            f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}\n"
            f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}\n"
            f"Algo4: Hired - {algo4_overspeed0}, Crowdsourced - {algo4_overspeed1}, Total rate - {algo4_overspeed}\n"
            "Average Price per order for Evaluation Between Algos:\n"
            f"Algo1: The average price of Hired's order is {algo1_price_per_order0} dollar (Var: {algo1_var0_price}) with {algo1_order0_num} orders, Crowdsourced's is {algo1_price_per_order1} dollar (Var: {algo1_var1_price}) with {algo1_order1_num} orders and for all is {algo1_price_per_order} dollar (Var: {algo1_var_price})\n"
            f"Algo2: The average price of Hired's order is {algo2_price_per_order0} dollar (Var: {algo2_var0_price}) with {algo2_order0_num} orders, Crowdsourced's is {algo2_price_per_order1} dollar (Var: {algo2_var1_price}) with {algo2_order1_num} orders and for all is {algo2_price_per_order} dollar (Var: {algo2_var_price})\n"
            f"Algo3: The average price of Hired's order is {algo3_price_per_order0} dollar (Var: {algo3_var0_price}) with {algo3_order0_num} orders, Crowdsourced's is {algo3_price_per_order1} dollar (Var: {algo3_var1_price}) with {algo3_order1_num} orders and for all is {algo3_price_per_order} dollar (Var: {algo3_var_price})\n"
            f"Algo4: The average price of Hired's order is {algo4_price_per_order0} dollar (Var: {algo4_var0_price}) with {algo4_order0_num} orders, Crowdsourced's is {algo4_price_per_order1} dollar (Var: {algo4_var1_price}) with {algo4_order1_num} orders and for all is {algo4_price_per_order} dollar (Var: {algo4_var_price})\n"
            "Average Income per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average income is {algo1_income0} dollar (Var: {algo1_var0_income}), Crowdsourced's average income is {algo1_income1} dollar (Var: {algo1_var1_income}) and Total income per courier is {algo1_income} dollar (Var: {algo1_var_income}), The platform total cost is {round(platform_cost1, 2)} dollar\n"
            f"Algo2: Hired's average income is {algo2_income0} dollar (Var: {algo2_var0_income}), Crowdsourced's average income is {algo2_income1} dollar (Var: {algo2_var1_income}) and Total income per courier is {algo2_income} dollar (Var: {algo2_var_income}), The platform total cost is {round(platform_cost2, 2)} dollar\n"
            f"Algo3: Hired's average income is {algo3_income0} dollar (Var: {algo3_var0_income}), Crowdsourced's average income is {algo3_income1} dollar (Var: {algo3_var1_income}) and Total income per courier is {algo3_income} dollar (Var: {algo3_var_income}), The platform total cost is {round(platform_cost3, 2)} dollar\n"
            f"Algo4: Hired's average income is {algo4_income0} dollar (Var: {algo4_var0_income}), Crowdsourced's average income is {algo4_income1} dollar (Var: {algo4_var1_income}) and Total income per courier is {algo4_income} dollar (Var: {algo4_var_income}), The platform total cost is {round(platform_cost4, 2)} dollar\n"
            "Average Leisure Time per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average leisure time is {algo1_avg0_leisure} minutes (Var: {algo1_var0_leisure}), Crowdsourced's average leisure time is {algo1_avg1_leisure} minutes (Var: {algo1_var1_leisure}) and Total leisure time per courier is {algo1_avg_leisure} minutes (Var: {algo1_var_leisure})\n"
            f"Algo2: Hired's average leisure time is {algo2_avg0_leisure} minutes (Var: {algo2_var0_leisure}), Crowdsourced's average leisure time is {algo2_avg1_leisure} minutes (Var: {algo2_var1_leisure}) and Total leisure time per courier is {algo2_avg_leisure} minutes (Var: {algo2_var_leisure})\n"
            f"Algo3: Hired's average leisure time is {algo3_avg0_leisure} minutes (Var: {algo3_var0_leisure}), Crowdsourced's average leisure time is {algo3_avg1_leisure} minutes (Var: {algo3_var1_leisure}) and Total leisure time per courier is {algo3_avg_leisure} minutes (Var: {algo3_var_leisure})\n"
            f"Algo4: Hired's average leisure time is {algo4_avg0_leisure} minutes (Var: {algo4_var0_leisure}), Crowdsourced's average leisure time is {algo4_avg1_leisure} minutes (Var: {algo4_var1_leisure}) and Total leisure time per courier is {algo4_avg_leisure} minutes (Var: {algo4_var_leisure})\n"
            "Average Running Time per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average running time is {algo1_avg0_running} minutes (Var: {algo1_var0_running}), Crowdsourced's average running time is {algo1_avg1_running} minutes (Var: {algo1_var1_running}) and Total running time per courier is {algo1_avg_running} minutes (Var: {algo1_var_running})\n"
            f"Algo2: Hired's average running time is {algo2_avg0_running} minutes (Var: {algo2_var0_running}), Crowdsourced's average running time is {algo2_avg1_running} minutes (Var: {algo2_var1_running}) and Total running time per courier is {algo2_avg_running} minutes (Var: {algo2_var_running})\n"
            f"Algo3: Hired's average running time is {algo3_avg0_running} minutes (Var: {algo3_var0_running}), Crowdsourced's average running time is {algo3_avg1_running} minutes (Var: {algo3_var1_running}) and Total running time per courier is {algo3_avg_running} minutes (Var: {algo3_var_running})\n"
            f"Algo4: Hired's average running time is {algo4_avg0_running} minutes (Var: {algo4_var0_running}), Crowdsourced's average running time is {algo4_avg1_running} minutes (Var: {algo4_var1_running}) and Total running time per courier is {algo4_avg_running} minutes (Var: {algo4_var_running})\n"
            "Average Order Finished per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired courier finishes average {algo1_finish0} orders (Var: {algo1_var0_finish}), Crowdsourced courier finishes average {algo1_finish1} orders (Var: {algo1_var1_finish}) and Total is {algo1_finish} orders (Var: {algo1_var_finish})\n"
            f"Algo2: Hired courier finishes average {algo2_finish0} orders (Var: {algo2_var0_finish}), Crowdsourced courier finishes average {algo2_finish1} orders (Var: {algo2_var1_finish}) and Total is {algo2_finish} orders (Var: {algo2_var_finish})\n"
            f"Algo3: Hired courier finishes average {algo3_finish0} orders (Var: {algo3_var0_finish}), Crowdsourced courier finishes average {algo3_finish1} orders (Var: {algo3_var1_finish}) and Total is {algo3_finish} orders (Var: {algo3_var_finish})\n"
            f"Algo4: Hired courier finishes average {algo4_finish0} orders (Var: {algo4_var0_finish}), Crowdsourced courier finishes average {algo4_finish1} orders (Var: {algo4_var1_finish}) and Total is {algo4_finish} orders (Var: {algo4_var_finish})\n"
        )
        
        if algo1_count_dropped_orders0 + algo1_count_dropped_orders1 == 0:
            print("No order is dropped in Algo1")
            algo1_late_rate = -1
            algo1_late_rate0 = -1
            algo1_late_rate1 = -1
            algo1_ETA_usage_rate = -1
            algo1_ETA_usage_rate0 = -1
            algo1_ETA_usage_rate1 = -1
            algo1_var_ETA = 0
            algo1_var0_ETA = 0
            algo1_var1_ETA = 0

            self.writter.add_scalar('Eval Late Order Rate/Algo1 Total', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Hired', algo1_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Crowdsourced', algo1_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total', algo1_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total Var', algo1_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired', algo1_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired Var', algo1_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced', algo1_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced Var', algo1_var1_ETA, self.eval_num)
            
            message += "No order is dropped in Algo1\n"
        else:
            if algo1_count_dropped_orders0:                
                algo1_late_rate0 = round(algo1_late_orders0 / algo1_count_dropped_orders0, 2)
                algo1_ETA_usage_rate0 = round(np.mean(algo1_ETA_usage0), 2)
                algo1_var0_ETA = round(np.var(algo1_ETA_usage0), 2)
            else:
                algo1_late_rate0 = -1
                algo1_ETA_usage_rate0 = -1
                algo1_var0_ETA = 0
                
            if algo1_count_dropped_orders1:                
                algo1_late_rate1 = round(algo1_late_orders1 / algo1_count_dropped_orders1, 2)
                algo1_ETA_usage_rate1 = round(np.mean(algo1_ETA_usage1), 2)
                algo1_var1_ETA = round(np.var(algo1_ETA_usage1), 2)
            else:
                algo1_late_rate1 = -1
                algo1_ETA_usage_rate1 = -1
                algo1_var1_ETA = 0
                
            algo1_late_rate = round((algo1_late_orders0 + algo1_late_orders1) / (algo1_count_dropped_orders0 +algo1_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate}")

            algo1_ETA_usage_rate = round(np.mean(algo1_ETA_usage0 + algo1_ETA_usage1), 2)
            algo1_var_ETA = round(np.var(algo1_ETA_usage0 + algo1_ETA_usage1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0} (Var: {algo1_var0_ETA}), Crowdsourced - {algo1_ETA_usage_rate1} (Var: {algo1_var1_ETA}), Total - {algo1_ETA_usage_rate} (Var: {algo1_var_ETA})")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Total', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Hired', algo1_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Crowdsourced', algo1_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total', algo1_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired', algo1_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced', algo1_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total Var', algo1_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired Var', algo1_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced Var', algo1_var1_ETA, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0} (Var: {algo1_var0_ETA}), Crowdsourced - {algo1_ETA_usage_rate1} (Var: {algo1_var1_ETA}), Total - {algo1_ETA_usage_rate} (Var: {algo1_var_ETA})\n"
        
        if algo2_count_dropped_orders0 + algo2_count_dropped_orders1 == 0:
            print("No order is dropped in Algo2")
            algo2_late_rate = -1
            algo2_late_rate0 = -1
            algo2_late_rate1 = -1
            algo2_ETA_usage_rate = -1
            algo2_ETA_usage_rate0 = -1
            algo2_ETA_usage_rate1 = -1
            algo2_var_ETA = 0
            algo2_var0_ETA = 0
            algo2_var1_ETA = 0

            self.writter.add_scalar('Eval Late Order Rate/Algo2 Total', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Hired', algo2_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Crowdsourced', algo2_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total', algo2_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total Var', algo2_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired', algo2_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired Var', algo2_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Crowdsourced', algo2_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Crowdsourced Var', algo2_var1_ETA, self.eval_num)
            
            message += "No order is dropped in Algo2\n"
        else:
            if algo2_count_dropped_orders0:                
                algo2_late_rate0 = round(algo2_late_orders0 / algo2_count_dropped_orders0, 2)
                algo2_ETA_usage_rate0 = round(np.mean(algo2_ETA_usage0), 2)
                algo2_var0_ETA = round(np.var(algo2_ETA_usage0), 2)
            else:
                algo2_late_rate0 = -1
                algo2_ETA_usage_rate0 = -1
                algo2_var0_ETA = 0
                
            if algo2_count_dropped_orders1:                
                algo2_late_rate1 = round(algo2_late_orders1 / algo2_count_dropped_orders1, 2)
                algo2_ETA_usage_rate1 = round(np.mean(algo2_ETA_usage1), 2)
                algo2_var1_ETA = round(np.var(algo2_ETA_usage1), 2)
            else:
                algo2_late_rate1 = -1
                algo2_ETA_usage_rate1 = -1
                algo2_var1_ETA = 0
                
            algo2_late_rate = round((algo2_late_orders0 + algo2_late_orders1) / (algo2_count_dropped_orders0 +algo2_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate}")

            algo2_ETA_usage_rate = round(np.mean(algo2_ETA_usage0 + algo2_ETA_usage1), 2)
            algo2_var_ETA = round(np.var(algo2_ETA_usage0 + algo2_ETA_usage1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo2: Hired - {algo2_ETA_usage_rate0} (Var: {algo2_var0_ETA}), Crowdsourced - {algo2_ETA_usage_rate1} (Var: {algo2_var1_ETA}), Total - {algo2_ETA_usage_rate} (Var: {algo2_var_ETA})")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Total', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Hired', algo2_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Crowdsourced', algo2_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total', algo2_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired', algo2_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Crowdsourced', algo2_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total Var', algo2_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired Var', algo2_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Crowdsourced Var', algo2_var1_ETA, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo2: Hired - {algo2_ETA_usage_rate0} (Var: {algo2_var0_ETA}), Crowdsourced - {algo2_ETA_usage_rate1} (Var: {algo2_var1_ETA}), Total - {algo2_ETA_usage_rate} (Var: {algo2_var_ETA})\n"

        if algo3_count_dropped_orders0 + algo3_count_dropped_orders1 == 0:
            print("No order is dropped in Algo3")
            algo3_late_rate = -1
            algo3_late_rate0 = -1
            algo3_late_rate1 = -1
            algo3_ETA_usage_rate = -1
            algo3_ETA_usage_rate0 = -1
            algo3_ETA_usage_rate1 = -1
            algo3_var_ETA = 0
            algo3_var0_ETA = 0
            algo3_var1_ETA = 0

            self.writter.add_scalar('Eval Late Order Rate/Algo3 Total', algo3_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Hired', algo3_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Crowdsourced', algo3_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total', algo3_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total Var', algo3_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired', algo3_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired Var', algo3_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced', algo3_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced Var', algo3_var1_ETA, self.eval_num)
            
            message += "No order is dropped in Algo3\n"
        else:
            if algo3_count_dropped_orders0:                
                algo3_late_rate0 = round(algo3_late_orders0 / algo3_count_dropped_orders0, 2)
                algo3_ETA_usage_rate0 = round(np.mean(algo3_ETA_usage0), 2)
                algo3_var0_ETA = round(np.var(algo3_ETA_usage0), 2)
            else:
                algo3_late_rate0 = -1
                algo3_ETA_usage_rate0 = -1
                algo3_var0_ETA = 0
                
            if algo3_count_dropped_orders1:                
                algo3_late_rate1 = round(algo3_late_orders1 / algo3_count_dropped_orders1, 2)
                algo3_ETA_usage_rate1 = round(np.mean(algo3_ETA_usage1), 2)
                algo3_var1_ETA = round(np.var(algo3_ETA_usage1), 2)
            else:
                algo3_late_rate1 = -1
                algo3_ETA_usage_rate1 = -1
                algo3_var1_ETA = 0
                
            algo3_late_rate = round((algo3_late_orders0 + algo3_late_orders1) / (algo3_count_dropped_orders0 +algo3_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate}")

            algo3_ETA_usage_rate = round(np.mean(algo3_ETA_usage0 + algo3_ETA_usage1), 2)
            algo3_var_ETA = round(np.var(algo3_ETA_usage0 + algo3_ETA_usage1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo3: Hired - {algo3_ETA_usage_rate0} (Var: {algo3_var0_ETA}), Crowdsourced - {algo3_ETA_usage_rate1} (Var: {algo3_var1_ETA}), Total - {algo3_ETA_usage_rate} (Var: {algo3_var_ETA})")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Total', algo3_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Hired', algo3_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Crowdsourced', algo3_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total', algo3_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired', algo3_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced', algo3_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total Var', algo3_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired Var', algo3_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced Var', algo3_var1_ETA, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo3: Hired - {algo3_ETA_usage_rate0} (Var: {algo3_var0_ETA}), Crowdsourced - {algo3_ETA_usage_rate1} (Var: {algo3_var1_ETA}), Total - {algo3_ETA_usage_rate} (Var: {algo3_var_ETA})\n"

        if algo4_count_dropped_orders0 + algo4_count_dropped_orders1 == 0:
            print("No order is dropped in Algo4")
            algo4_late_rate = -1
            algo4_late_rate0 = -1
            algo4_late_rate1 = -1
            algo4_ETA_usage_rate = -1
            algo4_ETA_usage_rate0 = -1
            algo4_ETA_usage_rate1 = -1
            algo4_var_ETA = 0
            algo4_var0_ETA = 0
            algo4_var1_ETA = 0

            self.writter.add_scalar('Eval Late Order Rate/Algo4 Total', algo4_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo4 Hired', algo4_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo4 Crowdsourced', algo4_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Total', algo4_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Total Var', algo4_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Hired', algo4_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Hired Var', algo4_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Crowdsourced', algo4_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Crowdsourced Var', algo4_var1_ETA, self.eval_num)
            
            message += "No order is dropped in Algo4\n"
        else:
            if algo4_count_dropped_orders0:                
                algo4_late_rate0 = round(algo4_late_orders0 / algo4_count_dropped_orders0, 2)
                algo4_ETA_usage_rate0 = round(np.mean(algo4_ETA_usage0), 2)
                algo4_var0_ETA = round(np.var(algo4_ETA_usage0), 2)
            else:
                algo4_late_rate0 = -1
                algo4_ETA_usage_rate0 = -1
                algo4_var0_ETA = 0
                
            if algo4_count_dropped_orders1:                
                algo4_late_rate1 = round(algo4_late_orders1 / algo4_count_dropped_orders1, 2)
                algo4_ETA_usage_rate1 = round(np.mean(algo4_ETA_usage1), 2)
                algo4_var1_ETA = round(np.var(algo4_ETA_usage1), 2)
            else:
                algo4_late_rate1 = -1
                algo4_ETA_usage_rate1 = -1
                algo4_var1_ETA = 0
                
            algo4_late_rate = round((algo4_late_orders0 + algo4_late_orders1) / (algo4_count_dropped_orders0 +algo4_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo4: Hired - {algo4_late_rate0}, Crowdsourced - {algo4_late_rate1}, Total - {algo4_late_rate}")

            algo4_ETA_usage_rate = round(np.mean(algo4_ETA_usage0 + algo4_ETA_usage1), 2)
            algo4_var_ETA = round(np.var(algo4_ETA_usage0 + algo4_ETA_usage1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo4: Hired - {algo4_ETA_usage_rate0} (Var: {algo4_var0_ETA}), Crowdsourced - {algo4_ETA_usage_rate1} (Var: {algo4_var1_ETA}), Total - {algo4_ETA_usage_rate} (Var: {algo4_var_ETA})")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo4 Total', algo4_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo4 Hired', algo4_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo4 Crowdsourced', algo4_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Total', algo4_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Hired', algo4_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Crowdsourced', algo4_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Total Var', algo4_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Hired Var', algo4_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo4 Crowdsourced Var', algo4_var1_ETA, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo4: Hired - {algo4_late_rate0}, Crowdsourced - {algo4_late_rate1}, Total - {algo4_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo4: Hired - {algo4_ETA_usage_rate0} (Var: {algo4_var0_ETA}), Crowdsourced - {algo4_ETA_usage_rate1} (Var: {algo4_var1_ETA}), Total - {algo4_ETA_usage_rate} (Var: {algo4_var_ETA})\n"

        logger.success(message)
            
        print("\n")
        
        return (
            algo1_eval_episode_rewards_sum,
            algo2_eval_episode_rewards_sum,
            algo3_eval_episode_rewards_sum,
            algo4_eval_episode_rewards_sum,
            
            algo1_distance0,
            algo1_distance1,
            algo1_distance,
            algo2_distance0,
            algo2_distance1,
            algo2_distance,
            algo3_distance0,
            algo3_distance1,
            algo3_distance,
            algo4_distance0,
            algo4_distance1,
            algo4_distance,
            algo1_var0_distance,
            algo1_var1_distance,
            algo1_var_distance,
            algo2_var_distance,
            algo2_var0_distance,
            algo2_var1_distance,
            algo3_var_distance,
            algo3_var0_distance,
            algo3_var1_distance,
            algo4_var_distance,
            algo4_var0_distance,
            algo4_var1_distance,
            
            algo1_avg0_speed,
            algo1_avg1_speed,
            algo1_avg_speed,
            algo2_avg0_speed,
            algo2_avg1_speed,
            algo2_avg_speed,
            algo3_avg0_speed,
            algo3_avg1_speed,
            algo3_avg_speed,
            algo4_avg0_speed,
            algo4_avg1_speed,
            algo4_avg_speed,
            algo1_var0_speed,
            algo1_var1_speed,
            algo1_var_speed,
            algo2_var0_speed,
            algo2_var1_speed,
            algo2_var_speed,
            algo3_var0_speed,
            algo3_var1_speed,
            algo3_var_speed,
            algo4_var0_speed,
            algo4_var1_speed,
            algo4_var_speed,
            
            algo1_overspeed0,
            algo1_overspeed1,
            algo1_overspeed,
            algo2_overspeed0,
            algo2_overspeed1,
            algo2_overspeed,
            algo3_overspeed0,
            algo3_overspeed1,
            algo3_overspeed,
            algo4_overspeed0,
            algo4_overspeed1,
            algo4_overspeed,
            
            # algo1_reject_rate_per_episode,
            # algo2_reject_rate_per_episode,
            # algo3_reject_rate_per_episode,
            # algo4_reject_rate_per_episode,
            
            # algo1_reject0,
            # algo1_reject1,
            # algo1_reject,
            # algo2_reject0,
            # algo2_reject1,
            # algo2_reject,
            # algo3_reject0,
            # algo3_reject1,
            # algo3_reject,
            # algo4_reject0,
            # algo4_reject1,
            # algo4_reject,
            # algo1_var0_reject,
            # algo1_var1_reject,
            # algo1_var_reject,
            # algo2_var0_reject,
            # algo2_var1_reject,
            # algo2_var_reject,
            # algo3_var0_reject,
            # algo3_var1_reject,
            # algo3_var_reject,
            # algo4_var0_reject,
            # algo4_var1_reject,
            # algo4_var_reject,
            
            algo1_price_per_order0,
            algo1_price_per_order1,
            algo1_price_per_order,
            algo2_price_per_order0,
            algo2_price_per_order1,
            algo2_price_per_order,
            algo3_price_per_order0,
            algo3_price_per_order1,
            algo3_price_per_order,
            algo4_price_per_order0,
            algo4_price_per_order1,
            algo4_price_per_order,
            algo1_var0_price,
            algo1_var1_price,
            algo1_var_price,
            algo2_var0_price,
            algo2_var1_price,
            algo2_var_price,
            algo3_var0_price,
            algo3_var1_price,
            algo3_var_price,
            algo4_var0_price,
            algo4_var1_price,
            algo4_var_price,
            
            algo1_income0,
            algo1_income1,
            algo1_income,
            platform_cost1,
            algo2_income0,
            algo2_income1,
            algo2_income,
            platform_cost2,
            algo3_income0,
            algo3_income1,
            algo3_income,
            platform_cost3,
            algo4_income0,
            algo4_income1,
            algo4_income,
            platform_cost4,
            algo1_var0_income,
            algo1_var1_income,
            algo1_var_income,
            algo2_var0_income,
            algo2_var1_income,
            algo2_var_income,
            algo3_var0_income,
            algo3_var1_income,
            algo3_var_income,
            algo4_var0_income,
            algo4_var1_income,
            algo4_var_income,
            
            algo1_finish0,
            algo1_finish1,
            algo1_finish,
            algo2_finish0,
            algo2_finish1,
            algo2_finish,
            algo3_finish0,
            algo3_finish1,
            algo3_finish,
            algo4_finish0,
            algo4_finish1,
            algo4_finish,
            algo1_var0_finish,
            algo1_var1_finish,
            algo1_var_finish,
            algo2_var0_finish,
            algo2_var1_finish,
            algo2_var_finish,
            algo3_var0_finish,
            algo3_var1_finish,
            algo3_var_finish,
            algo4_var0_finish,
            algo4_var1_finish,
            algo4_var_finish,
            
            algo1_avg0_leisure,
            algo1_avg1_leisure,
            algo1_avg_leisure,
            algo2_avg0_leisure,
            algo2_avg1_leisure,
            algo2_avg_leisure,
            algo3_avg0_leisure,
            algo3_avg1_leisure,
            algo3_avg_leisure,
            algo4_avg0_leisure,
            algo4_avg1_leisure,
            algo4_avg_leisure,
            algo1_var0_leisure,
            algo1_var1_leisure,
            algo1_var_leisure,
            algo2_var0_leisure,
            algo2_var1_leisure,
            algo2_var_leisure,
            algo3_var0_leisure,
            algo3_var1_leisure,
            algo3_var_leisure,
            algo4_var0_leisure,
            algo4_var1_leisure,
            algo4_var_leisure,
            
            algo1_avg0_running,
            algo1_avg1_running,
            algo1_avg_running,
            algo2_avg0_running,
            algo2_avg1_running,
            algo2_avg_running,
            algo3_avg0_running,
            algo3_avg1_running,
            algo3_avg_running,
            algo4_avg0_running,
            algo4_avg1_running,
            algo4_avg_running,
            algo1_var0_running,
            algo1_var1_running,
            algo1_var_running,
            algo2_var0_running,
            algo2_var1_running,
            algo2_var_running,
            algo3_var0_running,
            algo3_var1_running,
            algo3_var_running,
            algo4_var0_running,
            algo4_var1_running,
            algo4_var_running,
            
            algo1_late_rate0,
            algo1_late_rate1,
            algo1_late_rate,
            algo2_late_rate0,
            algo2_late_rate1,
            algo2_late_rate,
            algo3_late_rate0,
            algo3_late_rate1,
            algo3_late_rate,
            algo4_late_rate0,
            algo4_late_rate1,
            algo4_late_rate,
            
            algo1_ETA_usage_rate0,
            algo1_ETA_usage_rate1,
            algo1_ETA_usage_rate,
            algo2_ETA_usage_rate0,
            algo2_ETA_usage_rate1,
            algo2_ETA_usage_rate,
            algo3_ETA_usage_rate0,
            algo3_ETA_usage_rate1,
            algo3_ETA_usage_rate,
            algo4_ETA_usage_rate0,
            algo4_ETA_usage_rate1,
            algo4_ETA_usage_rate,
            algo1_var0_ETA,
            algo1_var1_ETA,
            algo1_var_ETA,
            algo2_var0_ETA,
            algo2_var1_ETA,
            algo2_var_ETA,
            algo3_var0_ETA,
            algo3_var1_ETA,
            algo3_var_ETA,
            algo4_var0_ETA,
            algo4_var1_ETA,
            algo4_var_ETA

        )