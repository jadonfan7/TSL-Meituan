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
            
            obs = self.envs.reset(episode % 4)
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
                
                # available_actions = self.envs.get_available_actions()
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
                        if c.travel_distance > 0:
                            Hired_distance_per_episode.append(c.travel_distance)
                        Hired_finish_num.append(c.finish_order_num)
                        Hired_leisure_time.append(c.total_leisure_time)
                        Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            Hired_avg_speed.append(c.avg_speed)
                        Hired_income.append(c.income)
                    else:
                        Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            Crowdsourced_distance_per_episode.append(c.travel_distance)
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
            print(f"There are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced with {Crowdsourced_on / self.envs.num_envs} ({Crowdsourced_on / Crowdsourced_num}) on, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1, {order_wait / self.envs.num_envs} ({round(100 * order_wait / (order_wait + order0_num + order1_num), 2)}%) Orders waiting to be paired")                
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
            overspeed_penalty = np.floor((count_overspeed0 + count_overspeed1) / self.envs.num_envs) * 50
            print(f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total rate - {overspeed}, Overspeed penalty - {overspeed_penalty}")
            self.writter.add_scalar('Overspeed Rate/Total rate', overspeed, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Hired', overspeed0, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Crowdsourced', overspeed1, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Overspeed Penalty', overspeed_penalty, episode + 1)
                        
            # ---------------------
            # average order price for courier
            price_per_order0 = round(np.mean(order0_price), 2)
            var_price0 = round(np.var(order0_price), 2)
            price_per_order1 = round(np.mean(order1_price), 2)
            var_price1 = round(np.var(order0_price), 2)
            price_per_order = round(np.mean(order0_price + order1_price), 2)
            var_price = round(np.var(order0_price + order1_price), 2)
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
            print(f"Hired running time is {avg0_running} minutes (Var: {var_running0}), Crowdsourced running time is {avg1_running} minutes (Var: {var_running1}) and Total running time per courier is {avg_running} minutes (Var: {var_running})")
            self.writter.add_scalar('Average running Time/Total', avg_running, episode + 1)
            self.writter.add_scalar('Average running Time/Total_Var', var_running, episode + 1)
            self.writter.add_scalar('Average running Time/Hired', avg0_running, episode + 1)
            self.writter.add_scalar('Average running Time/Hired_Var', var_running0, episode + 1)
            self.writter.add_scalar('Average running Time/Crowdsourced', avg1_running, episode + 1)
            self.writter.add_scalar('Average running Time/Crowdsourced_Var', var_running1, episode + 1)
            
            message = (
                f"\nThis is Train Episode {episode+1}\n"
                f"There are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1, {order_wait / self.envs.num_envs} ({round(100 * order_wait / (order_wait + order0_num + order1_num), 2)}%) Orders waiting to be paired\n"
                f"Average Travel Distance for Episode {episode+1}: Hired ({len(Hired_distance_per_episode)}) - {distance0} km (Var: {distance_var0}), Crowdsourced ({len(Crowdsourced_distance_per_episode)}) - {distance1} km (Var: {distance_var1}), Total ({len(Hired_distance_per_episode+Crowdsourced_distance_per_episode)}) - {distance} km (Var: {distance_var})\n"
                f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}\n"
                f"The average speed for Episode {episode+1}: Hired ({len(Hired_avg_speed)}) - {avg0_speed} m/s (Var: {var0_speed}), Crowdsourced ({len(Crowdsourced_avg_speed)}) - {avg1_speed} m/s (Var: {var1_speed}), Total ({len(Hired_avg_speed+Crowdsourced_avg_speed)}) - {avg_speed} m/s (Var: {var_speed})\n"
                f"Rate of Overspeed for Episode {episode+1}: Hired ({num_active_Hired}) - {overspeed0}, Crowdsourced ({num_active_Crowdsourced}) - {overspeed1}, Total ({num_active_Hired+num_active_Crowdsourced}) - {overspeed}\n"
                f"The average price for Episode {episode+1}: Hired ({len(order0_price)}) - {price_per_order0} dollar (Var: {var_price0}) with {order0_num} orders, Crowdsourced ({len(order1_price)}) - {price_per_order1} dollar (Var: {var_price1}) with {order1_num} orders, Total ({len(order0_price+order1_price)}) - {price_per_order} dollar (Var: {var_price})\n"
                f"The average income for Episode {episode+1}: Hired ({len(Hired_income)}) - {income0} dollar (Var: {var_income0}), Crowdsourced ({len(Crowdsourced_income)}) - {income1} dollar (Var: {var_income1}), Total ({len(Hired_income+Crowdsourced_income)}) - {income} dollar (Var: {var_income})\n"
                f"The platform total cost is {platform_cost} dollar\n"
                f"The average finish number for Episode {episode+1}: Hired ({len(Hired_finish_num)}) - {avg_finish0} (Var: {var_finish0}), Crowdsourced ({len(Crowdsourced_finish_num)}) - {avg_finish1} (Var: {var_finish1}), Total ({len(Hired_finish_num+Crowdsourced_finish_num)}) - {avg_finish} (Var: {var_finish})\n"
                f"The average leisure time for Episode {episode+1}: Hired ({len(Hired_leisure_time)}) - {avg0_leisure} minutes (Var: {var_leisure0}), Crowdsourced ({len(Crowdsourced_leisure_time)}) - {avg1_leisure} minutes (Var: {var_leisure1}), Total ({len(Hired_leisure_time+Crowdsourced_leisure_time)}) - {avg_leisure} minutes (Var: {var_leisure})\n"
                f"The average running time for Episode {episode+1}: Hired ({len(Hired_running_time)}) - {avg0_running} minutes (Var: {var_running0}), Crowdsourced ({len(Crowdsourced_running_time)}) - {avg1_running} minutes (Var: {var_running1}), Total ({len(Hired_running_time+Crowdsourced_running_time)}) - {avg_running} minutes (Var: {var_running})\n"
            )

            if count_dropped_orders0 + count_dropped_orders1 == 0:
                print("No order is dropped in this episode")
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

                ETA_usage_rate = round(np.mean(ETA_usage0 + ETA_usage1), 2)
                Var_ETA = round(np.var(ETA_usage0 + ETA_usage1), 2)
                print(f"Rate of ETA Usage for Episode {episode+1}: Hired - {ETA_usage_rate0} (Var: {Var_ETA0}), Crowdsourced - {ETA_usage_rate1} (Var: {Var_ETA1}), Total - {ETA_usage_rate} (Var: {Var_ETA})")
                
                message += f"Rate of Late Orders for Episode {episode+1}: Hired - {late_rate0} out of {count_dropped_orders0} orders, Crowdsourced - {late_rate1} out of {count_dropped_orders1} orders, Total - {late_rate} out of {count_dropped_orders0 + count_dropped_orders1} orders\n" + f"Rate of ETA Usage for Episode {episode+1}: Hired - {ETA_usage_rate0} (Var: {Var_ETA0}), Crowdsourced - {ETA_usage_rate1} (Var: {Var_ETA1}), Total - {ETA_usage_rate} (Var: {Var_ETA})\n"
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

            # social_welfare = sum(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
            # print(f"Social welfare is {social_welfare} dollar\n")
            # message += f"Social welfare is {social_welfare} dollar\n"
            # logger.success(message)
            # self.writter.add_scalar('Social Welfare', social_welfare, episode + 1)
            
            print("\n")      

            # compute return and update nrk
            self.compute()
            # train_infos = self.train()

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
                                
                self.eval(total_num_steps)


        self.writter.close()
        
    @torch.no_grad()
    # def collect(self, step, available_actions):
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
        
        eval_obs = self.eval_envs.reset(4)
        # eval_obs = self.eval_envs.reset(5)
        
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
        algo5_Hired_num = 0
        algo5_Crowdsourced_num = 0
        algo5_Crowdsourced_on = 0
        
        algo1_eval_episode_rewards_sum = 0
        algo2_eval_episode_rewards_sum = 0
        algo3_eval_episode_rewards_sum = 0
        algo4_eval_episode_rewards_sum = 0
        algo5_eval_episode_rewards_sum = 0

        algo1_Hired_distance_per_episode = []
        algo1_Crowdsourced_distance_per_episode = []
        algo2_Hired_distance_per_episode = []
        algo2_Crowdsourced_distance_per_episode = []
        algo3_Hired_distance_per_episode = []
        algo3_Crowdsourced_distance_per_episode = []
        algo4_Hired_distance_per_episode = []
        algo4_Crowdsourced_distance_per_episode = []
        algo5_Hired_distance_per_episode = []
        algo5_Crowdsourced_distance_per_episode = []

        algo1_count_overspeed0 = 0
        algo1_count_overspeed1 = 0
        algo2_count_overspeed0 = 0
        algo2_count_overspeed1 = 0
        algo3_count_overspeed0 = 0
        algo3_count_overspeed1 = 0
        algo4_count_overspeed0 = 0
        algo4_count_overspeed1 = 0
        algo5_count_overspeed0 = 0
        algo5_count_overspeed1 = 0

        algo1_num_active_couriers0 = 0
        algo1_num_active_couriers1 = 0
        algo2_num_active_couriers0 = 0
        algo2_num_active_couriers1 = 0
        algo3_num_active_couriers0 = 0
        algo3_num_active_couriers1 = 0
        algo4_num_active_couriers0 = 0
        algo4_num_active_couriers1 = 0
        algo5_num_active_couriers0 = 0
        algo5_num_active_couriers1 = 0

        algo1_late_orders0 = 0
        algo1_late_orders1 = 0
        algo2_late_orders0 = 0
        algo2_late_orders1 = 0
        algo3_late_orders0 = 0
        algo3_late_orders1 = 0
        algo4_late_orders0 = 0
        algo4_late_orders1 = 0
        algo5_late_orders0 = 0
        algo5_late_orders1 = 0

        algo1_ETA_usage0 = []
        algo1_ETA_usage1 = []
        algo2_ETA_usage0 = []
        algo2_ETA_usage1 = []
        algo3_ETA_usage0 = []
        algo3_ETA_usage1 = []
        algo4_ETA_usage0 = []
        algo4_ETA_usage1 = []
        algo5_ETA_usage0 = []
        algo5_ETA_usage1 = []

        algo1_count_dropped_orders0 = 0
        algo1_count_dropped_orders1 = 0
        algo2_count_dropped_orders0 = 0
        algo2_count_dropped_orders1 = 0
        algo3_count_dropped_orders0 = 0
        algo3_count_dropped_orders1 = 0
        algo4_count_dropped_orders0 = 0
        algo4_count_dropped_orders1 = 0
        algo5_count_dropped_orders0 = 0
        algo5_count_dropped_orders1 = 0

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
        algo5_order0_price = []
        algo5_order1_price = []
        algo5_order0_num = 0
        algo5_order1_num = 0
        algo5_order_wait = 0

        platform_cost1 = 0
        platform_cost2 = 0
        platform_cost3 = 0
        platform_cost4 = 0
        platform_cost5 = 0

        algo1_Hired_finish_num = []
        algo1_Crowdsourced_finish_num = []
        algo2_Hired_finish_num = []
        algo2_Crowdsourced_finish_num = []
        algo3_Hired_finish_num = []
        algo3_Crowdsourced_finish_num = []
        algo4_Hired_finish_num = []
        algo4_Crowdsourced_finish_num = []
        algo5_Hired_finish_num = []
        algo5_Crowdsourced_finish_num = []

        algo1_Hired_leisure_time = []
        algo1_Crowdsourced_leisure_time = []
        algo2_Hired_leisure_time = []
        algo2_Crowdsourced_leisure_time = []
        algo3_Hired_leisure_time = []
        algo3_Crowdsourced_leisure_time = []
        algo4_Hired_leisure_time = []
        algo4_Crowdsourced_leisure_time = []
        algo5_Hired_leisure_time = []
        algo5_Crowdsourced_leisure_time = []

        algo1_Hired_running_time = []
        algo1_Crowdsourced_running_time = []
        algo2_Hired_running_time = []
        algo2_Crowdsourced_running_time = []
        algo3_Hired_running_time = []
        algo3_Crowdsourced_running_time = []
        algo4_Hired_running_time = []
        algo4_Crowdsourced_running_time = []
        algo5_Hired_running_time = []
        algo5_Crowdsourced_running_time = []

        algo1_Hired_avg_speed = []
        algo1_Crowdsourced_avg_speed = []
        algo2_Hired_avg_speed = []
        algo2_Crowdsourced_avg_speed = []
        algo3_Hired_avg_speed = []
        algo3_Crowdsourced_avg_speed = []
        algo4_Hired_avg_speed = []
        algo4_Crowdsourced_avg_speed = []
        algo5_Hired_avg_speed = []
        algo5_Crowdsourced_avg_speed = []

        algo1_Hired_income = []
        algo1_Crowdsourced_income = []
        algo2_Hired_income = []
        algo2_Crowdsourced_income = []
        algo3_Hired_income = []
        algo3_Crowdsourced_income = []
        algo4_Hired_income = []
        algo4_Crowdsourced_income = []
        algo5_Hired_income = []
        algo5_Crowdsourced_income = []

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
            
            # print("-"*25)
            # print(f"THIS IS EVAL STEP {eval_step}")

            for i in range(self.eval_envs.num_envs):
                
                # print(f"ENVIRONMENT {i+1}")

                # print("Couriers:")
                # for c in self.eval_envs.envs_discrete[i].couriers:
                #     if c.state == 'active':
                #         print(c)
                # print("Orders:")
                # for o in self.eval_envs.envs_discrete[i].orders:
                #     print(o)  
                # print("\n")
                
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
            algo5_eval_episode_rewards_sum += sum(eval_rewards[4])

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
                elif i == 3:
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
                else:
                    for c in self.eval_envs.envs_discrete[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                algo5_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo5_count_overspeed0 += 1
                            else:
                                algo5_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo5_count_overspeed1 += 1

                                
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
                        algo1_Hired_finish_num.append(c.finish_order_num)
                        algo1_Hired_leisure_time.append(c.total_leisure_time)
                        algo1_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo1_Hired_avg_speed.append(c.avg_speed)
                        algo1_Hired_income.append(c.income)
                    else:
                        algo1_Crowdsourced_num += 1
                        algo1_Crowdsourced_distance_per_episode.append(c.travel_distance)
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
                        algo2_Hired_finish_num.append(c.finish_order_num)
                        algo2_Hired_leisure_time.append(c.total_leisure_time)
                        algo2_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo2_Hired_avg_speed.append(c.avg_speed)
                        algo2_Hired_income.append(c.income)
                    else:
                        algo2_Crowdsourced_num += 1
                        algo2_Crowdsourced_distance_per_episode.append(c.travel_distance)
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
                        algo3_Hired_finish_num.append(c.finish_order_num)
                        algo3_Hired_leisure_time.append(c.total_leisure_time)
                        algo3_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo3_Hired_avg_speed.append(c.avg_speed)
                        algo3_Hired_income.append(c.income)
                    else:
                        algo3_Crowdsourced_num += 1
                        algo3_Crowdsourced_distance_per_episode.append(c.travel_distance)
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
                                                
                    if o.status == 'wait_pair':
                        algo3_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo3_order0_price.append(o.price)
                            algo3_order0_num += 1
                        else:
                            algo3_order1_price.append(o.price)
                            algo3_order1_num += 1  
            elif i == 3:
                platform_cost4 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo4_Hired_num += 1
                        algo4_Hired_distance_per_episode.append(c.travel_distance)
                        algo4_Hired_finish_num.append(c.finish_order_num)
                        algo4_Hired_leisure_time.append(c.total_leisure_time)
                        algo4_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo4_Hired_avg_speed.append(c.avg_speed)
                        algo4_Hired_income.append(c.income)
                    else:
                        algo4_Crowdsourced_num += 1
                        algo4_Crowdsourced_distance_per_episode.append(c.travel_distance)
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

                    if o.status == 'wait_pair':
                        algo4_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo4_order0_price.append(o.price)
                            algo4_order0_num += 1
                        else:
                            algo4_order1_price.append(o.price)
                            algo4_order1_num += 1   
                            
            else:
                platform_cost5 += self.eval_envs.envs_discrete[i].platform_cost
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo5_Hired_num += 1
                        algo5_Hired_distance_per_episode.append(c.travel_distance)
                        algo5_Hired_finish_num.append(c.finish_order_num)
                        algo5_Hired_leisure_time.append(c.total_leisure_time)
                        algo5_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo5_Hired_avg_speed.append(c.avg_speed)
                        algo5_Hired_income.append(c.income)
                    else:
                        algo5_Crowdsourced_num += 1
                        algo5_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo5_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo5_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo5_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo5_Crowdsourced_avg_speed.append(c.avg_speed)
                        algo5_Crowdsourced_income.append(c.income)
                        if c.state == 'active':
                            algo5_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo5_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo5_late_orders0 += 1
                            else:
                                algo5_ETA_usage0.append(o.ETA_usage)
                        else:
                            algo5_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo5_late_orders1 += 1
                            else:
                                algo5_ETA_usage1.append(o.ETA_usage)
                                                
                    if o.status == 'wait_pair':
                        algo5_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo5_order0_price.append(o.price)
                            algo5_order0_num += 1
                        else:
                            algo5_order1_price.append(o.price)
                            algo5_order1_num += 1   
            
                            
        print(f"\nIn Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} ({algo1_Crowdsourced_on / algo1_Crowdsourced_num}) on, {algo1_order0_num} Order0, {algo1_order1_num} Order1, {algo1_order_wait} ({round(100 * algo1_order_wait / (algo1_order_wait + algo1_order0_num + algo1_order1_num), 2)}%) Orders waiting to be paired")
        print(f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} ({algo2_Crowdsourced_on / algo2_Crowdsourced_num}) on, {algo2_order0_num} Order0, {algo2_order1_num} Order1, {algo2_order_wait} ({round(100 * algo2_order_wait / (algo2_order_wait + algo2_order0_num + algo2_order1_num), 2)}%) Orders waiting to be paired")  
        print(f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} ({algo3_Crowdsourced_on / algo3_Crowdsourced_num}) on, {algo3_order0_num} Order0, {algo3_order1_num} Order1, {algo3_order_wait} ({round(100 * algo3_order_wait / (algo3_order_wait + algo3_order0_num + algo3_order1_num), 2)}%) Orders waiting to be paired")       
        print(f"In Algo4 there are {algo4_Hired_num} Hired, {algo4_Crowdsourced_num} Crowdsourced with {algo4_Crowdsourced_on} ({algo4_Crowdsourced_on / algo4_Crowdsourced_num}) on, {algo4_order0_num} Order0, {algo4_order1_num} Order1, {algo4_order_wait} ({round(100 * algo4_order_wait / (algo4_order_wait + algo4_order0_num + algo4_order1_num), 2)}%) Orders waiting to be paired")
        print(f"In Algo5 there are {algo5_Hired_num} Hired, {algo5_Crowdsourced_num} Crowdsourced with {algo5_Crowdsourced_on} {algo5_Crowdsourced_on / algo5_Crowdsourced_num} on, {algo5_order0_num} Order0, {algo5_order1_num} Order1, {algo5_order_wait} ({round(100 * algo5_order_wait / (algo5_order_wait + algo5_order0_num + algo5_order1_num), 2)}%) Orders waiting to be paired")

        # -----------------------
        # Reward
        print(f"Total Reward for Evaluation Between Algos:\nAlgo1: {round(algo1_eval_episode_rewards_sum, 2)}\nAlgo2: {round(algo2_eval_episode_rewards_sum, 2)}\nAlgo3: {round(algo3_eval_episode_rewards_sum, 2)}\nAlgo4: {round(algo4_eval_episode_rewards_sum, 2)}\nAlgo5: {round(algo5_eval_episode_rewards_sum, 2)}")
        self.writter.add_scalar('Eval Reward/Algo1', algo1_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo2', algo2_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo3', algo3_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo4', algo4_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo5', algo5_eval_episode_rewards_sum, self.eval_num)

        # -----------------------
        # Distance
        algo1_distance0 = round(np.mean(algo1_Hired_distance_per_episode) / 1000, 2)
        algo1_var0_distance = round(np.var(algo1_Hired_distance_per_episode) / 1000000, 2)
        algo1_distance1 = round(np.mean(algo1_Crowdsourced_distance_per_episode) / 1000, 2)
        algo1_var1_distance = round(np.var(algo1_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo1_distance = round(np.mean(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000, 2)
        algo1_var_distance = round(np.var(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo1_distance_courier_num = len(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode)
        
        algo2_distance0 = round(np.mean(algo2_Hired_distance_per_episode) / 1000, 2)
        algo2_var0_distance = round(np.var(algo2_Hired_distance_per_episode) / 1000000, 2)
        algo2_distance1 = round(np.mean(algo2_Crowdsourced_distance_per_episode) / 1000, 2)
        algo2_var1_distance = round(np.var(algo2_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo2_distance = round(np.mean(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000, 2)
        algo2_var_distance = round(np.var(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo2_distance_courier_num = len(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode)

        algo3_distance0 = round(np.mean(algo3_Hired_distance_per_episode) / 1000, 2)
        algo3_var0_distance = round(np.var(algo3_Hired_distance_per_episode) / 1000000, 2)
        algo3_distance1 = round(np.mean(algo3_Crowdsourced_distance_per_episode) / 1000, 2)
        algo3_var1_distance = round(np.var(algo3_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo3_distance = round(np.mean(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000, 2)
        algo3_var_distance = round(np.var(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo3_distance_courier_num = len(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode)

        algo4_distance0 = round(np.mean(algo4_Hired_distance_per_episode) / 1000, 2)
        algo4_var0_distance = round(np.var(algo4_Hired_distance_per_episode) / 1000000, 2)
        algo4_distance1 = round(np.mean(algo4_Crowdsourced_distance_per_episode) / 1000, 2)
        algo4_var1_distance = round(np.var(algo4_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo4_distance = round(np.mean(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000, 2)
        algo4_var_distance = round(np.var(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo4_distance_courier_num = len(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode)

        algo5_distance0 = round(np.mean(algo5_Hired_distance_per_episode) / 1000, 2)
        algo5_var0_distance = round(np.var(algo5_Hired_distance_per_episode) / 1000000, 2)
        algo5_distance1 = round(np.mean(algo5_Crowdsourced_distance_per_episode) / 1000, 2)
        algo5_var1_distance = round(np.var(algo5_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo5_distance = round(np.mean(algo5_Hired_distance_per_episode + algo5_Crowdsourced_distance_per_episode) / 1000, 2)
        algo5_var_distance = round(np.var(algo5_Hired_distance_per_episode + algo5_Crowdsourced_distance_per_episode) / 1000000, 2)
        algo5_distance_courier_num = len(algo5_Hired_distance_per_episode + algo5_Crowdsourced_distance_per_episode)

        print("Average Travel Distance and Var per Courier Between Algos:")
        print(f"Algo1: Hired - {algo1_distance0} km (Var: {algo1_var0_distance}), Crowdsourced - {algo1_distance1} km (Var: {algo1_var1_distance}), Total - {algo1_distance} km (Var: {algo1_var_distance})")
        print(f"Algo2: Hired - {algo2_distance0} km (Var: {algo2_var0_distance}), Crowdsourced - {algo2_distance1} km (Var: {algo2_var1_distance}), Total - {algo2_distance} km (Var: {algo2_var_distance})")
        print(f"Algo3: Hired - {algo3_distance0} km (Var: {algo3_var0_distance}), Crowdsourced - {algo3_distance1} km (Var: {algo3_var1_distance}), Total - {algo3_distance} km (Var: {algo3_var_distance})")
        print(f"Algo4: Hired - {algo4_distance0} km (Var: {algo4_var0_distance}), Crowdsourced - {algo4_distance1} km (Var: {algo4_var1_distance}), Total - {algo4_distance} km (Var: {algo4_var_distance})")
        print(f"Algo5: Hired - {algo5_distance0} km (Var: {algo5_var0_distance}), Crowdsourced - {algo5_distance1} km (Var: {algo5_var1_distance}), Total - {algo5_distance} km (Var: {algo5_var_distance})")

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
        
        self.writter.add_scalar('Eval Travel Distance/Algo5 Hired', algo5_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo5 Crowdsourced', algo5_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo5 Total', algo5_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo5 Hired Var', algo5_var0_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo5 Crowdsourced Var', algo5_var1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo5 Total Var', algo5_var_distance, self.eval_num)
        
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
        
        algo5_avg0_speed = round(np.mean(algo5_Hired_avg_speed), 2)
        algo5_var0_speed = round(np.var(algo5_Hired_avg_speed), 2)
        algo5_avg1_speed = round(np.mean(algo5_Crowdsourced_avg_speed), 2)
        algo5_var1_speed = round(np.var(algo5_Crowdsourced_avg_speed), 2)
        algo5_avg_speed = round(np.mean(algo5_Hired_avg_speed + algo5_Crowdsourced_avg_speed), 2)
        algo5_var_speed = round(np.var(algo5_Hired_avg_speed + algo5_Crowdsourced_avg_speed), 2)
        
        print("Average Speed and Variance per Courier Between Algos:")
        print(f"Algo1: Hired average speed is {algo1_avg0_speed} m/s (Var: {algo1_var0_speed}), Crowdsourced average speed is {algo1_avg1_speed} m/s (Var: {algo1_var1_speed}), Total average speed is {algo1_avg_speed} m/s (Var: {algo1_var_speed})")
        print(f"Algo2: Hired average speed is {algo2_avg0_speed} m/s (Var: {algo2_var0_speed}), Crowdsourced average speed is {algo2_avg1_speed} m/s (Var: {algo2_var1_speed}), Total average speed is {algo2_avg_speed} m/s (Var: {algo2_var_speed})")
        print(f"Algo3: Hired average speed is {algo3_avg0_speed} m/s (Var: {algo3_var0_speed}), Crowdsourced average speed is {algo3_avg1_speed} m/s (Var: {algo3_var1_speed}), Total average speed is {algo3_avg_speed} m/s (Var: {algo3_var_speed})")
        print(f"Algo4: Hired average speed is {algo4_avg0_speed} m/s (Var: {algo4_var0_speed}), Crowdsourced average speed is {algo4_avg1_speed} m/s (Var: {algo4_var1_speed}), Total average speed is {algo4_avg_speed} m/s (Var: {algo4_var_speed})")
        print(f"Algo5: Hired average speed is {algo5_avg0_speed} m/s (Var: {algo5_var0_speed}), Crowdsourced average speed is {algo5_avg1_speed} m/s (Var: {algo5_var1_speed}), Total average speed is {algo5_avg_speed} m/s (Var: {algo5_var_speed})")

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

        self.writter.add_scalar('Eval Average Speed/Algo5 Total', algo5_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo5 Hired', algo5_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo5 Crowdsourced', algo5_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo5 Total Var', algo5_var_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo5 Hired Var', algo5_var0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo5 Crowdsourced Var', algo5_var1_speed, self.eval_num)

        # -----------------------
        # Overspeed
        algo1_overspeed0 = round(algo1_count_overspeed0 / algo1_num_active_couriers0, 2)
        algo1_overspeed1 = round(algo1_count_overspeed1 / algo1_num_active_couriers1, 2)
        algo1_overspeed = round((algo1_count_overspeed0 + algo1_count_overspeed1) / (algo1_num_active_couriers0 + algo1_num_active_couriers1), 2)
        algo1_overspeed_penalty = np.floor(algo1_count_overspeed0 + algo1_count_overspeed1) * 50
        
        algo2_overspeed0 = round(algo2_count_overspeed0 / algo2_num_active_couriers0, 2)
        algo2_overspeed1 = round(algo2_count_overspeed1 / algo2_num_active_couriers1, 2)
        algo2_overspeed = round((algo2_count_overspeed0 + algo2_count_overspeed1) / (algo2_num_active_couriers0 + algo2_num_active_couriers1), 2)
        algo2_overspeed_penalty = np.floor(algo2_count_overspeed0 + algo2_count_overspeed1) * 50

        algo3_overspeed0 = round(algo3_count_overspeed0 / algo3_num_active_couriers0, 2)
        algo3_overspeed1 = round(algo3_count_overspeed1 / algo3_num_active_couriers1, 2)
        algo3_overspeed = round((algo3_count_overspeed0 + algo3_count_overspeed1) / (algo3_num_active_couriers0 + algo3_num_active_couriers1), 2)
        algo3_overspeed_penalty = np.floor(algo3_count_overspeed0 + algo3_count_overspeed1) * 50

        algo4_overspeed0 = round(algo4_count_overspeed0 / algo4_num_active_couriers0, 2)
        algo4_overspeed1 = round(algo4_count_overspeed1 / algo4_num_active_couriers1, 2)
        algo4_overspeed = round((algo4_count_overspeed0 + algo4_count_overspeed1) / (algo4_num_active_couriers0 + algo4_num_active_couriers1), 2)
        algo4_overspeed_penalty = np.floor(algo4_count_overspeed0 + algo4_count_overspeed1) * 50

        algo5_overspeed0 = round(algo5_count_overspeed0 / algo5_num_active_couriers0, 2)
        algo5_overspeed1 = round(algo5_count_overspeed1 / algo5_num_active_couriers1, 2)
        algo5_overspeed = round((algo5_count_overspeed0 + algo5_count_overspeed1) / (algo5_num_active_couriers0 + algo5_num_active_couriers1), 2)
        algo5_overspeed_penalty = np.floor(algo5_count_overspeed0 + algo5_count_overspeed1) * 50

        print("Rate of Overspeed for Evaluation Between Algos:")
        print(f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}, Overspeed penalty - {algo1_overspeed_penalty}")
        print(f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}, Overspeed penalty - {algo2_overspeed_penalty}")
        print(f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}, Overspeed penalty - {algo3_overspeed_penalty}")
        print(f"Algo4: Hired - {algo4_overspeed0}, Crowdsourced - {algo4_overspeed1}, Total rate - {algo4_overspeed}, Overspeed penalty - {algo4_overspeed_penalty}")
        print(f"Algo5: Hired - {algo5_overspeed0}, Crowdsourced - {algo5_overspeed1}, Total rate - {algo5_overspeed}, Overspeed penalty - {algo5_overspeed_penalty}")

        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Total', algo1_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Hired', algo1_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Crowdsourced', algo1_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Overspeed Penalty', algo1_overspeed_penalty, self.eval_num)

        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Total', algo2_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Hired', algo2_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Crowdsourced', algo2_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Overspeed Penalty', algo2_overspeed_penalty, self.eval_num)

        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Total', algo3_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Hired', algo3_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Crowdsourced', algo3_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Overspeed Penalty', algo3_overspeed_penalty, self.eval_num)

        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Total', algo4_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Hired', algo4_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Crowdsourced', algo4_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo4 Overspeed Penalty', algo4_overspeed_penalty, self.eval_num)

        self.writter.add_scalar('Eval Overspeed Rate/Algo5 Total', algo5_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo5 Hired', algo5_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo5 Crowdsourced', algo5_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo5 Overspeed Penalty', algo5_overspeed_penalty, self.eval_num)

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

        algo5_price_per_order0 = round(np.mean(algo5_order0_price), 2)
        algo5_var0_price = round(np.var(algo5_order0_price), 2)
        algo5_price_per_order1 = round(np.mean(algo5_order1_price), 2)
        algo5_var1_price = round(np.var(algo5_order1_price), 2)
        algo5_price_per_order = round(np.mean(algo5_order0_price + algo5_order1_price), 2)
        algo5_var_price = round(np.var(algo5_order0_price + algo5_order1_price), 2)

        print("Average Price per Order for Evaluation Between Algos:")
        print(f"Algo1: Hired average price per order is {algo1_price_per_order0} dollars (Var: {algo1_var0_price}), Crowdsourced is {algo1_price_per_order1} dollars (Var: {algo1_var1_price}), Total average is {algo1_price_per_order} dollars (Var: {algo1_var_price})")
        print(f"Algo2: Hired average price per order is {algo2_price_per_order0} dollars (Var: {algo2_var0_price}), Crowdsourced is {algo2_price_per_order1} dollars (Var: {algo2_var1_price}), Total average is {algo2_price_per_order} dollars (Var: {algo2_var_price})")
        print(f"Algo3: Hired average price per order is {algo3_price_per_order0} dollars (Var: {algo3_var0_price}), Crowdsourced is {algo3_price_per_order1} dollars (Var: {algo3_var1_price}), Total average is {algo3_price_per_order} dollars (Var: {algo3_var_price})")
        print(f"Algo4: Hired average price per order is {algo4_price_per_order0} dollars (Var: {algo4_var0_price}), Crowdsourced is {algo4_price_per_order1} dollars (Var: {algo4_var1_price}), Total average is {algo4_price_per_order} dollars (Var: {algo4_var_price})")
        print(f"Algo5: Hired average price per order is {algo5_price_per_order0} dollars (Var: {algo5_var0_price}), Crowdsourced is {algo5_price_per_order1} dollars (Var: {algo5_var1_price}), Total average is {algo5_price_per_order} dollars (Var: {algo5_var_price})")

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
         
        self.writter.add_scalar('Eval Average Price/Algo5 Total', algo5_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo5 Hired', algo5_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo5 Crowdsourced', algo5_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo5 Total Var', algo5_var_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo5 Hired Var', algo5_var0_price, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo5 Crowdsourced Var', algo5_var1_price, self.eval_num)

        # -----------------------
        # Average Courier Income
        algo1_income0 = round(np.mean(algo1_Hired_income), 2)
        algo1_var0_income = round(np.var(algo1_Hired_income), 2)
        algo1_income1 = round(np.mean(algo1_Crowdsourced_income), 2)
        algo1_var1_income = round(np.var(algo1_Crowdsourced_income), 2)
        algo1_income = round(np.mean(algo1_Hired_income + algo1_Crowdsourced_income), 2)
        algo1_var_income = round(np.var(algo1_Hired_income + algo1_Crowdsourced_income), 2)
        algo1_income_courier_num = len(algo1_Hired_income + algo1_Crowdsourced_income)

        algo2_income0 = round(np.mean(algo2_Hired_income), 2)
        algo2_var0_income = round(np.var(algo2_Hired_income), 2)
        algo2_income1 = round(np.mean(algo2_Crowdsourced_income), 2)
        algo2_var1_income = round(np.var(algo2_Crowdsourced_income), 2)
        algo2_income = round(np.mean(algo2_Hired_income + algo2_Crowdsourced_income), 2)
        algo2_var_income = round(np.var(algo2_Hired_income + algo2_Crowdsourced_income), 2)
        algo2_income_courier_num = len(algo2_Hired_income + algo2_Crowdsourced_income)

        algo3_income0 = round(np.mean(algo3_Hired_income), 2)
        algo3_var0_income = round(np.var(algo3_Hired_income), 2)
        algo3_income1 = round(np.mean(algo3_Crowdsourced_income), 2)
        algo3_var1_income = round(np.var(algo3_Crowdsourced_income), 2)
        algo3_income = round(np.mean(algo3_Hired_income + algo3_Crowdsourced_income), 2)
        algo3_var_income = round(np.var(algo3_Hired_income + algo3_Crowdsourced_income), 2)
        algo3_income_courier_num = len(algo3_Hired_income + algo3_Crowdsourced_income)

        algo4_income0 = round(np.mean(algo4_Hired_income), 2)
        algo4_var0_income = round(np.var(algo4_Hired_income), 2)
        algo4_income1 = round(np.mean(algo4_Crowdsourced_income), 2)
        algo4_var1_income = round(np.var(algo4_Crowdsourced_income), 2)
        algo4_income = round(np.mean(algo4_Hired_income + algo4_Crowdsourced_income), 2)
        algo4_var_income = round(np.var(algo4_Hired_income + algo4_Crowdsourced_income), 2)
        algo4_income_courier_num = len(algo4_Hired_income + algo4_Crowdsourced_income)

        algo5_income0 = round(np.mean(algo5_Hired_income), 2)
        algo5_var0_income = round(np.var(algo5_Hired_income), 2)
        algo5_income1 = round(np.mean(algo5_Crowdsourced_income), 2)
        algo5_var1_income = round(np.var(algo5_Crowdsourced_income), 2)
        algo5_income = round(np.mean(algo5_Hired_income + algo5_Crowdsourced_income), 2)
        algo5_var_income = round(np.var(algo5_Hired_income + algo5_Crowdsourced_income), 2)
        algo5_income_courier_num = len(algo5_Hired_income + algo5_Crowdsourced_income)

        print("Average Income per Courier for Evaluation Between Algos:")
        print(f"Algo1: Hired's average income is {algo1_income0} dollars (Var: {algo1_var0_income}), Crowdsourced's average income is {algo1_income1} dollars (Var: {algo1_var1_income}), Total income per courier is {algo1_income} dollars (Var: {algo1_var_income})")
        print(f"Algo2: Hired's average income is {algo2_income0} dollars (Var: {algo2_var0_income}), Crowdsourced's average income is {algo2_income1} dollars (Var: {algo2_var1_income}), Total income per courier is {algo2_income} dollars (Var: {algo2_var_income})")
        print(f"Algo3: Hired's average income is {algo3_income0} dollars (Var: {algo3_var0_income}), Crowdsourced's average income is {algo3_income1} dollars (Var: {algo3_var1_income}), Total income per courier is {algo3_income} dollars (Var: {algo3_var_income})")
        print(f"Algo4: Hired's average income is {algo4_income0} dollars (Var: {algo4_var0_income}), Crowdsourced's average income is {algo4_income1} dollars (Var: {algo4_var1_income}), Total income per courier is {algo4_income} dollars (Var: {algo4_var_income})")
        print(f"Algo5: Hired's average income is {algo5_income0} dollars (Var: {algo5_var0_income}), Crowdsourced's average income is {algo5_income1} dollars (Var: {algo5_var1_income}), Total income per courier is {algo5_income} dollars (Var: {algo5_var_income})")
        
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

        self.writter.add_scalar('Eval Average Income/Algo5 Total', algo5_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo5 Hired', algo5_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo5 Crowdsourced', algo5_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo5 Total Var', algo5_var_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo5 Hired Var', algo5_var0_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo5 Crowdsourced Var', algo5_var1_income, self.eval_num)
        self.writter.add_scalar('Eval Platform Cost/Algo5', platform_cost5, self.eval_num)

        # -----------------------
        # Average Courier Finishing Number
        algo1_finish0 = round(np.mean(algo1_Hired_finish_num), 2)
        algo1_var0_finish = round(np.var(algo1_Hired_finish_num), 2)
        algo1_finish1 = round(np.mean(algo1_Crowdsourced_finish_num), 2)
        algo1_var1_finish = round(np.var(algo1_Crowdsourced_finish_num), 2)
        algo1_finish = round(np.mean(algo1_Hired_finish_num + algo1_Crowdsourced_finish_num), 2)
        algo1_var_finish = round(np.var(algo1_Hired_finish_num + algo1_Crowdsourced_finish_num), 2)
        algo1_finished_num = np.sum(algo1_Hired_finish_num + algo1_Crowdsourced_finish_num)

        algo2_finish0 = round(np.mean(algo2_Hired_finish_num), 2)
        algo2_var0_finish = round(np.var(algo2_Hired_finish_num), 2)
        algo2_finish1 = round(np.mean(algo2_Crowdsourced_finish_num), 2)
        algo2_var1_finish = round(np.var(algo2_Crowdsourced_finish_num), 2)
        algo2_finish = round(np.mean(algo2_Hired_finish_num + algo2_Crowdsourced_finish_num), 2)
        algo2_var_finish = round(np.var(algo2_Hired_finish_num + algo2_Crowdsourced_finish_num), 2)
        algo2_finished_num = np.sum(algo2_Hired_finish_num + algo2_Crowdsourced_finish_num)

        algo3_finish0 = round(np.mean(algo3_Hired_finish_num), 2)
        algo3_var0_finish = round(np.var(algo3_Hired_finish_num), 2)
        algo3_finish1 = round(np.mean(algo3_Crowdsourced_finish_num), 2)
        algo3_var1_finish = round(np.var(algo3_Crowdsourced_finish_num), 2)
        algo3_finish = round(np.mean(algo3_Hired_finish_num + algo3_Crowdsourced_finish_num), 2)
        algo3_var_finish = round(np.var(algo3_Hired_finish_num + algo3_Crowdsourced_finish_num), 2)
        algo3_finished_num = np.sum(algo3_Hired_finish_num + algo3_Crowdsourced_finish_num)

        algo4_finish0 = round(np.mean(algo4_Hired_finish_num), 2)
        algo4_var0_finish = round(np.var(algo4_Hired_finish_num), 2)
        algo4_finish1 = round(np.mean(algo4_Crowdsourced_finish_num), 2)
        algo4_var1_finish = round(np.var(algo4_Crowdsourced_finish_num), 2)
        algo4_finish = round(np.mean(algo4_Hired_finish_num + algo4_Crowdsourced_finish_num), 2)
        algo4_var_finish = round(np.var(algo4_Hired_finish_num + algo4_Crowdsourced_finish_num), 2)
        algo4_finished_num = np.sum(algo4_Hired_finish_num + algo4_Crowdsourced_finish_num)

        algo5_finish0 = round(np.mean(algo5_Hired_finish_num), 2)
        algo5_var0_finish = round(np.var(algo5_Hired_finish_num), 2)
        algo5_finish1 = round(np.mean(algo5_Crowdsourced_finish_num), 2)
        algo5_var1_finish = round(np.var(algo5_Crowdsourced_finish_num), 2)
        algo5_finish = round(np.mean(algo5_Hired_finish_num + algo5_Crowdsourced_finish_num), 2)
        algo5_var_finish = round(np.var(algo5_Hired_finish_num + algo5_Crowdsourced_finish_num), 2)
        algo5_finished_num = np.sum(algo5_Hired_finish_num + algo5_Crowdsourced_finish_num)

        print("Average Finished Orders per Courier for Evaluation Between Algos:")
        print(f"Algo1: Hired finishes average {algo1_finish0} orders (Var: {algo1_var0_finish}), Crowdsourced finishes average {algo1_finish1} orders (Var: {algo1_var1_finish}), Total finish number per courier is {algo1_finish} orders (Var: {algo1_var_finish})")
        print(f"Algo2: Hired finishes average {algo2_finish0} orders (Var: {algo2_var0_finish}), Crowdsourced finishes average {algo2_finish1} orders (Var: {algo2_var1_finish}), Total finish number per courier is {algo2_finish} orders (Var: {algo2_var_finish})")
        print(f"Algo3: Hired finishes average {algo3_finish0} orders (Var: {algo3_var0_finish}), Crowdsourced finishes average {algo3_finish1} orders (Var: {algo3_var1_finish}), Total finish number per courier is {algo3_finish} orders (Var: {algo3_var_finish})")
        print(f"Algo4: Hired finishes average {algo4_finish0} orders (Var: {algo4_var0_finish}), Crowdsourced finishes average {algo4_finish1} orders (Var: {algo4_var1_finish}), Total finish number per courier is {algo4_finish} orders (Var: {algo4_var_finish})")
        print(f"Algo5: Hired finishes average {algo5_finish0} orders (Var: {algo5_var0_finish}), Crowdsourced finishes average {algo5_finish1} orders (Var: {algo5_var1_finish}), Total finish number per courier is {algo5_finish} orders (Var: {algo5_var_finish})")

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

        self.writter.add_scalar('Eval Average Finish/Algo5 Total', algo5_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo5 Hired', algo5_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo5 Crowdsourced', algo5_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo5 Total Var', algo5_var_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo5 Hired Var', algo5_var0_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo5 Crowdsourced Var', algo5_var1_finish, self.eval_num)

        # -----------------------
        # Average Courier Leisure Time
        algo1_avg0_leisure = round(np.mean(algo1_Hired_leisure_time) / 60, 2)
        algo1_var0_leisure = round(np.var(algo1_Hired_leisure_time) / 60**2, 2)
        algo1_avg1_leisure = round(np.mean(algo1_Crowdsourced_leisure_time) / 60, 2)
        algo1_var1_leisure = round(np.var(algo1_Crowdsourced_leisure_time) / 60**2, 2)
        algo1_avg_leisure = round(np.mean(algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time) / 60, 2)
        algo1_var_leisure = round(np.var(algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time) / 60**2, 2)
        algo1_leisure_courier_num = len(algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time)

        algo2_avg0_leisure = round(np.mean(algo2_Hired_leisure_time) / 60, 2)
        algo2_var0_leisure = round(np.var(algo2_Hired_leisure_time) / 60**2, 2)
        algo2_avg1_leisure = round(np.mean(algo2_Crowdsourced_leisure_time) / 60, 2)
        algo2_var1_leisure = round(np.var(algo2_Crowdsourced_leisure_time) / 60**2, 2)
        algo2_avg_leisure = round(np.mean(algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time) / 60, 2)
        algo2_var_leisure = round(np.var(algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time) / 60**2, 2)
        algo2_leisure_courier_num = len(algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time)

        algo3_avg0_leisure = round(np.mean(algo3_Hired_leisure_time) / 60, 2)
        algo3_var0_leisure = round(np.var(algo3_Hired_leisure_time) / 60**2, 2)
        algo3_avg1_leisure = round(np.mean(algo3_Crowdsourced_leisure_time) / 60, 2)
        algo3_var1_leisure = round(np.var(algo3_Crowdsourced_leisure_time) / 60**2, 2)
        algo3_avg_leisure = round(np.mean(algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time) / 60, 2)
        algo3_var_leisure = round(np.var(algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time) / 60**2, 2)
        algo3_leisure_courier_num = len(algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time)

        algo4_avg0_leisure = round(np.mean(algo4_Hired_leisure_time) / 60, 2)
        algo4_var0_leisure = round(np.var(algo4_Hired_leisure_time) / 60**2, 2)
        algo4_avg1_leisure = round(np.mean(algo4_Crowdsourced_leisure_time) / 60, 2)
        algo4_var1_leisure = round(np.var(algo4_Crowdsourced_leisure_time) / 60**2, 2)
        algo4_avg_leisure = round(np.mean(algo4_Hired_leisure_time + algo4_Crowdsourced_leisure_time) / 60, 2)
        algo4_var_leisure = round(np.var(algo4_Hired_leisure_time + algo4_Crowdsourced_leisure_time) / 60**2, 2)
        algo4_leisure_courier_num = len(algo4_Hired_leisure_time + algo4_Crowdsourced_leisure_time)

        algo5_avg0_leisure = round(np.mean(algo5_Hired_leisure_time) / 60, 2)
        algo5_var0_leisure = round(np.var(algo5_Hired_leisure_time) / 60**2, 2)
        algo5_avg1_leisure = round(np.mean(algo5_Crowdsourced_leisure_time) / 60, 2)
        algo5_var1_leisure = round(np.var(algo5_Crowdsourced_leisure_time) / 60**2, 2)
        algo5_avg_leisure = round(np.mean(algo5_Hired_leisure_time + algo5_Crowdsourced_leisure_time) / 60, 2)
        algo5_var_leisure = round(np.var(algo5_Hired_leisure_time + algo5_Crowdsourced_leisure_time) / 60**2, 2)
        algo5_leisure_courier_num = len(algo5_Hired_leisure_time + algo5_Crowdsourced_leisure_time)

        print("Average leisure time per courier for Evaluation Between Algos:")
        print(f"Algo1: Hired leisure time is {algo1_avg0_leisure} minutes (Var: {algo1_var0_leisure}), Crowdsourced leisure time is {algo1_avg1_leisure} minutes (Var: {algo1_var1_leisure}), Total leisure time per courier is {algo1_avg_leisure} minutes (Var: {algo1_var_leisure})")
        print(f"Algo2: Hired leisure time is {algo2_avg0_leisure} minutes (Var: {algo2_var0_leisure}), Crowdsourced leisure time is {algo2_avg1_leisure} minutes (Var: {algo2_var1_leisure}), Total leisure time per courier is {algo2_avg_leisure} minutes (Var: {algo2_var_leisure})")
        print(f"Algo3: Hired leisure time is {algo3_avg0_leisure} minutes (Var: {algo3_var0_leisure}), Crowdsourced leisure time is {algo3_avg1_leisure} minutes (Var: {algo3_var1_leisure}), Total leisure time per courier is {algo3_avg_leisure} minutes (Var: {algo3_var_leisure})")
        print(f"Algo4: Hired leisure time is {algo4_avg0_leisure} minutes (Var: {algo4_var0_leisure}), Crowdsourced leisure time is {algo4_avg1_leisure} minutes (Var: {algo4_var1_leisure}), Total leisure time per courier is {algo4_avg_leisure} minutes (Var: {algo4_var_leisure})")
        print(f"Algo5: Hired leisure time is {algo5_avg0_leisure} minutes (Var: {algo5_var0_leisure}), Crowdsourced leisure time is {algo5_avg1_leisure} minutes (Var: {algo5_var1_leisure}), Total leisure time per courier is {algo5_avg_leisure} minutes (Var: {algo5_var_leisure})")

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
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Total', algo5_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Hired', algo5_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Crowdsourced', algo5_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Total Var', algo5_var_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Hired Var', algo5_var0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo5 Crowdsourced Var', algo5_var1_leisure, self.eval_num)

        # -----------------------
        # Average Courier running Time
        algo1_avg0_running = round(np.mean(algo1_Hired_running_time) / 60, 2)
        algo1_var0_running = round(np.var(algo1_Hired_running_time) / 60**2, 2)
        algo1_avg1_running = round(np.mean(algo1_Crowdsourced_running_time) / 60, 2)
        algo1_var1_running = round(np.var(algo1_Crowdsourced_running_time) / 60**2, 2)
        algo1_avg_running = round(np.mean(algo1_Hired_running_time + algo1_Crowdsourced_running_time) / 60, 2)
        algo1_var_running = round(np.var(algo1_Hired_running_time + algo1_Crowdsourced_running_time) / 60**2, 2)
        algo1_running_courier_num = len(algo1_Hired_running_time + algo1_Crowdsourced_running_time)

        algo2_avg0_running = round(np.mean(algo2_Hired_running_time) / 60, 2)
        algo2_var0_running = round(np.var(algo2_Hired_running_time) / 60**2, 2)
        algo2_avg1_running = round(np.mean(algo2_Crowdsourced_running_time) / 60, 2)
        algo2_var1_running = round(np.var(algo2_Crowdsourced_running_time) / 60**2, 2)
        algo2_avg_running = round(np.mean(algo2_Hired_running_time + algo2_Crowdsourced_running_time) / 60, 2)
        algo2_var_running = round(np.var(algo2_Hired_running_time + algo2_Crowdsourced_running_time) / 60**2, 2)
        algo2_running_courier_num = len(algo2_Hired_running_time + algo2_Crowdsourced_running_time)

        algo3_avg0_running = round(np.mean(algo3_Hired_running_time) / 60, 2)
        algo3_var0_running = round(np.var(algo3_Hired_running_time) / 60**2, 2)
        algo3_avg1_running = round(np.mean(algo3_Crowdsourced_running_time) / 60, 2)
        algo3_var1_running = round(np.var(algo3_Crowdsourced_running_time) / 60**2, 2)
        algo3_avg_running = round(np.mean(algo3_Hired_running_time + algo3_Crowdsourced_running_time) / 60, 2)
        algo3_var_running = round(np.var(algo3_Hired_running_time + algo3_Crowdsourced_running_time) / 60**2, 2)
        algo3_running_courier_num = len(algo3_Hired_running_time + algo3_Crowdsourced_running_time)

        algo4_avg0_running = round(np.mean(algo4_Hired_running_time) / 60, 2)
        algo4_var0_running = round(np.var(algo4_Hired_running_time) / 60**2, 2)
        algo4_avg1_running = round(np.mean(algo4_Crowdsourced_running_time) / 60, 2)
        algo4_var1_running = round(np.var(algo4_Crowdsourced_running_time) / 60**2, 2)
        algo4_avg_running = round(np.mean(algo4_Hired_running_time + algo4_Crowdsourced_running_time) / 60, 2)
        algo4_var_running = round(np.var(algo4_Hired_running_time + algo4_Crowdsourced_running_time) / 60**2, 2)
        algo4_running_courier_num = len(algo4_Hired_running_time + algo4_Crowdsourced_running_time)

        algo5_avg0_running = round(np.mean(algo5_Hired_running_time) / 60, 2)
        algo5_var0_running = round(np.var(algo5_Hired_running_time) / 60**2, 2)
        algo5_avg1_running = round(np.mean(algo5_Crowdsourced_running_time) / 60, 2)
        algo5_var1_running = round(np.var(algo5_Crowdsourced_running_time) / 60**2, 2)
        algo5_avg_running = round(np.mean(algo5_Hired_running_time + algo5_Crowdsourced_running_time) / 60, 2)
        algo5_var_running = round(np.var(algo5_Hired_running_time + algo5_Crowdsourced_running_time) / 60**2, 2)
        algo5_running_courier_num = len(algo5_Hired_running_time + algo5_Crowdsourced_running_time)

        print("Average running time per courier for Evaluation Between Algos:")
        print(f"Algo1: Hired running time is {algo1_avg0_running} minutes (Var: {algo1_var0_running}), Crowdsourced running time is {algo1_avg1_running} minutes (Var: {algo1_var1_running}), Total running time per courier is {algo1_avg_running} minutes (Var: {algo1_var_running})")
        print(f"Algo2: Hired running time is {algo2_avg0_running} minutes (Var: {algo2_var0_running}), Crowdsourced running time is {algo2_avg1_running} minutes (Var: {algo2_var1_running}), Total running time per courier is {algo2_avg_running} minutes (Var: {algo2_var_running})")
        print(f"Algo3: Hired running time is {algo3_avg0_running} minutes (Var: {algo3_var0_running}), Crowdsourced running time is {algo3_avg1_running} minutes (Var: {algo3_var1_running}), Total running time per courier is {algo3_avg_running} minutes (Var: {algo3_var_running})")
        print(f"Algo4: Hired running time is {algo4_avg0_running} minutes (Var: {algo4_var0_running}), Crowdsourced running time is {algo4_avg1_running} minutes (Var: {algo4_var1_running}), Total running time per courier is {algo4_avg_running} minutes (Var: {algo4_var_running})")
        print(f"Algo5: Hired running time is {algo5_avg0_running} minutes (Var: {algo5_var0_running}), Crowdsourced running time is {algo5_avg1_running} minutes (Var: {algo5_var1_running}), Total running time per courier is {algo5_avg_running} minutes (Var: {algo5_var_running})")

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
        self.writter.add_scalar('Eval Average running Time/Algo5 Total', algo5_avg_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo5 Hired', algo5_avg0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo5 Crowdsourced', algo5_avg1_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo5 Total Var', algo5_var_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo5 Hired Var', algo5_var0_running, self.eval_num)
        self.writter.add_scalar('Eval Average running Time/Algo5 Crowdsourced Var', algo5_var1_running, self.eval_num)

        message = (
            f"\nIn Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} ({algo1_Crowdsourced_on / algo1_Crowdsourced_num}) on, finishing {algo1_finished_num} orders in {algo1_order0_num} Order0 and {algo1_order1_num} Order1, {algo1_order_wait} ({round(100 * algo1_order_wait / (algo1_order_wait + algo1_order0_num + algo1_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} ({algo2_Crowdsourced_on / algo2_Crowdsourced_num}) on, finishing {algo2_finished_num} orders in {algo2_order0_num} Order0 and {algo2_order1_num} Order1, {algo2_order_wait} ({round(100 * algo2_order_wait / (algo2_order_wait + algo2_order0_num + algo2_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} ({algo3_Crowdsourced_on / algo3_Crowdsourced_num}) on, finishing {algo3_finished_num} orders in {algo3_order0_num} Order0 and {algo3_order1_num} Order1, {algo3_order_wait} ({round(100 * algo3_order_wait / (algo3_order_wait + algo3_order0_num + algo3_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo4 there are {algo4_Hired_num} Hired, {algo4_Crowdsourced_num} Crowdsourced with {algo4_Crowdsourced_on} ({algo4_Crowdsourced_on / algo4_Crowdsourced_num}) on, finishing {algo4_finished_num} orders in {algo4_order0_num} Order0 and {algo4_order1_num} Order1, {algo4_order_wait} ({round(100 * algo4_order_wait / (algo4_order_wait + algo4_order0_num + algo4_order1_num), 2)}%) Orders waiting to be paired\n"
            f"In Algo5 there are {algo5_Hired_num} Hired, {algo5_Crowdsourced_num} Crowdsourced with {algo5_Crowdsourced_on} ({algo5_Crowdsourced_on / algo5_Crowdsourced_num}) on, finishing {algo2_finished_num} orders in {algo5_order0_num} Order0 and {algo5_order1_num} Order1, {algo5_order_wait} ({round(100 * algo5_order_wait / (algo5_order_wait + algo5_order0_num + algo5_order1_num), 2)}%) Orders waiting to be paired\n"
            f"Total Reward for Evaluation Between Algos:\n"
            f"Algo1: {round(algo1_eval_episode_rewards_sum, 2)}\n"
            f"Algo2: {round(algo2_eval_episode_rewards_sum, 2)}\n"
            f"Algo3: {round(algo3_eval_episode_rewards_sum, 2)}\n"
            f"Algo4: {round(algo4_eval_episode_rewards_sum, 2)}\n"
            f"Algo5: {round(algo5_eval_episode_rewards_sum, 2)}\n"
            f"Average Travel Distance per Courier Between Algos:\n"
            f"Algo1: Hired - {algo1_distance0} km (Var: {algo1_var0_distance}), Crowdsourced - {algo1_distance1} km (Var: {algo1_var1_distance}), Total ({algo1_distance_courier_num}) - {algo1_distance} km (Var: {algo1_var_distance})\n"
            f"Algo2: Hired - {algo2_distance0} km (Var: {algo2_var0_distance}), Crowdsourced - {algo2_distance1} km (Var: {algo2_var1_distance}), Total ({algo2_distance_courier_num}) - {algo2_distance} km (Var: {algo2_var_distance})\n"
            f"Algo3: Hired - {algo3_distance0} km (Var: {algo3_var0_distance}), Crowdsourced - {algo3_distance1} km (Var: {algo3_var1_distance}), Total ({algo3_distance_courier_num}) - {algo3_distance} km (Var: {algo3_var_distance})\n"
            f"Algo4: Hired - {algo4_distance0} km (Var: {algo4_var0_distance}), Crowdsourced - {algo4_distance1} km (Var: {algo4_var1_distance}), Total ({algo4_distance_courier_num}) - {algo4_distance} km (Var: {algo4_var_distance})\n"
            f"Algo5: Hired - {algo5_distance0} km (Var: {algo5_var0_distance}), Crowdsourced - {algo5_distance1} km (Var: {algo5_var1_distance}), Total ({algo5_distance_courier_num}) - {algo5_distance} km (Var: {algo5_var_distance})\n"
            "Average Speed per Courier Between Algos:\n"
            f"Algo1: Hired average speed is {algo1_avg0_speed} m/s (Var: {algo1_var0_speed}), Crowdsourced average speed is {algo1_avg1_speed} m/s (Var: {algo1_var1_speed}) and average speed per courier is {algo1_avg_speed} m/s (Var: {algo1_var_speed})\n"
            f"Algo2: Hired average speed is {algo2_avg0_speed} m/s (Var: {algo2_var0_speed}), Crowdsourced average speed is {algo2_avg1_speed} m/s (Var: {algo2_var1_speed}) and average speed per courier is {algo2_avg_speed} m/s (Var: {algo2_var_speed})\n"
            f"Algo3: Hired average speed is {algo3_avg0_speed} m/s (Var: {algo3_var0_speed}), Crowdsourced average speed is {algo3_avg1_speed} m/s (Var: {algo3_var1_speed}) and average speed per courier is {algo3_avg_speed} m/s (Var: {algo3_var_speed})\n"
            f"Algo4: Hired average speed is {algo4_avg0_speed} m/s (Var: {algo4_var0_speed}), Crowdsourced average speed is {algo4_avg1_speed} m/s (Var: {algo4_var1_speed}) and average speed per courier is {algo4_avg_speed} m/s (Var: {algo4_var_speed})\n"
            f"Algo5: Hired average speed is {algo5_avg0_speed} m/s (Var: {algo5_var0_speed}), Crowdsourced average speed is {algo5_avg1_speed} m/s (Var: {algo5_var1_speed}) and average speed per courier is {algo5_avg_speed} m/s (Var: {algo5_var_speed})\n"
            "Rate of Overspeed for Evaluation Between Algos:\n"
            f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}\n"
            f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}\n"
            f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}\n"
            f"Algo4: Hired - {algo4_overspeed0}, Crowdsourced - {algo4_overspeed1}, Total rate - {algo4_overspeed}\n"
            f"Algo5: Hired - {algo5_overspeed0}, Crowdsourced - {algo5_overspeed1}, Total rate - {algo5_overspeed}\n"
            "Average Price per order for Evaluation Between Algos:\n"
            f"Algo1: The average price of Hired's order is {algo1_price_per_order0} dollar (Var: {algo1_var0_price}) with {algo1_order0_num} orders, Crowdsourced's is {algo1_price_per_order1} dollar (Var: {algo1_var1_price}) with {algo1_order1_num} orders and for all is {algo1_price_per_order} dollar (Var: {algo1_var_price})\n"
            f"Algo2: The average price of Hired's order is {algo2_price_per_order0} dollar (Var: {algo2_var0_price}) with {algo2_order0_num} orders, Crowdsourced's is {algo2_price_per_order1} dollar (Var: {algo2_var1_price}) with {algo2_order1_num} orders and for all is {algo2_price_per_order} dollar (Var: {algo2_var_price})\n"
            f"Algo3: The average price of Hired's order is {algo3_price_per_order0} dollar (Var: {algo3_var0_price}) with {algo3_order0_num} orders, Crowdsourced's is {algo3_price_per_order1} dollar (Var: {algo3_var1_price}) with {algo3_order1_num} orders and for all is {algo3_price_per_order} dollar (Var: {algo3_var_price})\n"
            f"Algo4: The average price of Hired's order is {algo4_price_per_order0} dollar (Var: {algo4_var0_price}) with {algo4_order0_num} orders, Crowdsourced's is {algo4_price_per_order1} dollar (Var: {algo4_var1_price}) with {algo4_order1_num} orders and for all is {algo4_price_per_order} dollar (Var: {algo4_var_price})\n"
            f"Algo5: The average price of Hired's order is {algo5_price_per_order0} dollar (Var: {algo5_var0_price}) with {algo5_order0_num} orders, Crowdsourced's is {algo5_price_per_order1} dollar (Var: {algo5_var1_price}) with {algo5_order1_num} orders and for all is {algo5_price_per_order} dollar (Var: {algo5_var_price})\n"
            "Average Income per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average income is {algo1_income0} dollar (Var: {algo1_var0_income}), Crowdsourced's average income is {algo1_income1} dollar (Var: {algo1_var1_income}) and Total income per ({algo1_income_courier_num}) courier is {algo1_income} dollar (Var: {algo1_var_income}), The platform total cost is {round(platform_cost1, 2)} dollar\n"
            f"Algo2: Hired's average income is {algo2_income0} dollar (Var: {algo2_var0_income}), Crowdsourced's average income is {algo2_income1} dollar (Var: {algo2_var1_income}) and Total income per ({algo2_income_courier_num}) courier is {algo2_income} dollar (Var: {algo2_var_income}), The platform total cost is {round(platform_cost2, 2)} dollar\n"
            f"Algo3: Hired's average income is {algo3_income0} dollar (Var: {algo3_var0_income}), Crowdsourced's average income is {algo3_income1} dollar (Var: {algo3_var1_income}) and Total income per ({algo3_income_courier_num}) courier is {algo3_income} dollar (Var: {algo3_var_income}), The platform total cost is {round(platform_cost3, 2)} dollar\n"
            f"Algo4: Hired's average income is {algo4_income0} dollar (Var: {algo4_var0_income}), Crowdsourced's average income is {algo4_income1} dollar (Var: {algo4_var1_income}) and Total income per ({algo4_income_courier_num}) courier is {algo4_income} dollar (Var: {algo4_var_income}), The platform total cost is {round(platform_cost4, 2)} dollar\n"
            f"Algo5: Hired's average income is {algo5_income0} dollar (Var: {algo5_var0_income}), Crowdsourced's average income is {algo5_income1} dollar (Var: {algo5_var1_income}) and Total income per ({algo5_income_courier_num}) courier is {algo5_income} dollar (Var: {algo5_var_income}), The platform total cost is {round(platform_cost5, 2)} dollar\n"
            "Average Leisure Time per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average leisure time is {algo1_avg0_leisure} minutes (Var: {algo1_var0_leisure}), Crowdsourced's average leisure time is {algo1_avg1_leisure} minutes (Var: {algo1_var1_leisure}) and Total leisure time per ({algo1_leisure_courier_num}) courier is {algo1_avg_leisure} minutes (Var: {algo1_var_leisure})\n"
            f"Algo2: Hired's average leisure time is {algo2_avg0_leisure} minutes (Var: {algo2_var0_leisure}), Crowdsourced's average leisure time is {algo2_avg1_leisure} minutes (Var: {algo2_var1_leisure}) and Total leisure time per ({algo2_leisure_courier_num}) courier is {algo2_avg_leisure} minutes (Var: {algo2_var_leisure})\n"
            f"Algo3: Hired's average leisure time is {algo3_avg0_leisure} minutes (Var: {algo3_var0_leisure}), Crowdsourced's average leisure time is {algo3_avg1_leisure} minutes (Var: {algo3_var1_leisure}) and Total leisure time per ({algo3_leisure_courier_num}) courier is {algo3_avg_leisure} minutes (Var: {algo3_var_leisure})\n"
            f"Algo4: Hired's average leisure time is {algo4_avg0_leisure} minutes (Var: {algo4_var0_leisure}), Crowdsourced's average leisure time is {algo4_avg1_leisure} minutes (Var: {algo4_var1_leisure}) and Total leisure time per ({algo4_leisure_courier_num}) courier is {algo4_avg_leisure} minutes (Var: {algo4_var_leisure})\n"
            f"Algo5: Hired's average leisure time is {algo5_avg0_leisure} minutes (Var: {algo5_var0_leisure}), Crowdsourced's average leisure time is {algo5_avg1_leisure} minutes (Var: {algo5_var1_leisure}) and Total leisure time per ({algo5_leisure_courier_num}) courier is {algo5_avg_leisure} minutes (Var: {algo5_var_leisure})\n"
            "Average Running Time per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired's average running time is {algo1_avg0_running} minutes (Var: {algo1_var0_running}), Crowdsourced's average running time is {algo1_avg1_running} minutes (Var: {algo1_var1_running}) and Total running time per ({algo1_running_courier_num}) courier is {algo1_avg_running} minutes (Var: {algo1_var_running})\n"
            f"Algo2: Hired's average running time is {algo2_avg0_running} minutes (Var: {algo2_var0_running}), Crowdsourced's average running time is {algo2_avg1_running} minutes (Var: {algo2_var1_running}) and Total running time per ({algo2_running_courier_num}) courier is {algo2_avg_running} minutes (Var: {algo2_var_running})\n"
            f"Algo3: Hired's average running time is {algo3_avg0_running} minutes (Var: {algo3_var0_running}), Crowdsourced's average running time is {algo3_avg1_running} minutes (Var: {algo3_var1_running}) and Total running time per ({algo3_running_courier_num}) courier is {algo3_avg_running} minutes (Var: {algo3_var_running})\n"
            f"Algo4: Hired's average running time is {algo4_avg0_running} minutes (Var: {algo4_var0_running}), Crowdsourced's average running time is {algo4_avg1_running} minutes (Var: {algo4_var1_running}) and Total running time per ({algo4_running_courier_num}) courier is {algo4_avg_running} minutes (Var: {algo4_var_running})\n"
            f"Algo5: Hired's average running time is {algo5_avg0_running} minutes (Var: {algo5_var0_running}), Crowdsourced's average running time is {algo5_avg1_running} minutes (Var: {algo5_var1_running}) and Total running time per ({algo5_running_courier_num}) courier is {algo5_avg_running} minutes (Var: {algo5_var_running})\n"
            "Average Order Finished per Courier for Evaluation Between Algos:\n"
            f"Algo1: Hired courier finishes average {algo1_finish0} orders (Var: {algo1_var0_finish}), Crowdsourced courier finishes average {algo1_finish1} orders (Var: {algo1_var1_finish}) and Total is {algo1_finish} orders (Var: {algo1_var_finish})\n"
            f"Algo2: Hired courier finishes average {algo2_finish0} orders (Var: {algo2_var0_finish}), Crowdsourced courier finishes average {algo2_finish1} orders (Var: {algo2_var1_finish}) and Total is {algo2_finish} orders (Var: {algo2_var_finish})\n"
            f"Algo3: Hired courier finishes average {algo3_finish0} orders (Var: {algo3_var0_finish}), Crowdsourced courier finishes average {algo3_finish1} orders (Var: {algo3_var1_finish}) and Total is {algo3_finish} orders (Var: {algo3_var_finish})\n"
            f"Algo4: Hired courier finishes average {algo4_finish0} orders (Var: {algo4_var0_finish}), Crowdsourced courier finishes average {algo4_finish1} orders (Var: {algo4_var1_finish}) and Total is {algo4_finish} orders (Var: {algo4_var_finish})\n"
            f"Algo5: Hired courier finishes average {algo5_finish0} orders (Var: {algo5_var0_finish}), Crowdsourced courier finishes average {algo5_finish1} orders (Var: {algo5_var1_finish}) and Total is {algo5_finish} orders (Var: {algo5_var_finish})\n"
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
            print(f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate} out of ({algo1_count_dropped_orders0 +algo1_count_dropped_orders1})")

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
            
            message += f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate} out of ({algo1_count_dropped_orders0 +algo1_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0} (Var: {algo1_var0_ETA}), Crowdsourced - {algo1_ETA_usage_rate1} (Var: {algo1_var1_ETA}), Total - {algo1_ETA_usage_rate} (Var: {algo1_var_ETA})\n"
        
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
            print(f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate} out of ({algo2_count_dropped_orders0 +algo2_count_dropped_orders1})")

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
            
            message += f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate} out of ({algo2_count_dropped_orders0 +algo2_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo2: Hired - {algo2_ETA_usage_rate0} (Var: {algo2_var0_ETA}), Crowdsourced - {algo2_ETA_usage_rate1} (Var: {algo2_var1_ETA}), Total - {algo2_ETA_usage_rate} (Var: {algo2_var_ETA})\n"

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
            print(f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate} out of ({algo3_count_dropped_orders0 +algo3_count_dropped_orders1})")

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
            
            message += f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate} out of ({algo3_count_dropped_orders0 +algo3_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo3: Hired - {algo3_ETA_usage_rate0} (Var: {algo3_var0_ETA}), Crowdsourced - {algo3_ETA_usage_rate1} (Var: {algo3_var1_ETA}), Total - {algo3_ETA_usage_rate} (Var: {algo3_var_ETA})\n"

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
            print(f"Rate of Late Orders for Evaluation in Algo4: Hired - {algo4_late_rate0}, Crowdsourced - {algo4_late_rate1}, Total - {algo4_late_rate} out of ({algo4_count_dropped_orders0 +algo4_count_dropped_orders1})")

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
            
            message += f"Rate of Late Orders for Evaluation in Algo4: Hired - {algo4_late_rate0}, Crowdsourced - {algo4_late_rate1}, Total - {algo4_late_rate} out of ({algo4_count_dropped_orders0 +algo4_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo4: Hired - {algo4_ETA_usage_rate0} (Var: {algo4_var0_ETA}), Crowdsourced - {algo4_ETA_usage_rate1} (Var: {algo4_var1_ETA}), Total - {algo4_ETA_usage_rate} (Var: {algo4_var_ETA})\n"
            
        if algo5_count_dropped_orders0 + algo5_count_dropped_orders1 == 0:
            print("No order is dropped in Algo5")
            algo5_late_rate = -1
            algo5_late_rate0 = -1
            algo5_late_rate1 = -1
            algo5_ETA_usage_rate = -1
            algo5_ETA_usage_rate0 = -1
            algo5_ETA_usage_rate1 = -1
            algo5_var_ETA = 0
            algo5_var0_ETA = 0
            algo5_var1_ETA = 0

            self.writter.add_scalar('Eval Late Order Rate/Algo5 Total', algo5_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo5 Hired', algo5_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo5 Crowdsourced', algo5_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Total', algo5_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Total Var', algo5_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Hired', algo5_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Hired Var', algo5_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Crowdsourced', algo5_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Crowdsourced Var', algo5_var1_ETA, self.eval_num)
            
            message += "No order is dropped in Algo5\n"
        else:
            if algo5_count_dropped_orders0:                
                algo5_late_rate0 = round(algo5_late_orders0 / algo5_count_dropped_orders0, 2)
                algo5_ETA_usage_rate0 = round(np.mean(algo5_ETA_usage0), 2)
                algo5_var0_ETA = round(np.var(algo5_ETA_usage0), 2)
            else:
                algo5_late_rate0 = -1
                algo5_ETA_usage_rate0 = -1
                algo5_var0_ETA = 0
                
            if algo5_count_dropped_orders1:                
                algo5_late_rate1 = round(algo5_late_orders1 / algo5_count_dropped_orders1, 2)
                algo5_ETA_usage_rate1 = round(np.mean(algo5_ETA_usage1), 2)
                algo5_var1_ETA = round(np.var(algo5_ETA_usage1), 2)
            else:
                algo5_late_rate1 = -1
                algo5_ETA_usage_rate1 = -1
                algo5_var1_ETA = 0
                
            algo5_late_rate = round((algo5_late_orders0 + algo5_late_orders1) / (algo5_count_dropped_orders0 +algo5_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo5: Hired - {algo5_late_rate0}, Crowdsourced - {algo5_late_rate1}, Total - {algo5_late_rate} out of ({algo5_count_dropped_orders0 +algo5_count_dropped_orders1})")

            algo5_ETA_usage_rate = round(np.mean(algo5_ETA_usage0 + algo5_ETA_usage1), 2)
            algo5_var_ETA = round(np.var(algo5_ETA_usage0 + algo5_ETA_usage1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo5: Hired - {algo5_ETA_usage_rate0} (Var: {algo5_var0_ETA}), Crowdsourced - {algo5_ETA_usage_rate1} (Var: {algo5_var1_ETA}), Total - {algo5_ETA_usage_rate} (Var: {algo5_var_ETA})")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo5 Total', algo5_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo5 Hired', algo5_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo5 Crowdsourced', algo5_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Total', algo5_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Hired', algo5_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Crowdsourced', algo5_ETA_usage_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Total Var', algo5_var_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Hired Var', algo5_var0_ETA, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo5 Crowdsourced Var', algo5_var1_ETA, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo5: Hired - {algo5_late_rate0}, Crowdsourced - {algo5_late_rate1}, Total - {algo5_late_rate} out of ({algo5_count_dropped_orders0 +algo5_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo5: Hired - {algo5_ETA_usage_rate0} (Var: {algo5_var0_ETA}), Crowdsourced - {algo5_ETA_usage_rate1} (Var: {algo5_var1_ETA}), Total - {algo5_ETA_usage_rate} (Var: {algo5_var_ETA})\n"

        # algo1_social_welfare = sum(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo2_social_welfare = sum(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo3_social_welfare = sum(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo4_social_welfare = sum(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo5_social_welfare = sum(algo5_Hired_distance_per_episode + algo5_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # print(f"Algo1: The platform total cost is {round(platform_cost1, 2)} dollar, and social welfare is {algo1_social_welfare} dollar")
        # print(f"Algo2: The platform total cost is {round(platform_cost2, 2)} dollar, and social welfare is {algo2_social_welfare} dollar")
        # print(f"Algo3: The platform total cost is {round(platform_cost3, 2)} dollar, and social welfare is {algo3_social_welfare} dollar")
        # print(f"Algo4: The platform total cost is {round(platform_cost4, 2)} dollar, and social welfare is {algo4_social_welfare} dollar")
        # print(f"Algo5: The platform total cost is {round(platform_cost5, 2)} dollar, and social welfare is {algo5_social_welfare} dollar")
        # message += f"Algo1: Social welfare is {algo1_social_welfare} dollar\n" + f"Algo2: Social welfare is {algo2_social_welfare} dollar\n" + f"Algo3: Social welfare is {algo3_social_welfare} dollar\n" + f"Algo4: Social welfare is {algo4_social_welfare} dollar\n" + f"Algo5: Social welfare is {algo5_social_welfare} dollar\n"
        # self.writter.add_scalar('Social Welfare/Algo1', algo1_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo2', algo2_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo3', algo3_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo4', algo4_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo5', algo5_social_welfare, self.eval_num)

        logger.success(message)
            
        print("\n")