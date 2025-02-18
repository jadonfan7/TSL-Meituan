import time
import numpy as np
import torch

from runner.base_runner import Runner
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
            order_wait = 0
            ETA_usage = []
            
            order_waiting_time = []

            order_price = []
                     
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
            
            Hired_congestion_time = []
            Crowdsourced_congestion_time = []
            
            Hired_waiting_time = []
            Crowdsourced_waiting_time = []
            
            Hired_actual_speed = []
            Crowdsourced_actual_speed = []
            
            Hired_income = []
            Crowdsourced_income = []

            if self.use_linear_lr_decay:
                self.trainer1.policy.lr_decay(episode, episodes)
                self.trainer2.policy.lr_decay(episode, episodes)
            
            obs = self.envs.reset(episode % 4)
            # obs = self.envs.reset(1)
            # self.reset_courier_num(self.envs.envs_map[0].num_couriers)
            self.num_agents = self.envs.envs_map[0].num_couriers

            for step in range(self.episode_length):
                # print("-"*25)
                print(f"THIS IS STEP {step}")
                # dead_count = 0 # end the code

                for i in range(self.envs.num_envs):
                    # print(f"ENVIRONMENT {i+1}")

                    # print("Couriers:")
                    # for c in self.envs.envs_map[i].couriers:
                    #     if c.state == 'active':
                    #         print(c)
                    # print("Orders:")
                    # for o in self.envs.envs_map[i].orders:
                    #     print(o)  
                    # print("\n")
                    self.log_env(episode, step, i)

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

                # Obser reward and next obs
                obs, rewards, dones, infos, share_obs = self.envs.step(actions_env)

                episode_reward_sum += rewards.sum() / self.envs.num_envs

                num0 = 0
                num1 = 0
                count0 = 0
                count1 = 0
                for c in self.envs.envs_map[0].couriers:
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
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)


                self.envs.env_step()
            
            # Train over periods
            platform_cost = self.envs.envs_map[0].platform_cost
            
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
                    Hired_congestion_time.append(c.total_congestion_time)
                    Hired_waiting_time.append(c.total_waiting_time)
                    if c.actual_speed > 0:
                        Hired_actual_speed.append(c.actual_speed)
                    if c.income > 0:
                        Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                else:
                    Crowdsourced_num += 1
                    if c.travel_distance > 0:
                        Crowdsourced_distance_per_episode.append(c.travel_distance)
                    Crowdsourced_finished_num.append(c.finish_order_num)
                    Crowdsourced_unfinished_num.append(len(c.waybill)+len(c.wait_to_pick))
                    Crowdsourced_reject_num.append(c.reject_order_num)
                    Crowdsourced_leisure_time.append(c.total_leisure_time)
                    Crowdsourced_running_time.append(c.total_running_time)
                    Crowdsourced_congestion_time.append(c.total_congestion_time)
                    if c.actual_speed > 0:
                        Crowdsourced_actual_speed.append(c.actual_speed)
                    if c.income > 0:
                        Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    if c.state == 'active':
                        Crowdsourced_on += 1
            
            courier_num = len(self.envs.envs_map[0].couriers)
            
            for o in self.envs.envs_map[0].orders:
                if o.status == 'dropped':
                    count_dropped_orders += 1
                    if o.is_late == 1:
                        late_orders += 1
                    else:
                        ETA_usage.append(o.ETA_usage)
                else:
                    count_unfinished_orders += 1
                    if o.ETA <= self.envs.envs_map[0].clock:
                        unfinished_late_orders += 1
                
                if o.reject_count > 0:
                    count_reject_orders += 1
                    if max_reject_num <= o.reject_count:
                        max_reject_num = o.reject_count
                    
                if o.status == 'wait_pair':
                    order_wait += 1
                else:
                    order_waiting_time.append(o.wait_time)
                    order_price.append(o.price)
                    
            order_num = len(self.envs.envs_map[0].orders)
                            
            print(f"\nThis is Episode {episode+1}")                
            print(f"There are {courier_num} couriers ({Hired_num} Hired, {Crowdsourced_num} Crowdsourced with {Crowdsourced_on} ({round(100 * Crowdsourced_on / Crowdsourced_num, 2)}%) on), and {order_num} Orders ({count_dropped_orders} dropped, {count_unfinished_orders} unfinished, {order_wait} ({round(100 * order_wait / order_num, 2)}%) waiting to be paired)")     
            
            print(f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}")
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
            
            # ---------------------
            # distance
            distance0 = np.mean(Hired_distance_per_episode) / 1000
            distance_var0 = np.var(Hired_distance_per_episode) / 1000000
            distance1 = np.mean(Crowdsourced_distance_per_episode) / 1000
            distance_var1 = np.var(Crowdsourced_distance_per_episode) / 1000000
            distance = np.mean(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000
            distance_var = np.var(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000000
            print(
                f"Average Travel Distance per Hired: {distance0} km (Var: {distance_var0}), "
                f"Crowdsourced: {distance1} km (Var: {distance_var1}), "
                f"Total: {distance} km (Var: {distance_var})"
            )
            self.writter.add_scalar('Train/Total Distance/Total', distance, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Total_Var', distance_var, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Hired', distance0, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Hired_Var', distance_var0, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Crowdsourced', distance1, episode + 1)
            self.writter.add_scalar('Train/Total Distance/Crowdsourced_Var', distance_var1, episode + 1)
            
            # ---------------------
            # average courier finished order number
            avg_finish0 = np.mean(Hired_finished_num)
            var_finish0 = np.var(Hired_finished_num)
            avg_finish1 = np.mean(Crowdsourced_finished_num)
            var_finish1 = np.var(Crowdsourced_finished_num)
            avg_finish = np.mean(Hired_finished_num + Crowdsourced_finished_num)
            var_finish = np.var(Hired_finished_num + Crowdsourced_finished_num)
            print(
                f"Hired finishes average {avg_finish0} orders (Var: {var_finish0}), "
                f"Crowdsourced finishes average {avg_finish1} orders (Var: {var_finish1}), "
                f"Total finish number per courier is {avg_finish} (Var: {var_finish})"
            )
            self.writter.add_scalar('Train/Average Finish/Total', avg_finish, episode + 1)
            self.writter.add_scalar('Train/Average Finish/Total_Var', var_finish, episode + 1)
            self.writter.add_scalar('Train/Average Finish/Hired', avg_finish0, episode + 1)
            self.writter.add_scalar('Train/Average Finish/Hired_Var', var_finish0, episode + 1)
            self.writter.add_scalar('Train/Average Finish/Crowdsourced', avg_finish1, episode + 1)
            self.writter.add_scalar('Train/Average Finish/Crowdsourced_Var', var_finish1, episode + 1)

            # ---------------------
            # average courier unfinished order number
            avg_unfinish0 = np.mean(Hired_unfinished_num)
            var_unfinish0 = np.var(Hired_unfinished_num)
            avg_unfinish1 = np.mean(Crowdsourced_unfinished_num)
            var_unfinish1 = np.var(Crowdsourced_unfinished_num)
            avg_unfinish = np.mean(Hired_unfinished_num + Crowdsourced_unfinished_num)
            var_unfinish = np.var(Hired_unfinished_num + Crowdsourced_unfinished_num)
            print(
                f"Hired unfinishes average {avg_unfinish0} orders (Var: {var_unfinish0}), "
                f"Crowdsourced unfinishes average {avg_unfinish1} orders (Var: {var_unfinish1}), "
                f"Total unfinished number per courier is {avg_unfinish} (Var: {var_unfinish})"
            )
            self.writter.add_scalar('Train/Average Unfinish/Total', avg_unfinish, episode + 1)
            self.writter.add_scalar('Train/Average Unfinish/Total_Var', var_unfinish, episode + 1)
            self.writter.add_scalar('Train/Average Unfinish/Hired', avg_unfinish0, episode + 1)
            self.writter.add_scalar('Train/Average Unfinish/Hired_Var', var_unfinish0, episode + 1)
            self.writter.add_scalar('Train/Average Unfinish/Crowdsourced', avg_unfinish1, episode + 1)
            self.writter.add_scalar('Train/Average Unfinish/Crowdsourced_Var', var_unfinish1, episode + 1)

            # ---------------------
            # courier reject number
            avg_reject0 = np.mean(Hired_reject_num)
            var_reject0 = np.var(Hired_reject_num)
            avg_reject1 = np.mean(Crowdsourced_reject_num)
            var_reject1 = np.var(Crowdsourced_reject_num)
            avg_reject = np.mean(Hired_reject_num + Crowdsourced_reject_num)
            var_reject = np.var(Hired_reject_num + Crowdsourced_reject_num)
            print(
                f"The average rejection number for Episode {episode+1}: Hired - {avg_reject0} (Var: {var_reject0}), "
                f"Crowdsourced - {avg_reject1} (Var: {var_reject1}), "
                f"Total - {avg_reject} (Var: {var_reject})"
            )
            self.writter.add_scalar('Train/Reject Rate/Total', avg_reject, episode + 1)
            self.writter.add_scalar('Train/Reject Rate/Total_Var', var_reject, episode + 1)
            self.writter.add_scalar('Train/Reject Rate/Hired', avg_reject0, episode + 1)
            self.writter.add_scalar('Train/Reject Rate/Hired_Var', var_reject0, episode + 1)
            self.writter.add_scalar('Train/Reject Rate/Crowdsourced', avg_reject1, episode + 1)
            self.writter.add_scalar('Train/Reject Rate/Crowdsourced_Var', var_reject1, episode + 1)

            # ---------------------
            # average courier leisure time
            avg0_leisure = np.mean(Hired_leisure_time) / 60
            var_leisure0 = np.var(Hired_leisure_time) / 60**2
            avg1_leisure = np.mean(Crowdsourced_leisure_time) / 60
            var_leisure1 = np.var(Crowdsourced_leisure_time) / 60**2
            avg_leisure = np.mean(Hired_leisure_time + Crowdsourced_leisure_time) / 60
            var_leisure = np.var(Hired_leisure_time + Crowdsourced_leisure_time) / 60**2
            print(f"Hired leisure time is {avg0_leisure} minutes (Var: {var_leisure0}), Crowdsourced leisure time is {avg1_leisure} minutes (Var: {var_leisure1}) and Total leisure time per courier is {avg_leisure} minutes (Var: {var_leisure})")
            self.writter.add_scalar('Train/Average Leisure Time/Total', avg_leisure, episode + 1)
            self.writter.add_scalar('Train/Average Leisure Time/Total_Var', var_leisure, episode + 1)
            self.writter.add_scalar('Train/Average Leisure Time/Hired', avg0_leisure, episode + 1)
            self.writter.add_scalar('Train/Average Leisure Time/Hired_Var', var_leisure0, episode + 1)
            self.writter.add_scalar('Train/Average Leisure Time/Crowdsourced', avg1_leisure, episode + 1)
            self.writter.add_scalar('Train/Average Leisure Time/Crowdsourced_Var', var_leisure1, episode + 1)
            
            # ---------------------
            # average courier utilization time
            avg0_running = np.mean(Hired_running_time) / 60
            var_running0 = np.var(Hired_running_time) / 60**2
            avg1_running = np.mean(Crowdsourced_running_time) / 60
            var_running1 = np.var(Crowdsourced_running_time) / 60**2
            avg_running = np.mean(Hired_running_time + Crowdsourced_running_time) / 60
            var_running = np.var(Hired_running_time + Crowdsourced_running_time) / 60**2
            print(f"Hired running time is {avg0_running} minutes (Var: {var_running0}), Crowdsourced running time is {avg1_running} minutes (Var: {var_running1}) and Total running time per courier is {avg_running} minutes (Var: {var_running})")
            self.writter.add_scalar('Train/Average running Time/Total', avg_running, episode + 1)
            self.writter.add_scalar('Train/Average running Time/Total_Var', var_running, episode + 1)
            self.writter.add_scalar('Train/Average running Time/Hired', avg0_running, episode + 1)
            self.writter.add_scalar('Train/Average running Time/Hired_Var', var_running0, episode + 1)
            self.writter.add_scalar('Train/Average running Time/Crowdsourced', avg1_running, episode + 1)
            self.writter.add_scalar('Train/Average running Time/Crowdsourced_Var', var_running1, episode + 1)

            # ---------------------
            # average courier congestion time
            avg0_congestion = np.mean(Hired_congestion_time) / 60
            var_congestion0 = np.var(Hired_congestion_time) / 60**2
            avg1_congestion = np.mean(Crowdsourced_congestion_time) / 60
            var_congestion1 = np.var(Crowdsourced_congestion_time) / 60**2
            avg_congestion = np.mean(Hired_congestion_time + Crowdsourced_congestion_time) / 60
            var_congestion = np.var(Hired_congestion_time + Crowdsourced_congestion_time) / 60**2
            print(f"Hired congestion time is {avg0_congestion} minutes (Var: {var_congestion0}), Crowdsourced congestion time is {avg1_congestion} minutes (Var: {var_congestion1}) and Total congestion time per courier is {avg_congestion} minutes (Var: {var_congestion})")
            self.writter.add_scalar('Train/Average congestion Time/Total', avg_congestion, episode + 1)
            self.writter.add_scalar('Train/Average congestion Time/Total_Var', var_congestion, episode + 1)
            self.writter.add_scalar('Train/Average congestion Time/Hired', avg0_congestion, episode + 1)
            self.writter.add_scalar('Train/Average congestion Time/Hired_Var', var_congestion0, episode + 1)
            self.writter.add_scalar('Train/Average congestion Time/Crowdsourced', avg1_congestion, episode + 1)
            self.writter.add_scalar('Train/Average congestion Time/Crowdsourced_Var', var_congestion1, episode + 1)

            # ---------------------
            # average courier waiting time
            avg0_waiting = np.mean(Hired_waiting_time) / 60
            var_waiting0 = np.var(Hired_waiting_time) / 60**2
            avg1_waiting = np.mean(Crowdsourced_waiting_time) / 60
            var_waiting1 = np.var(Crowdsourced_waiting_time) / 60**2
            avg_waiting = np.mean(Hired_waiting_time + Crowdsourced_waiting_time) / 60
            var_waiting = np.var(Hired_waiting_time + Crowdsourced_waiting_time) / 60**2
            print(f"Hired waiting time is {avg0_waiting} minutes (Var: {var_waiting0}), Crowdsourced waiting time is {avg1_waiting} minutes (Var: {var_waiting1}) and Total waiting time per courier is {avg_waiting} minutes (Var: {var_waiting})")
            self.writter.add_scalar('Train/Average waiting Time/Total', avg_waiting, episode + 1)
            self.writter.add_scalar('Train/Average waiting Time/Total_Var', var_waiting, episode + 1)
            self.writter.add_scalar('Train/Average waiting Time/Hired', avg0_waiting, episode + 1)
            self.writter.add_scalar('Train/Average waiting Time/Hired_Var', var_waiting0, episode + 1)
            self.writter.add_scalar('Train/Average waiting Time/Crowdsourced', avg1_waiting, episode + 1)
            self.writter.add_scalar('Train/Average waiting Time/Crowdsourced_Var', var_waiting1, episode + 1)

            # ---------------------
            # actual speed
            actual0_speed = np.mean(Hired_actual_speed)
            var0_speed = np.var(Hired_actual_speed)
            actual1_speed = np.mean(Crowdsourced_actual_speed)
            var1_speed = np.var(Crowdsourced_actual_speed)
            actual_speed = np.mean(Hired_actual_speed + Crowdsourced_actual_speed)
            var_speed = np.var(Hired_actual_speed + Crowdsourced_actual_speed)
            print(
                f"Hired average speed is {actual0_speed} m/s (Var: {var0_speed}), "
                f"Crowdsourced average speed is {actual1_speed} m/s (Var: {var1_speed}), "
                f"Average speed per courier is {actual_speed} m/s (Var: {var_speed})"
            )
            self.writter.add_scalar('Train/Average Speed/Total', actual_speed, episode + 1)
            self.writter.add_scalar('Train/Average Speed/Total_Var', var_speed, episode + 1)
            self.writter.add_scalar('Train/Average Speed/Hired', actual0_speed, episode + 1)
            self.writter.add_scalar('Train/Average Speed/Hired_Var', var0_speed, episode + 1)
            self.writter.add_scalar('Train/Average Speed/Crowdsourced', actual1_speed, episode + 1)
            self.writter.add_scalar('Train/Average Speed/Crowdsourced_Var', var1_speed, episode + 1)

            # ---------------------
            # overspeed rate
            overspeed = np.mean(overspeed0_step + overspeed1_step)
            overspeed0 = np.mean(overspeed0_step)
            overspeed1 = np.mean(overspeed1_step)
            # overspeed_penalty = np.floor(np.sum(overspeed0_step + overspeed1_step)) * 50
            print(f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total rate - {overspeed}")
            self.writter.add_scalar('Train/Overspeed Rate/Total rate', overspeed, episode + 1)
            self.writter.add_scalar('Train/Overspeed Rate/Hired', overspeed0, episode + 1)
            self.writter.add_scalar('Train/Overspeed Rate/Crowdsourced', overspeed1, episode + 1)
            # self.writter.add_scalar('Overspeed Rate/Overspeed Penalty', overspeed_penalty, episode + 1)
            
            # ---------------------
            # average courier income
            income0 = np.mean(Hired_income)
            var_income0 = np.var(Hired_income)
            income1 = np.mean(Crowdsourced_income) if np.sum(Crowdsourced_income) != 0 else 0
            var_income1 = np.var(Crowdsourced_income) if np.sum(Crowdsourced_income) != 0 else 0
            income = np.mean(Hired_income + Crowdsourced_income)
            var_income = np.var(Hired_income + Crowdsourced_income)
            print(f"Hired's average income is {income0} dollar (Var: {var_income0}), Crowdsourced's average income is {income1} dollar (Var: {var_income1}) and Total income per courier is {income} dollar (Var: {var_income})")
            self.writter.add_scalar('Train/Average Income/Total', income, episode + 1)
            self.writter.add_scalar('Train/Average Income/Total_Var', var_income, episode + 1)
            self.writter.add_scalar('Train/Average Income/Hired', income0, episode + 1)
            self.writter.add_scalar('Train/Average Income/Hired_Var', var_income0, episode + 1)
            self.writter.add_scalar('Train/Average Income/Crowdsourced', income1, episode + 1)
            self.writter.add_scalar('Train/Average Income/Crowdsourced_Var', var_income1, episode + 1)
            
            # ---------------------
            # platform cost
            print(f"The platform total cost is {platform_cost} dollar")
            self.writter.add_scalar('Train/Platform Cost', platform_cost, episode + 1)
            
            # ---------------------
            # order reject rate
            reject_rate_per_episode = count_reject_orders / order_num # reject once or twice or more
            print(f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most")
            self.writter.add_scalar('Train/Reject rate', reject_rate_per_episode, episode + 1)
            
            # ---------------------
            # average waiting time for orders
            waiting_time_per_order = np.mean(order_waiting_time)
            var_waiting_time = np.var(order_waiting_time)
            print(f"The average waiting time for orders ({order_num - order_wait}) is {waiting_time_per_order} dollar (Var: {var_price})")
            self.writter.add_scalar('Train/Average Order Waiting Time/Total', waiting_time_per_order, episode + 1)
            self.writter.add_scalar('Train/Average Order Waiting Time/Total_Var', var_waiting_time, episode + 1)

            # ---------------------
            # average order price for courier
            price_per_order = np.mean(order_price)
            var_price = np.var(order_price)
            print(f"The average price of the order is {price_per_order} dollar (Var: {var_price})")
            self.writter.add_scalar('Train/Average Price/Total', price_per_order, episode + 1)
            self.writter.add_scalar('Train/Average Price/Total_Var', var_price, episode + 1)

            message = (
                f"\nThis is Train Episode {episode+1}\n"
                
                f"There are {courier_num} couriers ({Hired_num} Hired, {Crowdsourced_num} Crowdsourced with {Crowdsourced_on} ({round(100 * Crowdsourced_on / Crowdsourced_num, 2)}%) on), and {order_num} Orders ({count_dropped_orders} dropped, {count_unfinished_orders} unfinished, {order_wait} ({round(100 * order_wait / order_num, 2)}%) waiting to be paired\n"
                
                f"Total Reward for Episode {episode+1}: {int(episode_reward_sum)}\n"
                 
                f"Average Travel Distance for Episode {episode+1}: Hired ({len(Hired_distance_per_episode)}) - {distance0} km (Var: {distance_var0}), Crowdsourced ({len(Crowdsourced_distance_per_episode)}) - {distance1} km (Var: {distance_var1}), Total ({len(Hired_distance_per_episode+Crowdsourced_distance_per_episode)}) - {distance} km (Var: {distance_var})\n"
                                
                f"The average finished number for Episode {episode+1}: Hired ({len(Hired_finished_num)}) - {avg_finish0} (Var: {var_finish0}), Crowdsourced ({len(Crowdsourced_finished_num)}) - {avg_finish1} (Var: {var_finish1}), Total ({len(Hired_finished_num+Crowdsourced_finished_num)}) - {avg_finish} (Var: {var_finish})\n"
                
                f"The average unfinished number for Episode {episode+1}: Hired ({len(Hired_unfinished_num)}) - {avg_unfinish0} (Var: {var_unfinish0}), Crowdsourced ({len(Crowdsourced_unfinished_num)}) - {avg_unfinish1} (Var: {var_unfinish1}), Total ({len(Hired_unfinished_num+Crowdsourced_unfinished_num)}) - {avg_unfinish} (Var: {var_unfinish})\n"
                
                f"The average rejection number of couriers for Episode {episode+1}: Hired - {avg_reject0} (Var: {var_reject0}), Crowdsourced - {avg_reject1} (Var: {var_reject1}), Total - {avg_reject} (Var: {var_reject})\n"
                
                f"The average leisure time for Episode {episode+1}: Hired ({len(Hired_leisure_time)}) - {avg0_leisure} minutes (Var: {var_leisure0}), Crowdsourced ({len(Crowdsourced_leisure_time)}) - {avg1_leisure} minutes (Var: {var_leisure1}), Total ({len(Hired_leisure_time+Crowdsourced_leisure_time)}) - {avg_leisure} minutes (Var: {var_leisure})\n"
                
                f"The average running time for Episode {episode+1}: Hired ({len(Hired_running_time)}) - {avg0_running} minutes (Var: {var_running0}), Crowdsourced ({len(Crowdsourced_running_time)}) - {avg1_running} minutes (Var: {var_running1}), Total ({len(Hired_running_time+Crowdsourced_running_time)}) - {avg_running} minutes (Var: {var_running})\n"
            
                f"The average congestion time for Episode {episode+1}: Hired ({len(Hired_congestion_time)}) - {avg0_congestion} minutes (Var: {var_congestion0}), Crowdsourced ({len(Crowdsourced_congestion_time)}) - {avg1_congestion} minutes (Var: {var_congestion1}), Total ({len(Hired_congestion_time+Crowdsourced_congestion_time)}) - {avg_congestion} minutes (Var: {var_congestion})\n"
            
                f"The average waiting time for Episode {episode+1}: Hired ({len(Hired_waiting_time)}) - {avg0_waiting} minutes (Var: {var_waiting0}), Crowdsourced ({len(Crowdsourced_waiting_time)}) - {avg1_waiting} minutes (Var: {var_waiting1}), Total ({len(Hired_waiting_time+Crowdsourced_waiting_time)}) - {avg_waiting} minutes (Var: {var_waiting})\n"

                f"The actual average speed for Episode {episode+1}: Hired ({len(Hired_actual_speed)}) - {actual0_speed} m/s (Var: {var0_speed}), Crowdsourced ({len(Crowdsourced_actual_speed)}) - {actual1_speed} m/s (Var: {var1_speed}), Total ({len(Hired_actual_speed+Crowdsourced_actual_speed)}) - {actual_speed} m/s (Var: {var_speed})\n"
                
                f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total - {overspeed}\n"
                
                f"The average income for Episode {episode+1}: Hired ({len(Hired_income)}) - {income0} dollar (Var: {var_income0}), Crowdsourced ({len(Crowdsourced_income)}) - {income1} dollar (Var: {var_income1}), Total ({len(Hired_income+Crowdsourced_income)}) - {income} dollar (Var: {var_income})\n"
                
                f"The platform total cost is {platform_cost} dollar\n"
                
                f"Order rejection rate for Episode {episode+1}: {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most\n"
                
                f"The average waiting time for orders ({order_num - order_wait}) is {waiting_time_per_order} dollar (Var: {var_price})\n"
                
                f"The average price for Episode {episode+1}: Total ({len(order_price)}) - {price_per_order} dollar (Var: {var_price})\n"
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
                unfinished = count_unfinished_orders / (order_num - order_wait)
                unfinished_late_rate = unfinished_late_orders / count_unfinished_orders
                print(f"Unfinished Orders for Episode {episode+1} is {count_unfinished_orders} out of {order_num - order_wait} orders ({unfinished}), with {unfinished_late_rate} being late")
                
                message += f"Unfinished Orders for Episode {episode+1} is {count_unfinished_orders} out of {order_num - order_wait} orders ({unfinished}), with {unfinished_late_rate} being late\n"
                logger.success(message)
                self.writter.add_scalar('Train/Unfinished Orders Rate', unfinished, episode + 1)
                self.writter.add_scalar('Train/Unfinished Late Rate', unfinished_late_rate, episode + 1)
            

            # social_welfare = sum(Hired_distance_per_episode + Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
            # print(f"Social welfare is {social_welfare} dollar\n")
            # message += f"Social welfare is {social_welfare} dollar\n"
            # logger.success(message)
            # self.writter.add_scalar('Social Welfare', social_welfare, episode + 1)
            
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

                self.log_train(train_infos, total_num_steps)

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
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                np.array(list(share_obs[:, agent_id])),
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
                share_obs1.append(np.array(list(share_obs[:, agent_id])))
                obs1.append(np.array(list(obs[:, agent_id])))
                rnn_states1.append(rnn_states[:, agent_id])
                rnn_states_critic1.append(rnn_states_critic[:, agent_id])
                actions1.append(actions[:, agent_id])
                action_log_probs1.append(action_log_probs[:, agent_id])
                values1.append(values[:, agent_id])
                rewards1.append(rewards[:, agent_id].reshape(-1,1))
                masks1.append(masks[:, agent_id])
            else:
                share_obs2.append(np.array(list(share_obs[:, agent_id])))
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
        # eval_obs = self.eval_envs.reset(5)
        
        # eval info
        stats = {i: {
            "platform_cost": 0,
            "Hired_finish_num": [],
            "Hired_unfinish_num": [],
            "Hired_leisure_time": [],
            "Hired_running_time": [],
            "Hired_congestion_time": [],
            "Hired_waiting_time": [],
            "Crowdsourced_finish_num": [],
            "Crowdsourced_unfinish_num": [],
            "Crowdsourced_leisure_time": [],
            "Crowdsourced_running_time": [],
            "Crowdsourced_congestion_time": [],
            "Crowdsourced_waiting_time": [],

            "courier_num": 0,
            "Hired_num": 0,
            "Crowdsourced_num": 0,
            "Crowdsourced_on": 0,
            "Hired_distance_per_episode": [],
            "Crowdsourced_distance_per_episode": [],
            "Hired_actual_speed": [],
            "Hired_income": [],
            "Crowdsourced_actual_speed": [],
            "Crowdsourced_income": [],
            "overspeed_step": {"num0": [], "num1": []}, 
            
            "order_num": 0,
            "count_dropped_orders": 0,
            "count_unfinished_orders": 0,
            "unfinished_late_orders": 0,
            "late_orders": 0,
            "ETA_usage": [],
            "order_price": [],
            "order_wait": 0,
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
                
                # print(f"ENVIRONMENT {i+1}")

                # print("Couriers:")
                # for c in self.eval_envs.envs_map[i].couriers:
                #     if c.state == 'active':
                #         print(c)
                # print("Orders:")
                # for o in self.eval_envs.envs_map[i].orders:
                #     print(o)  
                # print("\n")
                
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
            eval_obs, eval_rewards, eval_dones, eval_infos, eval_share_obs = self.eval_envs.step(eval_actions_env)
            
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.eval_num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            algo_stats = {i: {"num0": 0, "num1": 0, "count0": 0, "count1": 0} for i in range(self.eval_envs.num_envs)}

            for i in range(self.eval_envs.num_envs):
                for c in self.eval_envs.envs_map[i].couriers:
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

                stats[i]["overspeed_step"]["num0"].append(overspeed_ratio0)
                stats[i]["overspeed_step"]["num1"].append(overspeed_ratio1) 
                          
            eval_obs = self.eval_envs.eval_env_step()

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
                stats[i][f"{category}_leisure_time"].append(c.total_leisure_time)
                stats[i][f"{category}_running_time"].append(c.total_running_time)
                stats[i][f"{category}_congestion_time"].append(c.total_congestion_time)
                stats[i][f"{category}_waiting_time"].append(c.total_waiting_time)

                if c.actual_speed > 0:
                    stats[i][f"{category}_actual_speed"].append(c.actual_speed)

                if c.income > 0:
                    stats[i][f"{category}_income"].append(
                        c.income / (c.total_running_time + c.total_leisure_time) * 3600
                    )

                if category == "Crowdsourced" and c.state == "active":
                    stats[i]["Crowdsourced_on"] += 1

            stats[i]["courier_num"] = len(env.couriers)

            for o in env.orders:
                if o.status == "dropped":
                    stats[i]["count_dropped_orders"] += 1
                    if o.is_late == 1:
                        stats[i]["late_orders"] += 1
                    else:
                        stats[i]["ETA_usage"].append(o.ETA_usage)
                else:
                    stats[i]["count_unfinished_orders"] += 1
                    if o.ETA <= self.envs.envs_map[0].clock:
                        stats[i]["unfinished_late_orders"] += 1

                if o.status == "wait_pair":
                    stats[i]["order_wait"] += 1
                else:
                    stats[i]["order_waiting_time"].append(o.wait_time)
                    stats[i]["order_price"].append(o.price)

            stats[i]["order_num"] = len(env.orders)
            
        for algo_num in range(self.eval_envs.num_envs):
            data = stats[algo_num]
            
            print(f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished, {data['order_wait']} ({round(100 * data['order_wait'] / data['order_num'], 2)}%) Orders waiting to be paired)")

            # -----------------------
            # Distance
            hired_distance = np.mean(data["Hired_distance_per_episode"]) / 1000
            var_hired_distance = np.var(data["Hired_distance_per_episode"]) / 1000000
            crowdsourced_distance = np.mean(data["Crowdsourced_distance_per_episode"]) / 1000
            var_crowdsourced_distance = np.var(data["Crowdsourced_distance_per_episode"]) / 1000000
            total_distance = np.mean(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000
            var_total_distance = np.var(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000000
            total_courier_num = data['courier_num']

            print(f"In Algo{algo_num + 1}, Total couriers: {total_courier_num}")            
            print(f"\nIn Algo{algo_num + 1}, Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})")
            
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired', hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced', crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total', total_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired Var', var_hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced Var', var_crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total Var', var_total_distance, self.eval_num)
            
            # -----------------------
            # Average Courier Finishing Number
            hired_finish_num = data["Hired_finish_num"]
            crowdsourced_finish_num = data["Crowdsourced_finish_num"]
            
            finish0 = np.mean(hired_finish_num)
            var0_finish = np.var(hired_finish_num)
            finish1 = np.mean(crowdsourced_finish_num)
            var1_finish = np.var(crowdsourced_finish_num)
            total_finish = np.mean(hired_finish_num + crowdsourced_finish_num)
            var_finish = np.var(hired_finish_num + crowdsourced_finish_num)

            print(f"Average Finished Orders per Courier for Algo{algo_num + 1}:")
            print(f"Hired finishes average {finish0} orders (Var: {var0_finish}), Crowdsourced finishes average {finish1} orders (Var: {var1_finish}), Total finish number per courier is {total_finish} orders (Var: {var_finish})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Total', total_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Hired', finish0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Crowdsourced', finish1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Total Var', var_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Hired Var', var0_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Crowdsourced Var', var1_finish, self.eval_num)
                    
            # -----------------------
            # Average Courier unfinishing Number
            hired_unfinish_num = data["Hired_unfinish_num"]
            crowdsourced_unfinish_num = data["Crowdsourced_unfinish_num"]
            
            unfinish0 = np.mean(hired_unfinish_num)
            var0_unfinish = np.var(hired_unfinish_num)
            unfinish1 = np.mean(crowdsourced_unfinish_num)
            var1_unfinish = np.var(crowdsourced_unfinish_num)
            total_unfinish = np.mean(hired_unfinish_num + crowdsourced_unfinish_num)
            var_unfinish = np.var(hired_unfinish_num + crowdsourced_unfinish_num)

            print(f"Average unfinished Orders per Courier for Algo{algo_num+1}:")
            print(f"Hired unfinishes average {unfinish0} orders (Var: {var0_unfinish}), Crowdsourced unfinishes average {unfinish1} orders (Var: {var1_unfinish}), Total unfinish number per courier is {total_unfinish} orders (Var: {var_unfinish})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Total', total_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Hired', unfinish0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Crowdsourced', unfinish1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Total Var', var_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Hired Var', var0_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Crowdsourced Var', var1_unfinish, self.eval_num)


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
            leisure_courier_num = len(Hired_leisure_time + Crowdsourced_leisure_time)

            print(f"Average leisure time per courier for Algo {algo_num+1}:")
            print(f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})")

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
            running_courier_num = len(Hired_running_time + Crowdsourced_running_time)

            print(f"Average running time per courier for Algo {algo_num+1}:")
            print(f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total', avg_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired', hired_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced', Crowdsourced_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total Var', avg_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired Var', hired_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced Var', Crowdsourced_running_var, self.eval_num)

            # -----------------------
            # Average Courier congestion Time
            Hired_congestion_time = data['Hired_congestion_time']
            Crowdsourced_congestion_time = data['Crowdsourced_congestion_time']
            
            hired_congestion = np.mean(Hired_congestion_time) / 60
            hired_congestion_var = np.var(Hired_congestion_time) / 60**2
            Crowdsourced_congestion = np.mean(Crowdsourced_congestion_time) / 60
            Crowdsourced_congestion_var = np.var(Crowdsourced_congestion_time) / 60**2
            
            avg_congestion = np.mean(Hired_congestion_time + Crowdsourced_congestion_time) / 60
            avg_congestion_var = np.var(Hired_congestion_time + Crowdsourced_congestion_time) / 60**2
            congestion_courier_num = len(Hired_congestion_time + Crowdsourced_congestion_time)

            print(f"Average congestion time per courier for Algo {algo_num+1}:")
            print(f"Hired congestion time is {hired_congestion} minutes (Var: {hired_congestion_var}), Crowdsourced congestion time is {Crowdsourced_congestion} minutes (Var: {Crowdsourced_congestion_var}), Total congestion time per courier is {avg_congestion} minutes (Var: {avg_congestion_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Total', avg_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Hired', hired_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Crowdsourced', Crowdsourced_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Total Var', avg_congestion_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Hired Var', hired_congestion_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Crowdsourced Var', Crowdsourced_congestion_var, self.eval_num)
        
            # -----------------------
            # Average Courier waiting Time
            Hired_waiting_time = data['Hired_waiting_time']
            Crowdsourced_waiting_time = data['Crowdsourced_waiting_time']
            
            hired_waiting = np.mean(Hired_waiting_time) / 60
            hired_waiting_var = np.var(Hired_waiting_time) / 60**2
            Crowdsourced_waiting = np.mean(Crowdsourced_waiting_time) / 60
            Crowdsourced_waiting_var = np.var(Crowdsourced_waiting_time) / 60**2
            
            avg_waiting = np.mean(Hired_waiting_time + Crowdsourced_waiting_time) / 60
            avg_waiting_var = np.var(Hired_waiting_time + Crowdsourced_waiting_time) / 60**2
            waiting_courier_num = len(Hired_waiting_time + Crowdsourced_waiting_time)

            print(f"Average waiting time per courier for Algo {algo_num+1}:")
            print(f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Total', avg_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Hired', hired_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Crowdsourced', Crowdsourced_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Total Var', avg_waiting_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Hired Var', hired_waiting_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Crowdsourced Var', Crowdsourced_waiting_var, self.eval_num)
            
            # -----------------------
            # Average Speed
            Hired_actual_speed = data['Hired_actual_speed']
            Crowdsourced_actual_speed = data['Crowdsourced_actual_speed']

            hired_speed = np.mean(Hired_actual_speed)
            hired_speed_var = np.var(Hired_actual_speed)
            crowdsourced_speed = np.mean(Crowdsourced_actual_speed)
            crowdsourced_speed_var = np.var(Crowdsourced_actual_speed)
            avg_speed = np.mean(Hired_actual_speed + Crowdsourced_actual_speed)
            avg_speed_var = np.var(Hired_actual_speed + Crowdsourced_actual_speed)

            print(f"Average speed per courier for Algo{algo_num+1}:")
            print(f"Hired average speed is {hired_speed} m/s (Var: {hired_speed_var}), Crowdsourced average speed is {crowdsourced_speed} m/s (Var: {crowdsourced_speed_var}), Total average speed per courier is {avg_speed} m/s (Var: {avg_speed_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Total', avg_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Hired', hired_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Crowdsourced', crowdsourced_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Total Var', avg_speed_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Hired Var', hired_speed_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Crowdsourced Var', crowdsourced_speed_var, self.eval_num)
            
            # -----------------------
            # Overspeed
            Hired_overspeed = data['Hired_overspeed']
            Crowdsourced_overspeed = data['Crowdsourced_overspeed']
            hired_overspeed = np.mean(Hired_overspeed)
            crowdsourced_overspeed = np.mean(Crowdsourced_overspeed)
            total_overspeed = np.mean(Hired_overspeed + Crowdsourced_overspeed)

            print(f"Rate of Overspeed for Evaluation for Algo{algo_num+1}:")
            print(f"Hired overspeed rate is {hired_overspeed}, Crowdsourced overspeed rate is {crowdsourced_overspeed}, Total overspeed rate is {total_overspeed}")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Total', total_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Hired', hired_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Crowdsourced', crowdsourced_overspeed, self.eval_num)

            # -----------------------
            # Average Courier Income
            hired_income = data['Hired_income']
            crowdsourced_income = data['Crowdsourced_income']
            hired_income = np.mean(hired_income)
            crowdsourced_income = np.mean(crowdsourced_income)
            total_income = np.mean(hired_income + crowdsourced_income)
            hired_income_var = np.var(hired_income)
            crowdsourced_income_var = np.var(crowdsourced_income)
            total_income_var = np.var(hired_income + crowdsourced_income)

            print(f"Average Income per Courier for Algo{algo_num+1}:")
            print(f"Total: Hired's average income is {hired_income} dollars (Var: {hired_income_var}), Crowdsourced's average income is {crowdsourced_income} dollars (Var: {crowdsourced_income_var}), Total income per courier is {total_income} dollars (Var: {total_income_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Total', total_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Hired', hired_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Crowdsourced', crowdsourced_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Total Var', total_income_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Hired Var', hired_income_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Crowdsourced Var', crowdsourced_income_var, self.eval_num)
            
            # -----------------------
            # Platform cost
            platform_cost = data['platform_cost']
            print(f"The platform cost for Algo{algo_num+1} is {platform_cost} dollars.")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Platform Cost', platform_cost, self.eval_num)

            # -----------------------
            # Average Order Price
            order_price = data['order_price']

            order_price_per_order = np.mean(order_price)
            order_price_var = np.var(order_price)

            print(f"Average Price per Order for Algo{algo_num+1}:")
            print(f"Total average is {order_price_per_order} dollars (Var: {order_price_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Price/Total', order_price_per_order, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Price/Total Var', order_price_var, self.eval_num)
            
            message = (
                f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished, {data['order_wait']} ({round(100 * data['order_wait'] / data['order_num'], 2)}%) Orders waiting to be paired)\n"
                                
                f"Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})\n"
                
                f"Hired finishes average {finish0} orders (Var: {var0_finish}), Crowdsourced finishes average {finish1} orders (Var: {var1_finish}), Total finish number per courier is {total_finish} orders (Var: {var_finish})\n"
                
                f"Hired unfinishes average {unfinish0} orders (Var: {var0_unfinish}), Crowdsourced unfinishes average {unfinish1} orders (Var: {var1_unfinish}), Total unfinish number per courier is {total_unfinish} orders (Var: {var_unfinish})\n"
                
                f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})\n"
                
                f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})\n"

                f"Hired congestion time is {hired_congestion} minutes (Var: {hired_congestion_var}), Crowdsourced congestion time is {Crowdsourced_congestion} minutes (Var: {Crowdsourced_congestion_var}), Total congestion time per courier is {avg_congestion} minutes (Var: {avg_congestion_var})\n"
                
                f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})\n"
                
                f"Hired average speed is {hired_speed} m/s (Var: {hired_speed_var}), Crowdsourced average speed is {crowdsourced_speed} m/s (Var: {crowdsourced_speed_var}), Total average speed per courier is {avg_speed} m/s (Var: {avg_speed_var})\n"

                f"Hired overspeed rate is {hired_overspeed}, Crowdsourced overspeed rate is {crowdsourced_overspeed}, Total overspeed rate is {total_overspeed}\n"     
                
                f"Total: Hired's average income is {hired_income} dollars (Var: {hired_income_var}), Crowdsourced's average income is {crowdsourced_income} dollars (Var: {crowdsourced_income_var}), Total income per courier is {total_income} dollars (Var: {total_income_var})\n"                
                           
                f"Total average is {order_price_per_order} dollars (Var: {order_price_var})\n"
                                
                f"The platform1 total cost is {platform_cost} dollar\n"
                
            )

        if data['count_dropped_orders'] == 0:
            print(f"No order is dropped in Algo{algo_num+1}")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', -1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', -1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var/Crowdsourced Var', 0, self.eval_num)
            
            message += "No order is dropped in Algo1\n"
        else:                
            late_rate = data['late_orders'] / data['count_dropped_orders']     
            ETA_usage_rate = np.mean(data['ETA_usage0'])
            var_ETA = np.var(data['ETA_usage0'])
            print(f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders")
            print(f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})")
            
            message += f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders\n" + f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})\n"

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', late_rate, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', ETA_usage_rate, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var', var_ETA, self.eval_num)
        
        if data['count_unfinished_orders'] == 0:
            print(f"No order is unfinished in Algo{algo_num+1}")
            message += f"No order is unfinished in Algo{algo_num+1}\n"
            logger.success(message)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', 0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', 0, self.eval_num)
        else:
            unfinished = data['count_unfinished_orders'] / (data['order_num'] - data['order_wait'])
            unfinished_late_rate = data['unfinished_late_orders'] / data['count_unfinished_orders']
            print(f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num'] - data['order_wait']} orders ({unfinished}), with {unfinished_late_rate} being late")
            
            message += f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num'] - data['order_wait']} orders ({unfinished}), with {unfinished_late_rate} being late\n"
            logger.success(message)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', unfinished, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', unfinished_late_rate, self.eval_num)
           
           
        

        # algo1_social_welfare = sum(algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo2_social_welfare = sum(algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo3_social_welfare = sum(algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo4_social_welfare = sum(algo4_Hired_distance_per_episode + algo4_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # algo5_social_welfare = sum(algo5_Hired_distance_per_episode + algo5_Crowdsourced_distance_per_episode) / 1000 * 0.6214 * 404 / 1e6 * 105
        # print(f"Algo1: The platform total cost is {platform_cost1, 2)} dollar, and social welfare is {algo1_social_welfare} dollar")
        # print(f"Algo2: The platform total cost is {platform_cost2, 2)} dollar, and social welfare is {algo2_social_welfare} dollar")
        # print(f"Algo3: The platform total cost is {platform_cost3, 2)} dollar, and social welfare is {algo3_social_welfare} dollar")
        # print(f"Algo4: The platform total cost is {platform_cost4, 2)} dollar, and social welfare is {algo4_social_welfare} dollar")
        # print(f"Algo5: The platform total cost is {platform_cost5, 2)} dollar, and social welfare is {algo5_social_welfare} dollar")
        # message += f"Algo1: Social welfare is {algo1_social_welfare} dollar\n" + f"Algo2: Social welfare is {algo2_social_welfare} dollar\n" + f"Algo3: Social welfare is {algo3_social_welfare} dollar\n" + f"Algo4: Social welfare is {algo4_social_welfare} dollar\n" + f"Algo5: Social welfare is {algo5_social_welfare} dollar\n"
        # self.writter.add_scalar('Social Welfare/Algo1', algo1_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo2', algo2_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo3', algo3_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo4', algo4_social_welfare, self.eval_num)
        # self.writter.add_scalar('Social Welfare/Algo5', algo5_social_welfare, self.eval_num)

        logger.success(message)
            
        print("\n")