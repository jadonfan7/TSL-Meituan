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

        distance_total = []
        episode_rewards = []
        rate_of_overspeed = []
        rate_of_late_order = []
        rate_of_ETA_usage = []
        reject_rate = []
        order_price_total = []
        courier_reject_num_total = []
        courier_finish_num_total = []
        leisure_time_total = []
        avg_speed_total = []
        income_total = []
        
        algo1_eval_distance = []
        algo2_eval_distance = []
        algo3_eval_distance = []
        algo1_eval_episode_rewards = []
        algo2_eval_episode_rewards = []
        algo3_eval_episode_rewards = []
        algo1_eval_speed = []
        algo2_eval_speed = []
        algo3_eval_speed = []
        algo1_eval_overspeed_rate = []
        algo2_eval_overspeed_rate = []
        algo3_eval_overspeed_rate = []
        algo1_eval_reject_rate = []
        algo2_eval_reject_rate = []
        algo3_eval_reject_rate = []
        algo1_eval_reject = []
        algo2_eval_reject = []
        algo3_eval_reject = []
        algo1_eval_order_price = []
        algo2_eval_order_price = []
        algo3_eval_order_price = []
        algo1_eval_income = []
        algo2_eval_income = []
        algo3_eval_income = []
        algo1_eval_finish = []
        algo2_eval_finish = []
        algo3_eval_finish = []
        algo1_eval_leisure = [] 
        algo2_eval_leisure = []
        algo3_eval_leisure = [] 
        algo1_rate_of_late_order = []
        algo2_rate_of_late_order = []
        algo3_rate_of_late_order = []
        algo1_rate_of_ETA_usage = []
        algo2_rate_of_ETA_usage = []
        algo3_rate_of_ETA_usage = []

        for episode in range(episodes):
            print(f"THE START OF EPISODE {episode+1}")

            Hired_distance_per_episode = 0
            Crowdsourced_distance_per_episode = 0
            Hired_num = 0
            Crowdsourced_num = 0
            Crowdsourced_on = 0

            episode_reward_sum = 0

            count_overspeed0 = 0
            num_active_Hired = 0
            count_overspeed1 = 0
            num_active_Crowdsourced = 0

            count_reject_orders = 0
            max_reject_num = 0

            late_orders0 = 0
            late_orders1 = 0
            
            ETA_usage0 = 0
            ETA_usage1 = 0

            count_dropped_orders0 = 0
            count_dropped_orders1 = 0

            
            order0_price = 0
            order1_price = 0
            order0_num = 0
            order1_num = 0
            order_wait = 0

            Hired_reject_num = 0
            Crowdsourced_reject_num = 0
            
            Hired_finish_num = 0
            Crowdsourced_finish_num = 0
            
            Hired_leisure_time = 0
            Crowdsourced_leisure_time = 0
            
            Hired_avg_speed = 0
            Crowdsourced_avg_speed = 0
            
            Hired_income = 0
            Crowdsourced_income = 0

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
                #     print(f"ENVIRONMENT {i+1}")

                #     print("Couriers:")
                #     for c in self.envs.envs_discrete[i].couriers:
                #         if c.state == 'active':
                #             print(c)
                #     print("Orders:")
                #     for o in self.envs.envs_discrete[i].orders:
                #         print(o)  
                #     print("\n")
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
                for c in self.envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        Hired_num += 1
                        Hired_distance_per_episode += c.travel_distance
                        Hired_reject_num += c.reject_order_num
                        Hired_finish_num += c.finish_order_num
                        Hired_leisure_time += c.total_leisure_time
                        Hired_avg_speed += c.avg_speed
                        Hired_income += c.income
                    else:
                        Crowdsourced_num += 1
                        Crowdsourced_distance_per_episode += c.travel_distance
                        Crowdsourced_reject_num += c.reject_order_num
                        Crowdsourced_finish_num += c.finish_order_num
                        Crowdsourced_leisure_time += c.total_leisure_time
                        Crowdsourced_avg_speed += c.avg_speed
                        Crowdsourced_income += c.income
                        if c.state == 'active':
                            Crowdsourced_on += 1
                
                for o in self.envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            count_dropped_orders0 += 1
                            if o.is_late == 1:
                                late_orders0 += 1
                            else:
                                ETA_usage0 += o.ETA_usage 
                        else:
                            count_dropped_orders1 += 1
                            if o.is_late == 1:
                                late_orders1 += 1
                            else:
                                ETA_usage1 += o.ETA_usage 
                        
                    if o.reject_count > 0:
                        count_reject_orders += 1
                        if max_reject_num <= o.reject_count:
                            max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            order0_price += o.price
                            order0_num += 1
                        else:
                            order1_price += o.price
                            order1_num += 1              
                            
            print(f"\nThere are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced with {Crowdsourced_on / self.envs.num_envs} on, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1, {order_wait / self.envs.num_envs} Orders waiting to be paired")                
            episode_rewards.append(episode_reward_sum)
            print(f"Total Reward for Episode {episode+1}: {episode_reward_sum}")
            self.writter.add_scalar('Total Reward', episode_reward_sum, episode + 1)
                                    
            distance0 = round(Hired_distance_per_episode / Hired_num, 2)
            distance1 = round(Crowdsourced_distance_per_episode / Crowdsourced_num, 2)
            distance = (Hired_distance_per_episode + Crowdsourced_distance_per_episode) / (Hired_num + Crowdsourced_num)
            distance = round(distance, 2)
            distance_total.append([distance0, distance1, distance])
            print(f"Average Travel Distance per Hired: {distance0} meters, Crowdsourced: {distance1} meters, Total: {distance} meters")
            self.writter.add_scalar('Total Distance/Total', distance, episode + 1)
            self.writter.add_scalar('Total Distance/Hired', distance0, episode + 1)
            self.writter.add_scalar('Total Distance/Crowdsourced', distance1, episode + 1)
            
            avg0_speed = round(Hired_avg_speed / Hired_num, 2)
            avg1_speed = round(Crowdsourced_avg_speed / Crowdsourced_num, 2)
            avg_speed = (Hired_avg_speed + Crowdsourced_avg_speed) / (Hired_num + Crowdsourced_num)
            avg_speed = round(avg_speed, 2)
            avg_speed_total.append([avg0_speed, avg1_speed, avg_speed])    
            print(f"Hired average speed is {avg0_speed} m/s, Crowdsourced average speed is {avg1_speed} m/s and average speed per courier is {avg_speed} m/s")
            self.writter.add_scalar('Average Speed/Total', avg_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Hired', avg0_speed, episode + 1)
            self.writter.add_scalar('Average Speed/Crowdsourced', avg1_speed, episode + 1)

            overspeed = (count_overspeed0 + count_overspeed1) / (num_active_Hired + num_active_Crowdsourced)
            overspeed = round(overspeed, 2)
            overspeed0 = round(count_overspeed0 / num_active_Hired, 2)
            overspeed1 = round(count_overspeed1 / num_active_Crowdsourced, 2)
            print(f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total rate - {overspeed}")
            rate_of_overspeed.append([overspeed0, overspeed1, overspeed])
            self.writter.add_scalar('Overspeed Rate/Total rate', overspeed, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Hired', overspeed0, episode + 1)
            self.writter.add_scalar('Overspeed Rate/Crowdsourced', overspeed1, episode + 1)
            
            reject_rate_per_episode = round(count_reject_orders / (order_wait + order0_num + order1_num), 2) # reject once or twice or more
            reject_rate.append(reject_rate_per_episode)
            print(f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most")
            self.writter.add_scalar('Reject rate', reject_rate_per_episode, episode + 1)
            
            reject0 = round(Hired_reject_num / Hired_num, 2)
            reject1 = round(Crowdsourced_reject_num / Crowdsourced_num, 2)
            reject = (Crowdsourced_reject_num + Crowdsourced_reject_num) / (Hired_num + Crowdsourced_num)
            reject = round(reject, 2)
            courier_reject_num_total.append([reject0, reject1, reject])
            print(f"Hired rejects average {reject0} orders, Crowdsourced rejects average {reject1} orders and Total reject number per courier is {reject}")
            self.writter.add_scalar('Average Rejection/Total', reject, episode + 1)
            self.writter.add_scalar('Average Rejection/Hired', reject0, episode + 1)
            self.writter.add_scalar('Average Rejection/Crowdsourced', reject1, episode + 1)
        
            price_per_order0 = round(order0_price / order0_num, 2)
            price_per_order1 = round(order1_price / order1_num, 2)
            order_price = (order0_price + order1_price) / (order0_num + order1_num)
            order_price = round(order_price, 2)
            order_price_total.append([price_per_order0, price_per_order1, order_price])
            print(f"The average price of Hired's order is {price_per_order0} yuan with {order0_num} orders, Crowdsourced's is {price_per_order1} yuan with {order1_num} orders and for all is {order_price}")
            self.writter.add_scalar('Average Price/Total', order_price, episode + 1)
            self.writter.add_scalar('Average Price/Hired', order0_price, episode + 1)
            self.writter.add_scalar('Average Price/Crowdsourced', order1_price, episode + 1)
            
            income0 = round(Hired_income / Hired_num, 2)
            income1 = round(Crowdsourced_income / Crowdsourced_num, 2)
            income = round((Hired_income + Crowdsourced_income) / (Hired_num + Crowdsourced_num), 2)
            platform_cost = round((Hired_income + Crowdsourced_income) / self.envs.num_envs, 2)
            income_total.append([income0, income1, income, platform_cost])
            print(f"Hired's average income is {income0} yuan, Crowdsourced's average income is {income1} yuan and Total income per courier is {income} yuan")
            print(f"The platform total cost is {platform_cost} yuan")
            self.writter.add_scalar('Average Income/Total', income, episode + 1)
            self.writter.add_scalar('Average Income/Hired', income0, episode + 1)
            self.writter.add_scalar('Average Income/Crowdsourced', income1, episode + 1)
            self.writter.add_scalar('Platform Cost', platform_cost, episode + 1)

            finish0 = round(Hired_finish_num / Hired_num, 2)
            finish1 = round(Crowdsourced_finish_num / Crowdsourced_num, 2)
            finish = round((Hired_finish_num + Crowdsourced_finish_num) / (Hired_num + Crowdsourced_num), 2)
            courier_finish_num_total.append([finish0, finish1, finish])
            print(f"Hired finishes average {finish0} orders while Crowdsourced finishes average {finish1} orders, Total finish number per courier is {finish}")
            self.writter.add_scalar('Average Finish/Total', finish, episode + 1)
            self.writter.add_scalar('Average Finish/Hired', finish0, episode + 1)
            self.writter.add_scalar('Average Finish/Crowdsourced', finish1, episode + 1)
            
            avg0_leisure = round(Hired_leisure_time / Hired_num / 60, 2)
            avg1_leisure = round(Crowdsourced_leisure_time / Crowdsourced_num / 60, 2)
            avg_leisure = round((Hired_leisure_time + Crowdsourced_leisure_time) / (Hired_num + Crowdsourced_num) / 60, 2)
            leisure_time_total.append([avg0_leisure, avg1_leisure, avg_leisure])
            print(f"Hired leisure time is {avg0_leisure} minutes, Crowdsourced leisure time is {avg1_leisure} minutes and Total leisure time per courier is {avg_leisure} minutes")
            self.writter.add_scalar('Average Leisure Time/Total', avg_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Hired', avg0_leisure, episode + 1)
            self.writter.add_scalar('Average Leisure Time/Crowdsourced', avg1_leisure, episode + 1)
            
            message = (
                f"There are {Hired_num / self.envs.num_envs} Hired, {Crowdsourced_num / self.envs.num_envs} Crowdsourced, {order0_num / self.envs.num_envs} Order0, {order1_num / self.envs.num_envs} Order1\n"
                f"Average Travel Distance for Episode {episode+1}: Hired - {distance0} meters, Crowdsourced - {distance1} meters, Total - {distance} meters\n"
                f"Total Reward for Episode {episode+1}: {episode_reward_sum}\n"
                f"The average speed for Episode {episode+1}: Hired - {avg0_speed} m/s, Crowdsourced - {avg1_speed} m/s, Total - {avg_speed} m/s\n"
                f"Rate of Overspeed for Episode {episode+1}: Hired - {overspeed0}, Crowdsourced - {overspeed1}, Total - {overspeed}\n"
                f"Order rejection rate for Episode {episode+1}: {reject_rate_per_episode} and the order is rejected by {max_reject_num} times at most\n"
                f"The average rejection number for Episode {episode+1}: Hired - {reject0}, Crowdsourced - {reject1}, Total - {reject}\n"
                f"The average price for Episode {episode+1}: Hired - {price_per_order0} yuan with {order0_num} orders, Crowdsourced - {price_per_order1} yuan with {order1_num} orders, Total - {order_price} yuan\n"
                f"The average income for Episode {episode+1}: Hired- {income0} yuan, Crowdsourced - {income1} yuan, Total - {income} yuan\n"
                f"The platform total cost is {platform_cost} yuan\n"
                f"The average finish number for Episode {episode+1}: Hired - {finish0}, Crowdsourced - {finish1}, Total - {finish}\n"
                f"The average leisure time for Episode {episode+1}: Hired - {avg0_leisure} minutes, Crowdsourced - {avg1_leisure} minutes, Total - {avg_leisure} minutes\n"
            )

            if count_dropped_orders0 + count_dropped_orders1 == 0:
                print("No order is dropped in this episode")
                rate_of_late_order.append([-1, -1, -1])
                rate_of_ETA_usage.append([-1, -1, -1])
                message += "No order is dropped in this episode\n"
                logger.success(message)
                self.writter.add_scalar('Late Orders Rate/Total', -1, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Hired', -1, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Crowdsourced', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired', -1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced', -1, episode + 1)
            else:
                if count_dropped_orders0 != 0:
                    late_rate0 = round(late_orders0 / count_dropped_orders0, 2)
                    ETA_usage_rate0 = round(ETA_usage0 / count_dropped_orders0,2)
                else:
                    late_rate0 = -1
                    ETA_usage_rate0 = -1
                    
                if count_dropped_orders1 != 0:                    
                    late_rate1 = round(late_orders1 / count_dropped_orders1, 2)
                    ETA_usage_rate1 = round(ETA_usage1 / count_dropped_orders1,2)
                else:
                    late_rate1 = -1
                    ETA_usage_rate1 = -1
                    
                late_rate = round((late_orders0 + late_orders1) / (count_dropped_orders0 + count_dropped_orders1), 2)
                print(f"Rate of Late Orders for Episode {episode+1}: Hired - {late_rate0}, Crowdsourced - {late_rate1}, Total - {late_rate}")
                rate_of_late_order.append([late_rate0, late_rate1, late_rate])

                ETA_usage_rate = round((ETA_usage0 + ETA_usage1) / (count_dropped_orders0 + count_dropped_orders1),2)
                print(f"Rate of ETA Usage for Episode {episode+1}: Hired- {ETA_usage_rate0}, Crowdsourced - {ETA_usage_rate1}, Total - {ETA_usage_rate}")
                rate_of_ETA_usage.append([ETA_usage_rate0, ETA_usage_rate1, ETA_usage_rate])
                
                message += f"Rate of Late Orders for Episode {episode+1}: Hired - {late_rate0}, Crowdsourced - {late_rate1}, Total - {late_rate}\n" + f"Rate of ETA Usage for Episode {episode+1}: Hired - {ETA_usage_rate0}, Crowdsourced - {ETA_usage_rate1}, Total - {ETA_usage_rate}\n"
                logger.success(message)
                
                self.writter.add_scalar('Late Orders Rate/Total', late_rate, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Hired', late_rate0, episode + 1)
                self.writter.add_scalar('Late Orders Rate/Crowdsourced', late_rate1, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Total', ETA_usage_rate, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Hired', ETA_usage_rate0, episode + 1)
                self.writter.add_scalar('ETA Usage Rate/Crowdsourced', ETA_usage_rate1, episode + 1)

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
                    algo3_eval_episode_rewards_sum,
                    
                    algo1_distance0,
                    algo1_distance1,
                    algo1_distance,
                    
                    algo2_distance0,
                    algo2_distance1,
                    algo2_distance,
                    
                    algo3_distance0,
                    algo3_distance1,
                    algo3_distance,
                    
                    algo1_avg0_speed,
                    algo1_avg1_speed,
                    algo1_avg_speed,
                    
                    algo2_avg0_speed,
                    algo2_avg1_speed,
                    algo2_avg_speed,
                    
                    algo3_avg0_speed,
                    algo3_avg1_speed,
                    algo3_avg_speed,
                    
                    algo1_overspeed0,
                    algo1_overspeed1,
                    algo1_overspeed,
                    
                    algo2_overspeed0,
                    algo2_overspeed1,
                    algo2_overspeed,
                    
                    algo3_overspeed0,
                    algo3_overspeed1,
                    algo3_overspeed,
                    
                    algo1_reject_rate_per_episode,
                    algo2_reject_rate_per_episode,
                    algo3_reject_rate_per_episode,
                    
                    algo1_reject0,
                    algo1_reject1,
                    algo1_reject,
                    
                    algo2_reject0,
                    algo2_reject1,
                    algo2_reject,
                    
                    algo3_reject0,
                    algo3_reject1,
                    algo3_reject,
                    
                    algo1_price_per_order0,
                    algo1_price_per_order1,
                    algo1_price_per_order,
                    
                    algo2_price_per_order0,
                    algo2_price_per_order1,
                    algo2_price_per_order,
                    
                    algo3_price_per_order0,
                    algo3_price_per_order1,
                    algo3_price_per_order,
                    
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
                    
                    algo1_finish0,
                    algo1_finish1,
                    algo1_finish,
                    
                    algo2_finish0,
                    algo2_finish1,
                    algo2_finish,
                    
                    algo3_finish0,
                    algo3_finish1,
                    algo3_finish,
                    
                    algo1_avg0_leisure,
                    algo1_avg1_leisure,
                    algo1_avg_leisure,

                    algo2_avg0_leisure,
                    algo2_avg1_leisure,
                    algo2_avg_leisure,
                    
                    algo3_avg0_leisure,
                    algo3_avg1_leisure,
                    algo3_avg_leisure,
                    
                    algo1_late_rate0,
                    algo1_late_rate1,
                    algo1_late_rate,
                    
                    algo2_late_rate0,
                    algo2_late_rate1,
                    algo2_late_rate,
                    
                    algo3_late_rate0,
                    algo3_late_rate1,
                    algo3_late_rate,
                    
                    algo1_ETA_usage_rate0,
                    algo1_ETA_usage_rate1,
                    algo1_ETA_usage_rate,
                    
                    algo2_ETA_usage_rate0,
                    algo2_ETA_usage_rate1,
                    algo2_ETA_usage_rate,
                    
                    algo3_ETA_usage_rate0,
                    algo3_ETA_usage_rate1,
                    algo3_ETA_usage_rate,
                    
                ) = self.eval(total_num_steps)

                algo1_eval_distance.append([algo1_distance0, algo1_distance1, algo1_distance])
                algo2_eval_distance.append([algo2_distance0, algo2_distance1, algo2_distance])
                algo3_eval_distance.append([algo3_distance0, algo3_distance1, algo3_distance])

                algo1_eval_episode_rewards.append(algo1_eval_episode_rewards_sum)
                algo2_eval_episode_rewards.append(algo2_eval_episode_rewards_sum)
                algo3_eval_episode_rewards.append(algo3_eval_episode_rewards_sum)
                
                algo1_eval_speed.append([algo1_avg0_speed, algo1_avg1_speed, algo1_avg_speed])
                algo2_eval_speed.append([algo2_avg0_speed, algo2_avg1_speed, algo2_avg_speed])
                algo3_eval_speed.append([algo3_avg0_speed, algo3_avg1_speed, algo3_avg_speed])

                algo1_eval_overspeed_rate.append([algo1_overspeed0, algo1_overspeed1, algo1_overspeed])
                algo2_eval_overspeed_rate.append([algo2_overspeed0, algo2_overspeed1, algo2_overspeed])
                algo3_eval_overspeed_rate.append([algo3_overspeed0, algo3_overspeed1, algo3_overspeed])
                
                algo1_eval_reject_rate.append(algo1_reject_rate_per_episode)
                algo2_eval_reject_rate.append(algo2_reject_rate_per_episode)
                algo3_eval_reject_rate.append(algo3_reject_rate_per_episode)

                algo1_eval_reject.append([algo1_reject0, algo1_reject1, algo1_reject])
                algo2_eval_reject.append([algo2_reject0, algo2_reject1, algo2_reject])
                algo3_eval_reject.append([algo3_reject0, algo3_reject1, algo3_reject])

                algo1_eval_order_price.append([algo1_price_per_order0, algo1_price_per_order1, algo1_price_per_order])
                algo2_eval_order_price.append([algo2_price_per_order0, algo2_price_per_order1, algo2_price_per_order])
                algo3_eval_order_price.append([algo3_price_per_order0, algo3_price_per_order1, algo3_price_per_order])

                algo1_eval_income.append([algo1_income0, algo1_income1, algo1_income, platform_cost1])
                algo2_eval_income.append([algo2_income0, algo2_income1, algo2_income, platform_cost2])
                algo3_eval_income.append([algo3_income0, algo3_income1, algo3_income, platform_cost3])

                algo1_eval_finish.append([algo1_finish0, algo1_finish1, algo1_finish])
                algo2_eval_finish.append([algo2_finish0, algo2_finish1, algo2_finish])
                algo3_eval_finish.append([algo3_finish0, algo3_finish1, algo3_finish])

                algo1_eval_leisure.append([algo1_avg0_leisure, algo1_avg1_leisure, algo1_avg_leisure])
                algo2_eval_leisure.append([algo2_avg0_leisure, algo2_avg1_leisure, algo2_avg_leisure])
                algo3_eval_leisure.append([algo3_avg0_leisure, algo3_avg1_leisure, algo3_avg_leisure])

                algo1_rate_of_late_order.append([algo1_late_rate0, algo1_late_rate1, algo1_late_rate])
                algo2_rate_of_late_order.append([algo2_late_rate0, algo2_late_rate1, algo2_late_rate])
                algo3_rate_of_late_order.append([algo3_late_rate0, algo3_late_rate1, algo3_late_rate])

                algo1_rate_of_ETA_usage.append([algo1_ETA_usage_rate0, algo1_ETA_usage_rate1, algo1_ETA_usage_rate])
                algo2_rate_of_ETA_usage.append([algo2_ETA_usage_rate0, algo2_ETA_usage_rate1, algo2_ETA_usage_rate])
                algo3_rate_of_ETA_usage.append([algo3_ETA_usage_rate0, algo3_ETA_usage_rate1, algo3_ETA_usage_rate])

        self.writter.close()
        

        # draw the Train graph
        Hired_distances = [d[0] for d in distance_total]
        Crowdsourced_distances = [d[1] for d in distance_total]
        courier_distances = [d[2] for d in distance_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_distances, label="Hired", color='blue')
        plt.plot(Crowdsourced_distances, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_distances, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Total Distances')
        plt.title('Train: Distance over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_Distance.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episode_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Rewards')
        plt.title('Train: Reward over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_reward_curve.png')
        
        Hired_speed = [s[0] for s in avg_speed_total]
        Crowdsourced_speed = [s[1] for s in avg_speed_total]
        courier_speed = [s[2] for s in avg_speed_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_speed, label="Hired", color='blue')
        plt.plot(Crowdsourced_speed, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_speed, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Speed')
        plt.title('Train: average speed over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_speed.png')
        
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
        
        plt.figure(figsize=(12, 8))
        plt.plot(reject_rate)
        plt.xlabel('Episodes')
        plt.ylabel('Order Reject Rate')
        plt.title('Train: order reject rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_order_reject_rate.png')
        
        avg_reject0 = [r[0] for r in courier_reject_num_total]
        avg_reject1 = [r[1] for r in courier_reject_num_total]
        avg_reject = [r[2] for r in courier_reject_num_total]
        plt.figure(figsize=(12, 8))
        plt.plot(avg_reject0, label="Hired", color='blue')
        plt.plot(avg_reject1, label="Crowdsourced Courier", color='orange')
        plt.plot(avg_reject, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rejection Number')
        plt.title('Train: average rejection number')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_rejection_num.png')
        
        price0 = [p[0] for p in order_price_total]
        price1 = [p[1] for p in order_price_total]
        price = [p[2] for p in order_price_total]
        plt.figure(figsize=(12, 8))
        plt.plot(price0, label="Hired", color='blue')
        plt.plot(price1, label="Crowdsourced Courier", color='orange')
        plt.plot(price, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Price of Order')
        plt.title('Train: average price of order')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_price_of_order.png')
        
        Hired_income = [i[0] for i in income_total]
        Crowdsourced_income = [i[1] for i in income_total] 
        courier_income = [i[2] for i in income_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_income, label="Hired", color='blue')
        plt.plot(Crowdsourced_income, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_income, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Income per Courier')
        plt.title('Train: average income per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_income_per_courier.png')
        
        platform_total_cost = [i[3] for i in income_total]
        plt.figure(figsize=(12, 8))
        plt.plot(platform_total_cost, label="Platform total cost", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Platform Total Cost')
        plt.title('Train: Platform Total Cost')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_platform_total_cost.png')
        
        Hired_finish = [f[0] for f in courier_finish_num_total]
        Crowdsourced_finish = [f[1] for f in courier_finish_num_total]
        courier_finish = [f[2] for f in courier_finish_num_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_finish, label="Hired", color='blue')
        plt.plot(Crowdsourced_finish, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_finish, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Finish per Courier')
        plt.title('Train: average finish per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_finish_per_courier.png')
        
        Hired_leisure = [f[0] for f in leisure_time_total]
        Crowdsourced_leisure = [f[1] for f in leisure_time_total]
        courier_leisure = [f[2] for f in leisure_time_total]
        plt.figure(figsize=(12, 8))
        plt.plot(Hired_leisure, label="Hired", color='blue')
        plt.plot(Crowdsourced_leisure, label="Crowdsourced Courier", color='orange')
        plt.plot(courier_leisure, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Average Leisure Time per Courier')
        plt.title('Train: average leisure time per courier')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_avg_leisure_time_per_courier.png')
        
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
        
        order0_ETA = [e[0] for e in rate_of_ETA_usage]
        order1_ETA = [e[1] for e in rate_of_ETA_usage]
        order_ETA = [e[2] for e in rate_of_ETA_usage]
        plt.figure(figsize=(12, 8))
        plt.plot(order0_ETA, label="Hired", color='blue')
        plt.plot(order1_ETA, label="Crowdsourced Courier", color='orange')
        plt.plot(order_ETA, label="Courier", color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Rate of ETA Usage')
        plt.title('Train: rate of ETA usage over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Train_rate_of_ETA_usage.png')
        
        #--------------------------
        # draw the Evaluation graph
        
        episodes = list(range(1, len(algo1_eval_distance) + 1))
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_distance], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_distance], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_distance], label='Algo1 Total Distance', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_distance], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_distance], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_distance], label='Algo2 Total Distance', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_distance], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_distance], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_distance], label='Algo3 Total Distance', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Distances')
        plt.title('Eval: Distance Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Distance.png')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, algo1_eval_episode_rewards, label='Algo1 Reward', color='blue', marker='o')
        plt.plot(episodes, algo2_eval_episode_rewards, label='Algo2 Reward', color='green', marker='o')
        plt.plot(episodes, algo3_eval_episode_rewards, label='Algo3 Reward', color='yellow', marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Eval: Reward over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Reward.png')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_speed], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_speed], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_speed], label='Algo1 average speed', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_speed], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_speed], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_speed], label='Algo2 average speed', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_speed], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_speed], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_speed], label='Algo3 average speed', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Average Speed')
        plt.title('Eval: Average Speed Comparison')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Average_Speed.png')
        
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
        plt.xlabel('Episodes')
        plt.ylabel('Overspeed Rate')
        plt.title('Eval: Overspeed Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Overspeed_Rate.png')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, algo1_eval_reject_rate, label='Algo1', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, algo2_eval_reject_rate, label='Algo2', color='green', linestyle='--', marker='o')
        plt.plot(episodes, algo3_eval_reject_rate, label='Algo3', color='yellow', linestyle='--', marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Reject Rate')
        plt.title('Eval: Reject Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Reject_Rate.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_reject], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_reject], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_reject], label='Algo1 Courier Reject Number', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_reject], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_reject], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_reject], label='Algo2 Courier Reject Number', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_reject], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_reject], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_reject], label='Algo3 Courier Reject Number', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Courier Reject Number')
        plt.title('Eval: Courier Reject Number over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Courier_Reject_Number.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_order_price], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_order_price], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_order_price], label='Algo1 Average Order Price', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_order_price], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_order_price], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_order_price], label='Algo2 Average Order Price', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_order_price], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_order_price], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[3] for x in algo3_eval_order_price], label='Algo3 Average Order Price', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Average Order Price')
        plt.title('Eval: Average Order Price over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_order_price.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_income], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_income], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_income], label='Algo1 Courier Average Income', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_income], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_income], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_income], label='Algo2 Courier Average Income', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_income], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_income], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_income], label='Algo3 Courier Average Income', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Courier Average Income')
        plt.title('Eval: Courier Average Income over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_courier_avg_income.png')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[3] for x in algo1_eval_income], label='Algo1', color='blue', linestyle='-', marker='o')
        plt.plot(episodes, [x[3] for x in algo2_eval_income], label='Algo2', color='green', linestyle='-', marker='o')
        plt.plot(episodes, [x[3] for x in algo3_eval_income], label='Algo3', color='yellow', linestyle='-', marker='o')
        plt.xlabel('Episodes')
        plt.ylabel('Platform Total Cost')
        plt.title('Eval: Platform Total Cost over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_platform_total_cost.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_finish], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_finish], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_finish], label='Algo1 Average Finished Orders per Courier', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_finish], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_finish], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_finish], label='Algo2 Average Finished Orders per Courier', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_finish], label='Algoo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_finish], label='Algoo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_finish], label='Algoo3 Average Finished Orders per Courier', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Average Finished Orders per Courier')
        plt.title('Eval: Average Finished Orders per Courier over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_order_finished.png')

        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_eval_leisure], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_eval_leisure], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_eval_leisure], label='Algo1 Average Leisure Time', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_eval_leisure], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_eval_leisure], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_eval_leisure], label='Algo2 Average Leisure Time', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_eval_leisure], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_eval_leisure], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_eval_leisure], label='Algo3 Average Leisure Time', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('Average Leisure Time')
        plt.title('Eval: Average Leisure Time over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_leisure_time.png')

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
        plt.xlabel('Episodes')
        plt.ylabel('Late Order Rate')
        plt.title('Eval: Late Order Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_Late_Order_Rate.png')
        
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, [x[0] for x in algo1_rate_of_ETA_usage], label='Algo1 Hired', color='blue', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo1_rate_of_ETA_usage], label='Algo1 Crowdsourced', color='blue', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo1_rate_of_ETA_usage], label='Algo1 Late Order Rate', color='blue', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo2_rate_of_ETA_usage], label='Algo2 Hired', color='green', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo2_rate_of_ETA_usage], label='Algo2 Crowdsourced', color='green', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo2_rate_of_ETA_usage], label='Algo2 Late Order Rate', color='green', linestyle='-', marker='^')
        plt.plot(episodes, [x[0] for x in algo3_rate_of_ETA_usage], label='Algo3 Hired', color='yellow', linestyle='--', marker='o')
        plt.plot(episodes, [x[1] for x in algo3_rate_of_ETA_usage], label='Algo3 Crowdsourced', color='yellow', linestyle='-.', marker='s')
        plt.plot(episodes, [x[2] for x in algo3_rate_of_ETA_usage], label='Algo3 Late Order Rate', color='yellow', linestyle='-', marker='^')
        plt.xlabel('Episodes')
        plt.ylabel('ETA Usage Rate')
        plt.title('Eval: ETA Usage Rate over Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('Eval_ETA_Usage_Rate.png')

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
        
        algo1_Hired_num = 0
        algo1_Crowdsourced_num = 0
        algo1_Crowdsourced_on = 0
        algo2_Hired_num = 0
        algo2_Crowdsourced_num = 0
        algo2_Crowdsourced_on = 0
        algo3_Hired_num = 0
        algo3_Crowdsourced_num = 0
        algo3_Crowdsourced_on = 0
        
        algo1_eval_episode_rewards_sum = 0
        algo2_eval_episode_rewards_sum = 0
        algo3_eval_episode_rewards_sum = 0        
        
        algo1_Hired_distance_per_episode = 0
        algo1_Crowdsourced_distance_per_episode = 0
        algo2_Hired_distance_per_episode = 0
        algo2_Crowdsourced_distance_per_episode = 0
        algo3_Hired_distance_per_episode = 0
        algo3_Crowdsourced_distance_per_episode = 0

        algo1_count_overspeed0 = 0
        algo1_count_overspeed1 = 0
        algo2_count_overspeed0 = 0
        algo2_count_overspeed1 = 0
        algo3_count_overspeed0 = 0
        algo3_count_overspeed1 = 0
        
        algo1_num_active_couriers0 = 0
        algo1_num_active_couriers1 = 0
        algo2_num_active_couriers0 = 0
        algo2_num_active_couriers1 = 0
        algo3_num_active_couriers0 = 0
        algo3_num_active_couriers1 = 0
        
        algo1_count_reject_orders = 0
        algo1_max_reject_num = 0
        algo2_count_reject_orders = 0
        algo2_max_reject_num = 0
        algo3_count_reject_orders = 0
        algo3_max_reject_num = 0

        algo1_late_orders0 = 0
        algo1_late_orders1 = 0
        algo2_late_orders0 = 0
        algo2_late_orders1 = 0
        algo3_late_orders0 = 0
        algo3_late_orders1 = 0
        
        algo1_ETA_usage0 = 0
        algo1_ETA_usage1 = 0
        algo2_ETA_usage0 = 0
        algo2_ETA_usage1 = 0
        algo3_ETA_usage0 = 0
        algo3_ETA_usage1 = 0
        
        algo1_count_dropped_orders0 = 0
        algo1_count_dropped_orders1 = 0
        algo2_count_dropped_orders0 = 0
        algo2_count_dropped_orders1 = 0
        algo3_count_dropped_orders0 = 0
        algo3_count_dropped_orders1 = 0
        
        algo1_order0_price = 0
        algo1_order1_price = 0
        algo1_order0_num = 0
        algo1_order1_num = 0
        algo1_order_wait = 0
        algo2_order0_price = 0
        algo2_order1_price = 0
        algo2_order0_num = 0
        algo2_order1_num = 0
        algo2_order_wait = 0
        algo3_order0_price = 0
        algo3_order1_price = 0
        algo3_order0_num = 0
        algo3_order1_num = 0
        algo3_order_wait = 0
        
        algo1_Hired_reject_num = 0
        algo1_Crowdsourced_reject_num = 0
        algo2_Hired_reject_num = 0
        algo2_Crowdsourced_reject_num = 0
        algo3_Hired_reject_num = 0
        algo3_Crowdsourced_reject_num = 0
        
        algo1_Hired_finish_num = 0
        algo1_Crowdsourced_finish_num = 0
        algo2_Hired_finish_num = 0
        algo2_Crowdsourced_finish_num = 0
        algo3_Hired_finish_num = 0
        algo3_Crowdsourced_finish_num = 0
        
        algo1_Hired_leisure_time = 0
        algo1_Crowdsourced_leisure_time = 0
        algo2_Hired_leisure_time = 0
        algo2_Crowdsourced_leisure_time = 0
        algo3_Hired_leisure_time = 0
        algo3_Crowdsourced_leisure_time = 0
        
        algo1_Hired_avg_speed = 0
        algo1_Crowdsourced_avg_speed = 0
        algo2_Hired_avg_speed = 0
        algo2_Crowdsourced_avg_speed = 0
        algo3_Hired_avg_speed = 0
        algo3_Crowdsourced_avg_speed = 0
        
        algo1_Hired_income = 0
        algo1_Crowdsourced_income = 0
        algo2_Hired_income = 0
        algo2_Crowdsourced_income = 0
        algo3_Hired_income = 0
        algo3_Crowdsourced_income = 0
        
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
                else:
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
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo1_Hired_num += 1
                        algo1_Hired_distance_per_episode += c.travel_distance
                        algo1_Hired_reject_num += c.reject_order_num
                        algo1_Hired_finish_num += c.finish_order_num
                        algo1_Hired_leisure_time += c.total_leisure_time
                        algo1_Hired_avg_speed += c.avg_speed
                        algo1_Hired_income += c.income
                    else:
                        algo1_Crowdsourced_num += 1
                        algo1_Crowdsourced_distance_per_episode += c.travel_distance
                        algo1_Crowdsourced_reject_num += c.reject_order_num
                        algo1_Crowdsourced_finish_num += c.finish_order_num
                        algo1_Crowdsourced_leisure_time += c.total_leisure_time
                        algo1_Crowdsourced_avg_speed += c.avg_speed
                        algo1_Crowdsourced_income += c.income
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo1_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo1_late_orders0 += 1
                            else:
                                algo1_ETA_usage0 += o.ETA_usage
                        else:
                            algo1_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo1_late_orders1 += 1
                            else:
                                algo1_ETA_usage1 += o.ETA_usage
                            
                    if o.reject_count > 0:
                        algo1_count_reject_orders += 1
                        if algo1_max_reject_num <= o.reject_count:
                            algo1_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo1_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo1_order0_price += o.price
                            algo1_order0_num += 1
                        else:
                            algo1_order1_price += o.price
                            algo1_order1_num += 1             
                    
            elif i == 1:
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo2_Hired_num += 1
                        algo2_Hired_distance_per_episode += c.travel_distance
                        algo2_Hired_reject_num += c.reject_order_num
                        algo2_Hired_finish_num += c.finish_order_num
                        algo2_Hired_leisure_time += c.total_leisure_time
                        algo2_Hired_avg_speed += c.avg_speed
                        algo2_Hired_income += c.income
                    else:
                        algo2_Crowdsourced_num += 1
                        algo2_Crowdsourced_distance_per_episode += c.travel_distance
                        algo2_Crowdsourced_reject_num += c.reject_order_num
                        algo2_Crowdsourced_finish_num += c.finish_order_num
                        algo2_Crowdsourced_leisure_time += c.total_leisure_time
                        algo2_Crowdsourced_avg_speed += c.avg_speed
                        algo2_Crowdsourced_income += c.income
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo2_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo2_late_orders0 += 1
                            else:
                                algo2_ETA_usage0 += o.ETA_usage
                        else:
                            algo2_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo2_late_orders1 += 1
                            else:
                                algo2_ETA_usage1 += o.ETA_usage
                            
                    if o.reject_count > 0:
                        algo2_count_reject_orders += 1
                        if algo2_max_reject_num <= o.reject_count:
                            algo2_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo2_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo2_order0_price += o.price
                            algo2_order0_num += 1
                        else:
                            algo2_order1_price += o.price
                            algo2_order1_num += 1
            else:
                for c in self.eval_envs.envs_discrete[i].couriers:
                    if c.courier_type == 0:
                        algo3_Hired_num += 1
                        algo3_Hired_distance_per_episode += c.travel_distance
                        algo3_Hired_reject_num += c.reject_order_num
                        algo3_Hired_finish_num += c.finish_order_num
                        algo3_Hired_leisure_time += c.total_leisure_time
                        algo3_Hired_avg_speed += c.avg_speed
                        algo3_Hired_income += c.income
                    else:
                        algo3_Crowdsourced_num += 1
                        algo3_Crowdsourced_distance_per_episode += c.travel_distance
                        algo3_Crowdsourced_reject_num += c.reject_order_num
                        algo3_Crowdsourced_finish_num += c.finish_order_num
                        algo3_Crowdsourced_leisure_time += c.total_leisure_time
                        algo3_Crowdsourced_avg_speed += c.avg_speed
                        algo3_Crowdsourced_income += c.income
                
                for o in self.eval_envs.envs_discrete[i].orders:
                    if o.status == 'dropped':
                        if o.pair_courier.courier_type == 0:
                            algo3_count_dropped_orders0 += 1
                            if o.is_late == 1:
                                algo3_late_orders0 += 1
                            else:
                                algo3_ETA_usage0 += o.ETA_usage
                        else:
                            algo3_count_dropped_orders1 += 1
                            if o.is_late == 1:
                                algo3_late_orders1 += 1
                            else:
                                algo3_ETA_usage1 += o.ETA_usage
                            
                    if o.reject_count > 0:
                        algo3_count_reject_orders += 1
                        if algo3_max_reject_num <= o.reject_count:
                            algo3_max_reject_num = o.reject_count
                    
                    if o.status == 'wait_pair':
                        algo3_order_wait += 1
                    else:
                        if o.pair_courier.courier_type == 0:
                            algo3_order0_price += o.price
                            algo3_order0_num += 1
                        else:
                            algo3_order1_price += o.price
                            algo3_order1_num += 1     
                            
        print(f"\nIn Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} on, {algo1_order0_num} Order0, {algo1_order1_num} Order1, {algo1_order_wait} Orders waiting to be paired")
        print(f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} on, {algo2_order0_num} Order0, {algo2_order1_num} Order1, {algo2_order_wait} Orders waiting to be paired")  
        print(f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} on, {algo3_order0_num} Order0, {algo3_order1_num} Order1, {algo3_order_wait} Orders waiting to be paired")       
     
        
        print(f"Total Reward for Evaluation Between Two Algos:\nAlgo1: {algo1_eval_episode_rewards_sum}\nAlgo2: {algo2_eval_episode_rewards_sum}\nAlgo3: {algo3_eval_episode_rewards_sum}")
        self.writter.add_scalar('Eval Reward/Algo1', algo1_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo2', algo2_eval_episode_rewards_sum, self.eval_num)
        self.writter.add_scalar('Eval Reward/Algo3', algo3_eval_episode_rewards_sum, self.eval_num)

        algo1_distance0 = round(algo1_Hired_distance_per_episode / algo1_Hired_num, 2)
        algo1_distance1 = round(algo1_Crowdsourced_distance_per_episode / algo1_Crowdsourced_num, 2)
        algo1_distance = (algo1_Hired_distance_per_episode + algo1_Crowdsourced_distance_per_episode)/ (algo1_Hired_num + algo1_Crowdsourced_num)
        algo1_distance = round(algo1_distance, 2)
        algo2_distance0 = round(algo2_Hired_distance_per_episode / algo2_Hired_num, 2)
        algo2_distance1 = round(algo2_Crowdsourced_distance_per_episode / algo2_Crowdsourced_num, 2)
        algo2_distance = (algo2_Hired_distance_per_episode + algo2_Crowdsourced_distance_per_episode)/ (algo2_Hired_num + algo2_Crowdsourced_num)
        algo2_distance = round(algo2_distance, 2)
        algo3_distance0 = round(algo3_Hired_distance_per_episode / algo3_Hired_num, 2)
        algo3_distance1 = round(algo3_Crowdsourced_distance_per_episode / algo3_Crowdsourced_num, 2)
        algo3_distance = (algo3_Hired_distance_per_episode + algo3_Crowdsourced_distance_per_episode)/ (algo3_Hired_num + algo3_Crowdsourced_num)
        algo3_distance = round(algo3_distance, 2)
        print("Average Travel Distance per Courier Between Two Algos:")
        print(f"Algo1: Hired - {algo1_distance0} meters, Crowdsourced - {algo1_distance1} meters, Total - {algo1_distance} meters")
        print(f"Algo2: Hired - {algo2_distance0} meters, Crowdsourced - {algo2_distance1} meters, Total - {algo2_distance} meters")
        print(f"Algo3: Hired - {algo3_distance0} meters, Crowdsourced - {algo3_distance1} meters, Total - {algo3_distance} meters")
        self.writter.add_scalar('Eval Travel Distance/Algo1 Hired', algo1_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Crowdsourced', algo1_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo1 Total', algo1_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Hired', algo2_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Crowdsourced', algo2_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo2 Total', algo2_distance, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Hired', algo3_distance0, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Crowdsourced', algo3_distance1, self.eval_num)
        self.writter.add_scalar('Eval Travel Distance/Algo3 Total', algo3_distance, self.eval_num)

        algo1_avg0_speed = round(algo1_Hired_avg_speed / algo1_Hired_num, 2)
        algo1_avg1_speed = round(algo1_Crowdsourced_avg_speed / algo1_Crowdsourced_num, 2)
        algo1_avg_speed = (algo1_Hired_avg_speed + algo1_Crowdsourced_avg_speed) / (algo1_Hired_num + algo1_Crowdsourced_num)
        algo1_avg_speed = round(algo1_avg_speed, 2)
        algo2_avg0_speed = round(algo2_Hired_avg_speed / algo2_Hired_num, 2)
        algo2_avg1_speed = round(algo2_Crowdsourced_avg_speed / algo2_Crowdsourced_num, 2)
        algo2_avg_speed = (algo2_Hired_avg_speed + algo2_Crowdsourced_avg_speed) / (algo2_Hired_num + algo2_Crowdsourced_num)
        algo2_avg_speed = round(algo2_avg_speed, 2)
        algo3_avg0_speed = round(algo3_Hired_avg_speed / algo3_Hired_num, 2)
        algo3_avg1_speed = round(algo3_Crowdsourced_avg_speed / algo3_Crowdsourced_num, 2)
        algo3_avg_speed = (algo3_Hired_avg_speed + algo3_Crowdsourced_avg_speed) / (algo3_Hired_num + algo3_Crowdsourced_num)
        algo3_avg_speed = round(algo3_avg_speed, 2)
        print("Average Speed per Courier Between Two Algos:")
        print(f"Algo1: Hired average speed is {algo1_avg0_speed} m/s, Crowdsourced average speed is {algo1_avg1_speed} m/s and average speed per courier is {algo1_avg_speed} m/s")
        print(f"Algo2: Hired average speed is {algo2_avg0_speed} m/s, Crowdsourced average speed is {algo2_avg1_speed} m/s and average speed per courier is {algo2_avg_speed} m/s")
        print(f"Algo3: Hired average speed is {algo3_avg0_speed} m/s, Crowdsourced average speed is {algo3_avg1_speed} m/s and average speed per courier is {algo3_avg_speed} m/s")
        self.writter.add_scalar('Eval Average Speed/Algo1 Total', algo1_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Hired', algo1_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo1 Crowdsourced', algo1_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Total', algo2_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Hired', algo2_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo2 Crowdsourced', algo2_avg1_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Total', algo3_avg_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Hired', algo3_avg0_speed, self.eval_num)
        self.writter.add_scalar('Eval Average Speed/Algo3 Crowdsourced', algo3_avg1_speed, self.eval_num)

        algo1_overspeed0 = round(algo1_count_overspeed0 / algo1_num_active_couriers0, 2)
        algo1_overspeed1 = round(algo1_count_overspeed1 / algo1_num_active_couriers1, 2)
        algo1_overspeed = round((algo1_count_overspeed0 + algo1_count_overspeed1) / (algo1_num_active_couriers0 + algo1_num_active_couriers1), 2)
        algo2_overspeed0 = round(algo2_count_overspeed0 / algo2_num_active_couriers0, 2)
        algo2_overspeed1 = round(algo2_count_overspeed1 / algo2_num_active_couriers1, 2)
        algo2_overspeed = round((algo2_count_overspeed0 + algo2_count_overspeed1) / (algo2_num_active_couriers0 + algo2_num_active_couriers1), 2)
        algo3_overspeed0 = round(algo3_count_overspeed0 / algo3_num_active_couriers0, 2)
        algo3_overspeed1 = round(algo3_count_overspeed1 / algo3_num_active_couriers1, 2)
        algo3_overspeed = round((algo3_count_overspeed0 + algo3_count_overspeed1) / (algo3_num_active_couriers0 + algo3_num_active_couriers1), 2)
        print("Rate of Overspeed for Evaluation Between Two Algos:")
        print(f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}")
        print(f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}")
        print(f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}")
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Total', algo1_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Hired', algo1_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo1 Crowdsourced', algo1_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Total', algo2_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Hired', algo2_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo2 Crowdsourced', algo2_overspeed1, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Total', algo3_overspeed, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Hired', algo3_overspeed0, self.eval_num)
        self.writter.add_scalar('Eval Overspeed Rate/Algo3 Crowdsourced', algo3_overspeed1, self.eval_num)

        algo1_reject_rate_per_episode = round(algo1_count_reject_orders / len(self.eval_envs.envs_discrete[0].orders), 2)
        algo2_reject_rate_per_episode = round(algo2_count_reject_orders / len(self.eval_envs.envs_discrete[1].orders), 2)
        algo3_reject_rate_per_episode = round(algo3_count_reject_orders / len(self.eval_envs.envs_discrete[2].orders), 2)
        print("Reject Rate for Evaluation Between Two Algos:")
        print(f"Algo1: {algo1_reject_rate_per_episode} and the order is rejected by {algo1_max_reject_num} times at most")
        print(f"Algo2: {algo2_reject_rate_per_episode} and the order is rejected by {algo2_max_reject_num} times at most")
        print(f"Algo3: {algo3_reject_rate_per_episode} and the order is rejected by {algo3_max_reject_num} times at most")
        self.writter.add_scalar('Eval Reject rate/Algo1', algo1_reject_rate_per_episode, self.eval_num)
        self.writter.add_scalar('Eval Reject rate/Algo2', algo2_reject_rate_per_episode, self.eval_num)
        self.writter.add_scalar('Eval Reject rate/Algo3', algo3_reject_rate_per_episode, self.eval_num)

        algo1_reject0 = round(algo1_Hired_reject_num / algo1_Hired_num, 2)
        algo1_reject1 = round(algo1_Crowdsourced_reject_num / algo1_Crowdsourced_num, 2)
        algo1_reject = (algo1_Crowdsourced_reject_num + algo1_Crowdsourced_reject_num) / (algo1_Hired_num + algo1_Crowdsourced_num)
        algo1_reject = round(algo1_reject, 2)
        algo2_reject0 = round(algo2_Hired_reject_num / algo2_Hired_num, 2)
        algo2_reject1 = round(algo2_Crowdsourced_reject_num / algo2_Crowdsourced_num, 2)
        algo2_reject = (algo2_Crowdsourced_reject_num + algo2_Crowdsourced_reject_num) / (algo2_Hired_num + algo2_Crowdsourced_num)
        algo2_reject = round(algo2_reject, 2)
        algo3_reject0 = round(algo3_Hired_reject_num / algo3_Hired_num, 2)
        algo3_reject1 = round(algo3_Crowdsourced_reject_num / algo3_Crowdsourced_num, 2)
        algo3_reject = (algo3_Crowdsourced_reject_num + algo3_Crowdsourced_reject_num) / (algo3_Hired_num + algo3_Crowdsourced_num)
        algo3_reject = round(algo3_reject, 2)
        print("Average Reject Numbers per Courier for Evaluation Between Two Algos:")
        print(f"Algo1: Hired rejects average {algo1_reject0} orders, Crowdsourced rejects average {algo1_reject1} orders and Total reject number per courier is {algo1_reject}")
        print(f"Algo2: Hired rejects average {algo2_reject0} orders, Crowdsourced rejects average {algo2_reject1} orders and Total reject number per courier is {algo2_reject}")
        print(f"Algo3: Hired rejects average {algo3_reject0} orders, Crowdsourced rejects average {algo3_reject1} orders and Total reject number per courier is {algo3_reject}")
        self.writter.add_scalar('Eval Average Rejection/Algo1 Total', algo1_reject, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo1 Hired', algo1_reject0, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo1 Crowdsourced', algo1_reject1, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo2 Total', algo2_reject, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo2 Hired', algo2_reject0, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo2 Crowdsourced', algo2_reject1, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo3 Total', algo3_reject, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo3 Hired', algo3_reject0, self.eval_num)
        self.writter.add_scalar('Eval Average Rejection/Algo3 Crowdsourced', algo3_reject1, self.eval_num)

        algo1_price_per_order0 = round(algo1_order0_price / algo1_order0_num, 2)
        algo1_price_per_order1 = round(algo1_order1_price / algo1_order1_num, 2)
        algo1_price_per_order = round((algo1_order0_price + algo1_order1_price) / (algo1_order0_num + algo1_order1_num), 2)
        algo2_price_per_order0 = round(algo2_order0_price / algo2_order0_num, 2)
        algo2_price_per_order1 = round(algo2_order1_price / algo2_order1_num, 2)
        algo2_price_per_order = round((algo2_order0_price + algo2_order1_price) / (algo2_order0_num + algo2_order1_num), 2)
        algo3_price_per_order0 = round(algo3_order0_price / algo3_order0_num, 2)
        algo3_price_per_order1 = round(algo3_order1_price / algo3_order1_num, 2)
        algo3_price_per_order = round((algo3_order0_price + algo3_order1_price) / (algo3_order0_num + algo3_order1_num), 2)
        print("Average Price per order for Evaluation Between Two Algos:")
        print(f"Algo1: The average price of Hired's order is {algo1_price_per_order0} yuan with {algo1_order0_num} orders, Crowdsourced's is {algo1_price_per_order1} yuan with {algo1_order1_num} orders and for all is {algo1_price_per_order}")
        print(f"Algo2: The average price of Hired's order is {algo2_price_per_order0} yuan with {algo2_order0_num} orders, Crowdsourced's is {algo2_price_per_order1} yuan with {algo2_order1_num} orders and for all is {algo2_price_per_order}")
        print(f"Algo3: The average price of Hired's order is {algo3_price_per_order0} yuan with {algo3_order0_num} orders, Crowdsourced's is {algo3_price_per_order1} yuan with {algo3_order1_num} orders and for all is {algo3_price_per_order}")
        self.writter.add_scalar('Eval Average Price/Algo1 Total', algo1_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Hired', algo1_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo1 Crowdsourced', algo1_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Total', algo2_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Hired', algo2_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo2 Crowdsourced', algo2_price_per_order1, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Total', algo3_price_per_order, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Hired', algo3_price_per_order0, self.eval_num)
        self.writter.add_scalar('Eval Average Price/Algo3 Crowdsourced', algo3_price_per_order1, self.eval_num)

        algo1_income0 = round(algo1_Hired_income / algo1_Hired_num, 2)
        algo1_income1 = round(algo1_Crowdsourced_income / algo1_Crowdsourced_num, 2)
        algo1_income = round((algo1_Hired_income + algo1_Crowdsourced_income) / (algo1_Hired_num + algo1_Crowdsourced_num), 2)
        platform_cost1 = algo1_Hired_income + algo1_Crowdsourced_income
        algo2_income0 = round(algo2_Hired_income / algo2_Hired_num, 2)
        algo2_income1 = round(algo2_Crowdsourced_income / algo2_Crowdsourced_num, 2)
        algo2_income = round((algo2_Hired_income + algo2_Crowdsourced_income) / (algo2_Hired_num + algo2_Crowdsourced_num), 2)
        platform_cost2 = algo2_Hired_income + algo2_Crowdsourced_income
        algo3_income0 = round(algo3_Hired_income / algo3_Hired_num, 2)
        algo3_income1 = round(algo3_Crowdsourced_income / algo3_Crowdsourced_num, 2)
        algo3_income = round((algo3_Hired_income + algo3_Crowdsourced_income) / (algo3_Hired_num + algo3_Crowdsourced_num), 2)
        platform_cost3 = algo3_Hired_income + algo3_Crowdsourced_income
        print("Average Income per Courier for Evaluation Between Two Algos:")
        print(f"Algo1: Hired's average income is {algo1_income0} yuan, Crowdsourced's average income is {algo1_income1} yuan and Total income per courier is {algo1_income}, The platform total cost is {platform_cost1} yuan")
        print(f"Algo2: Hired's average income is {algo2_income0} yuan, Crowdsourced's average income is {algo2_income1} yuan and Total income per courier is {algo2_income}, The platform total cost is {platform_cost2} yuan")
        print(f"Algo3: Hired's average income is {algo3_income0} yuan, Crowdsourced's average income is {algo3_income1} yuan and Total income per courier is {algo3_income}, The platform total cost is {platform_cost3} yuan")
        self.writter.add_scalar('Eval Average Income/Algo1 Total', algo1_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Hired', algo1_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo1 Crowdsourced', algo1_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Total', algo2_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Hired', algo2_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo2 Crowdsourced', algo2_income1, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Total', algo3_income, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Hired', algo3_income0, self.eval_num)
        self.writter.add_scalar('Eval Average Income/Algo3 Crowdsourced', algo3_income1, self.eval_num)
        self.writter.add_scalar('Eval Platform Total Cost/Algo1', platform_cost1, self.eval_num)
        self.writter.add_scalar('Eval Platform Total Cost/Algo2', platform_cost2, self.eval_num)
        self.writter.add_scalar('Eval Platform Total Cost/Algo3', platform_cost3, self.eval_num)

        algo1_finish0 = round(algo1_Hired_finish_num / algo1_Hired_num, 2)
        algo1_finish1 = round(algo1_Crowdsourced_finish_num / algo1_Crowdsourced_num, 2)
        algo1_finish = round((algo1_Hired_finish_num + algo1_Crowdsourced_finish_num) / (algo1_Hired_num + algo1_Crowdsourced_num), 2)
        algo2_finish0 = round(algo2_Hired_finish_num / algo2_Hired_num, 2)
        algo2_finish1 = round(algo2_Crowdsourced_finish_num / algo2_Crowdsourced_num, 2)
        algo2_finish = round((algo2_Hired_finish_num + algo2_Crowdsourced_finish_num) / (algo2_Hired_num + algo2_Crowdsourced_num), 2)
        algo3_finish0 = round(algo3_Hired_finish_num / algo3_Hired_num, 2)
        algo3_finish1 = round(algo3_Crowdsourced_finish_num / algo3_Crowdsourced_num, 2)
        algo3_finish = round((algo3_Hired_finish_num + algo3_Crowdsourced_finish_num) / (algo3_Hired_num + algo3_Crowdsourced_num), 2)
        print("Average Order finished per courier for Evaluation Between Two Algos:")
        print(f"Algo1: Hired finishes average {algo1_finish0} orders while Crowdsourced finishes average {algo1_finish1} orders, Total finish number per courier is {algo1_finish}")
        print(f"Algo2: Hired finishes average {algo2_finish0} orders while Crowdsourced finishes average {algo2_finish1} orders, Total finish number per courier is {algo2_finish}")
        print(f"Algo3: Hired finishes average {algo3_finish0} orders while Crowdsourced finishes average {algo3_finish1} orders, Total finish number per courier is {algo3_finish}")
        self.writter.add_scalar('Eval Average Finish/Algo1 Total', algo1_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Hired', algo1_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo1 Crowdsourced', algo1_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Total', algo2_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Hired', algo2_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo2 Crowdsourced', algo2_finish1, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Total', algo3_finish, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Hired', algo3_finish0, self.eval_num)
        self.writter.add_scalar('Eval Average Finish/Algo3 Crowdsourced', algo3_finish1, self.eval_num)

        algo1_avg0_leisure = round(algo1_Hired_leisure_time / algo1_Hired_num / 60, 2)
        algo1_avg1_leisure = round(algo1_Crowdsourced_leisure_time / algo1_Crowdsourced_num / 60, 2)
        algo1_avg_leisure = round((algo1_Hired_leisure_time + algo1_Crowdsourced_leisure_time) / (algo1_Hired_num + algo1_Crowdsourced_num) / 60, 2)
        algo2_avg0_leisure = round(algo2_Hired_leisure_time / algo2_Hired_num / 60, 2)
        algo2_avg1_leisure = round(algo2_Crowdsourced_leisure_time / algo2_Crowdsourced_num / 60, 2)
        algo2_avg_leisure = round((algo2_Hired_leisure_time + algo2_Crowdsourced_leisure_time) / (algo2_Hired_num + algo2_Crowdsourced_num) / 60, 2)
        algo3_avg0_leisure = round(algo3_Hired_leisure_time / algo3_Hired_num / 60, 2)
        algo3_avg1_leisure = round(algo3_Crowdsourced_leisure_time / algo3_Crowdsourced_num / 60, 2)
        algo3_avg_leisure = round((algo3_Hired_leisure_time + algo3_Crowdsourced_leisure_time) / (algo3_Hired_num + algo3_Crowdsourced_num) / 60, 2)
        print("Average leisure time per courier for Evaluation Between Two Algos:")
        print(f"Algo1: Hired leisure time is {algo1_avg0_leisure} minutes, Crowdsourced leisure time is {algo1_avg1_leisure} minutes and Total leisure time per courier is {algo1_avg_leisure} minutes")
        print(f"Algo2: Hired leisure time is {algo2_avg0_leisure} minutes, Crowdsourced leisure time is {algo2_avg1_leisure} minutes and Total leisure time per courier is {algo2_avg_leisure} minutes")
        print(f"Algo3: Hired leisure time is {algo3_avg0_leisure} minutes, Crowdsourced leisure time is {algo3_avg1_leisure} minutes and Total leisure time per courier is {algo3_avg_leisure} minutes")
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Total', algo1_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Hired', algo1_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo1 Crowdsourced', algo1_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Total', algo2_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Hired', algo2_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo2 Crowdsourced', algo2_avg1_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Total', algo3_avg_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Hired', algo3_avg0_leisure, self.eval_num)
        self.writter.add_scalar('Eval Average Leisure Time/Algo3 Crowdsourced', algo3_avg1_leisure, self.eval_num)

        message = (
            f"In Algo1 there are {algo1_Hired_num} Hired, {algo1_Crowdsourced_num} Crowdsourced with {algo1_Crowdsourced_on} on, {algo1_order0_num} Order0, {algo1_order1_num} Order1, {algo1_order_wait} Orders waiting to be paired\n"
            f"In Algo2 there are {algo2_Hired_num} Hired, {algo2_Crowdsourced_num} Crowdsourced with {algo2_Crowdsourced_on} on, {algo2_order0_num} Order0, {algo2_order1_num} Order1, {algo2_order_wait} Orders waiting to be paired\n"
            f"In Algo3 there are {algo3_Hired_num} Hired, {algo3_Crowdsourced_num} Crowdsourced with {algo3_Crowdsourced_on} on, {algo3_order0_num} Order0, {algo3_order1_num} Order1, {algo3_order_wait} Orders waiting to be paired\n"
            f"Total Reward for Evaluation Between Two Algos:\n"
            f"Algo1: {algo1_eval_episode_rewards_sum}\n"
            f"Algo2: {algo2_eval_episode_rewards_sum}\n"
            f"Algo3: {algo3_eval_episode_rewards_sum}\n"
            f"Average Travel Distance per Courier Between Two Algos:\n"
            f"Algo1: Hired - {algo1_distance0} meters, Crowdsourced - {algo1_distance1} meters, Total - {algo1_distance} meters\n"
            f"Algo2: Hired - {algo2_distance0} meters, Crowdsourced - {algo2_distance1} meters, Total - {algo2_distance}\n"
            f"Algo3: Hired - {algo3_distance0} meters, Crowdsourced - {algo3_distance1} meters, Total - {algo3_distance} meters\n"
            "Average Speed per Courier Between Two Algos:\n"
            f"Algo1: Hired average speed is {algo1_avg0_speed} m/s, Crowdsourced average speed is {algo1_avg1_speed} m/s and average speed per courier is {algo1_avg_speed} m/s\n"
            f"Algo2: Hired average speed is {algo2_avg0_speed} m/s, Crowdsourced average speed is {algo2_avg1_speed} m/s and average speed per courier is {algo2_avg_speed} m/s\n"
            f"Algo3: Hired average speed is {algo3_avg0_speed} m/s, Crowdsourced average speed is {algo3_avg1_speed} m/s and average speed per courier is {algo3_avg_speed} m/s\n"
            "Rate of Overspeed for Evaluation Between Two Algos:\n"
            f"Algo1: Hired - {algo1_overspeed0}, Crowdsourced - {algo1_overspeed1}, Total rate - {algo1_overspeed}\n"
            f"Algo2: Hired - {algo2_overspeed0}, Crowdsourced - {algo2_overspeed1}, Total rate - {algo2_overspeed}\n"
            f"Algo3: Hired - {algo3_overspeed0}, Crowdsourced - {algo3_overspeed1}, Total rate - {algo3_overspeed}\n"
            "Reject Rate for Evaluation Between Two Algos:\n"
            f"Algo1: {algo1_reject_rate_per_episode} and the order is rejected by {algo1_max_reject_num} times at most\n"
            f"Algo2: {algo2_reject_rate_per_episode} and the order is rejected by {algo2_max_reject_num} times at most\n"
            f"Algo3: {algo3_reject_rate_per_episode} and the order is rejected by {algo3_max_reject_num} times at most\n"
            "Average Reject Numbers per Courier for Evaluation Between Two Algos:\n"
            f"Algo1: Hired rejects average {algo1_reject0} orders, Crowdsourced rejects average {algo1_reject1} orders and Total reject number per courier is {algo1_reject}\n"
            f"Algo2: Hired rejects average {algo2_reject0} orders, Crowdsourced rejects average {algo2_reject1} orders and Total reject number per courier is {algo2_reject}\n"
            f"Algo3: Hired rejects average {algo3_reject0} orders, Crowdsourced rejects average {algo3_reject1} orders and Total reject number per courier is {algo3_reject}\n"
            "Average Price per order for Evaluation Between Two Algos:\n"
            f"Algo1: The average price of Hired's order is {algo1_price_per_order0} yuan with {algo1_order0_num} orders, Crowdsourced's is {algo1_price_per_order1} yuan with {algo1_order1_num} orders and for all is {algo1_price_per_order}\n"
            f"Algo2: The average price of Hired's order is {algo2_price_per_order0} yuan with {algo2_order0_num} orders, Crowdsourced's is {algo2_price_per_order1} yuan with {algo2_order1_num} orders and for all is {algo2_price_per_order}\n"
            f"Algo3: The average price of Hired's order is {algo3_price_per_order0} yuan with {algo3_order0_num} orders, Crowdsourced's is {algo3_price_per_order1} yuan with {algo3_order1_num} orders and for all is {algo3_price_per_order}\n"
            "Average Income per Courier for Evaluation Between Two Algos:\n"
            f"Algo1: Hired's average income is {algo1_income0} yuan, Crowdsourced's average income is {algo1_income1} yuan and Total income per courier is {algo1_income}, The platform total cost is {platform_cost1} yuan\n"
            f"Algo2: Hired's average income is {algo2_income0} yuan, Crowdsourced's average income is {algo2_income1} yuan and Total income per courier is {algo2_income}, The platform total cost is {platform_cost2} yuan\n"
            f"Algo3: Hired's average income is {algo3_income0} yuan, Crowdsourced's average income is {algo3_income1} yuan and Total income per courier is {algo3_income}, The platform total cost is {platform_cost3} yuan\n"
            "Average Order finished per courier for Evaluation Between Two Algos:\n"
            f"Algo1: Hired finishes average {algo1_finish0} orders while Crowdsourced finishes average {algo1_finish1} orders, Total finish number per courier is {algo1_finish}\n"
            f"Algo2: Hired finishes average {algo2_finish0} orders while Crowdsourced finishes average {algo2_finish1} orders, Total finish number per courier is {algo2_finish}\n"
            f"Algo3: Hired finishes average {algo3_finish0} orders while Crowdsourced finishes average {algo3_finish1} orders, Total finish number per courier is {algo3_finish}\n"
            "Average leisure time per courier for Evaluation Between Two Algos:\n"
            f"Algo1: Hired leisure time is {algo1_avg0_leisure} minutes, Crowdsourced leisure time is {algo1_avg1_leisure} minutes and Total leisure time per courier is {algo1_avg_leisure} minutes\n"
            f"Algo2: Hired leisure time is {algo2_avg0_leisure} minutes, Crowdsourced leisure time is {algo2_avg1_leisure} minutes and Total leisure time per courier is {algo2_avg_leisure} minutes\n"
            f"Algo3: Hired leisure time is {algo3_avg0_leisure} minutes, Crowdsourced leisure time is {algo3_avg1_leisure} minutes and Total leisure time per courier is {algo3_avg_leisure} minutes\n"
        )
        
        if algo1_count_dropped_orders0 + algo1_count_dropped_orders1 == 0:
            print("No order is dropped in Algo1")
            algo1_late_rate = -1
            algo1_late_rate0 = -1
            algo1_late_rate1 = -1
            algo1_ETA_usage_rate = -1
            algo1_ETA_usage_rate0 = -1
            algo1_ETA_usage_rate1 = -1

            self.writter.add_scalar('Eval Late Order Rate/Algo1 Total', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Hired', algo1_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Crowdsourced', algo1_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total', algo1_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired', algo1_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced', algo1_ETA_usage_rate1, self.eval_num)
            
            message += "No order is dropped in Algo1\n"
        else:
            if algo1_count_dropped_orders0:                
                algo1_late_rate0 = round(algo1_late_orders0 / algo1_count_dropped_orders0, 2)
                algo1_ETA_usage_rate0 = round(algo1_ETA_usage0 / algo1_count_dropped_orders0, 2)
            else:
                algo1_late_rate0 = -1
                algo1_ETA_usage_rate0 = -1
                
            if algo1_count_dropped_orders1:                
                algo1_late_rate1 = round(algo1_late_orders1 / algo1_count_dropped_orders1, 2)
                algo1_ETA_usage_rate1 = round(algo1_ETA_usage1 / algo1_count_dropped_orders1, 2)
            else:
                algo1_late_rate1 = -1
                algo1_ETA_usage_rate1 = -1
                
            algo1_late_rate = round((algo1_late_orders0 + algo1_late_orders1) / (algo1_count_dropped_orders0 +algo1_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate}")

            algo1_ETA_usage_rate = round((algo1_ETA_usage0 + algo1_ETA_usage1) / (algo1_count_dropped_orders0 +algo1_count_dropped_orders1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0}, Crowdsourced - {algo1_ETA_usage_rate1}, Total - {algo1_ETA_usage_rate}")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Total', algo1_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Hired', algo1_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo1 Crowdsourced', algo1_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Total', algo1_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Hired', algo1_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo1 Crowdsourced', algo1_ETA_usage_rate1, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0}, Crowdsourced - {algo1_ETA_usage_rate1}, Total - {algo1_ETA_usage_rate}\n"

        
        if algo2_count_dropped_orders0 + algo2_count_dropped_orders1 == 0:
            print("No order is dropped in Algo2")
            algo2_late_rate = -1
            algo2_late_rate0 = -1
            algo2_late_rate1 = -1
            algo2_ETA_usage_rate = -1
            algo2_ETA_usage_rate0 = -1
            algo2_ETA_usage_rate1 = -1

            self.writter.add_scalar('Eval Late Order Rate/Algo2 Total', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Hired', algo2_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Crowdsourced', algo2_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total', algo2_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired', algo2_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Crowdsourced', algo2_ETA_usage_rate1, self.eval_num)

            message += "No order is dropped in Algo2\n"
            
        else:
            if algo2_count_dropped_orders0:                
                algo2_late_rate0 = round(algo2_late_orders0 / algo2_count_dropped_orders0, 2)
                algo2_ETA_usage_rate0 = round(algo2_ETA_usage0 / algo2_count_dropped_orders0, 2)
            else:
                algo2_late_rate0 = -1
                algo2_ETA_usage_rate0 = -1
                
            if algo2_count_dropped_orders1:                
                algo2_late_rate1 = round(algo2_late_orders1 / algo2_count_dropped_orders1, 2)
                algo2_ETA_usage_rate1 = round(algo2_ETA_usage1 / algo2_count_dropped_orders1, 2)
            else:
                algo2_late_rate1 = -1
                algo2_ETA_usage_rate1 = -1
                
            algo2_late_rate = round((algo2_late_orders0 + algo2_late_orders1) / (algo2_count_dropped_orders0 + algo2_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate}")

            algo2_ETA_usage_rate = round((algo2_ETA_usage0 + algo2_ETA_usage1) / (algo2_count_dropped_orders0 + algo2_count_dropped_orders1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo2: Hired - {algo2_ETA_usage_rate0}, Crowdsourced - {algo2_ETA_usage_rate1}, Total - {algo2_ETA_usage_rate}")

            self.writter.add_scalar('Eval Late Order Rate/Algo2 Total', algo2_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Hired', algo2_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo2 Crowdsourced', algo2_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Total', algo2_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 Hired', algo2_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo2 fCrowdsourced', algo2_ETA_usage_rate1, self.eval_num)

            message += f"Rate of Late Orders for Evaluation in Algo2: Hired - {algo2_late_rate0}, Crowdsourced - {algo2_late_rate1}, Total - {algo2_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo2: Hired - {algo2_ETA_usage_rate0}, Crowdsourced - {algo2_ETA_usage_rate1}, Total - {algo2_ETA_usage_rate}\n"
            
        if algo3_count_dropped_orders0 + algo3_count_dropped_orders1 == 0:
            print("No order is dropped in Algo3")
            algo3_late_rate = -1
            algo3_late_rate0 = -1
            algo3_late_rate1 = -1
            algo3_ETA_usage_rate = -1
            algo3_ETA_usage_rate0 = -1
            algo3_ETA_usage_rate1 = -1

            self.writter.add_scalar('Eval Late Order Rate/Algo3 Total', algo3_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Hired', algo3_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Crowdsourced', algo3_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total', algo3_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired', algo3_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced', algo3_ETA_usage_rate1, self.eval_num)
            
            message += "No order is dropped in Algo3\n"
        else:
            if algo3_count_dropped_orders0:                
                algo3_late_rate0 = round(algo3_late_orders0 / algo3_count_dropped_orders0, 2)
                algo3_ETA_usage_rate0 = round(algo3_ETA_usage0 / algo3_count_dropped_orders0, 2)
            else:
                algo3_late_rate0 = -1
                algo3_ETA_usage_rate0 = -1
                
            if algo3_count_dropped_orders1:                
                algo3_late_rate1 = round(algo3_late_orders1 / algo3_count_dropped_orders1, 2)
                algo3_ETA_usage_rate1 = round(algo3_ETA_usage1 / algo3_count_dropped_orders1, 2)
            else:
                algo3_late_rate1 = -1
                algo3_ETA_usage_rate1 = -1
                
            algo3_late_rate = round((algo3_late_orders0 + algo3_late_orders1) / (algo3_count_dropped_orders0 +algo3_count_dropped_orders1), 2)
            print(f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate}")

            algo3_ETA_usage_rate = round((algo3_ETA_usage0 + algo3_ETA_usage1) / (algo3_count_dropped_orders0 +algo3_count_dropped_orders1), 2)
            print(f"Rate of ETA Usage for Evaluation in Algo3: Hired - {algo3_ETA_usage_rate0}, Crowdsourced - {algo3_ETA_usage_rate1}, Total - {algo3_ETA_usage_rate}")
            
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Total', algo3_late_rate, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Hired', algo3_late_rate0, self.eval_num)
            self.writter.add_scalar('Eval Late Order Rate/Algo3 Crowdsourced', algo3_late_rate1, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Total', algo3_ETA_usage_rate, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Hired', algo3_ETA_usage_rate0, self.eval_num)
            self.writter.add_scalar('Eval ETA Usage Rate/Algo3 Crowdsourced', algo3_ETA_usage_rate1, self.eval_num)
            
            message += f"Rate of Late Orders for Evaluation in Algo3: Hired - {algo3_late_rate0}, Crowdsourced - {algo3_late_rate1}, Total - {algo3_late_rate}\n" + f"Rate of ETA Usage for Evaluation in Algo3: Hired - {algo3_ETA_usage_rate0}, Crowdsourced - {algo3_ETA_usage_rate1}, Total - {algo3_ETA_usage_rate}\n"

        logger.success(message)
            
        print("\n")
        
        return (
            algo1_eval_episode_rewards_sum,
            algo2_eval_episode_rewards_sum,
            algo3_eval_episode_rewards_sum,
            
            algo1_distance0,
            algo1_distance1,
            algo1_distance,
            
            algo2_distance0,
            algo2_distance1,
            algo2_distance,
            
            algo3_distance0,
            algo3_distance1,
            algo3_distance,
            
            algo1_avg0_speed,
            algo1_avg1_speed,
            algo1_avg_speed,
            
            algo2_avg0_speed,
            algo2_avg1_speed,
            algo2_avg_speed,
            
            algo3_avg0_speed,
            algo3_avg1_speed,
            algo3_avg_speed,
            
            algo1_overspeed0,
            algo1_overspeed1,
            algo1_overspeed,
            
            algo2_overspeed0,
            algo2_overspeed1,
            algo2_overspeed,
            
            algo3_overspeed0,
            algo3_overspeed1,
            algo3_overspeed,
            
            algo1_reject_rate_per_episode,
            algo2_reject_rate_per_episode,
            algo3_reject_rate_per_episode,
            
            algo1_reject0,
            algo1_reject1,
            algo1_reject,
            
            algo2_reject0,
            algo2_reject1,
            algo2_reject,
            
            algo3_reject0,
            algo3_reject1,
            algo3_reject,
            
            algo1_price_per_order0,
            algo1_price_per_order1,
            algo1_price_per_order,
            
            algo2_price_per_order0,
            algo2_price_per_order1,
            algo2_price_per_order,
            
            algo3_price_per_order0,
            algo3_price_per_order1,
            algo3_price_per_order,
            
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
            
            algo1_finish0,
            algo1_finish1,
            algo1_finish,
            
            algo2_finish0,
            algo2_finish1,
            algo2_finish,
            
            algo3_finish0,
            algo3_finish1,
            algo3_finish,
            
            algo1_avg0_leisure,
            algo1_avg1_leisure,
            algo1_avg_leisure,

            algo2_avg0_leisure,
            algo2_avg1_leisure,
            algo2_avg_leisure,
            
            algo3_avg0_leisure,
            algo3_avg1_leisure,
            algo3_avg_leisure,
            
            algo1_late_rate0,
            algo1_late_rate1,
            algo1_late_rate,
            
            algo2_late_rate0,
            algo2_late_rate1,
            algo2_late_rate,
            
            algo3_late_rate0,
            algo3_late_rate1,
            algo3_late_rate,
            
            algo1_ETA_usage_rate0,
            algo1_ETA_usage_rate1,
            algo1_ETA_usage_rate,
            
            algo2_ETA_usage_rate0,
            algo2_ETA_usage_rate1,
            algo2_ETA_usage_rate,
            
            algo3_ETA_usage_rate0,
            algo3_ETA_usage_rate1,
            algo3_ETA_usage_rate
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
