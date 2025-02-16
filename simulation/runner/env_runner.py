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
    def __init__(self):
        super(EnvRunner, self).__init__()

    def run(self, current_eval_time):
        
        self.eval_envs.reset(current_eval_time % 5)
        
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

        for eval_step in range(self.eval_episodes_length):
            
            # print("-"*25)
            print(f"THIS IS EVAL STEP {eval_step}")

            for i in range(self.num_envs):
                
                self.log_env(eval_step, i)
                            
            self.eval_envs.step()
                        
            for i in range(self.eval_envs.num_envs):
                if i == 0:
                    for c in self.eval_envs.envs_map[i].couriers:
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
                    for c in self.eval_envs.envs_map[i].couriers:
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
                    for c in self.eval_envs.envs_map[i].couriers:
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
                    for c in self.eval_envs.envs_map[i].couriers:
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
                    for c in self.eval_envs.envs_map[i].couriers:
                        if c.state == 'active':
                            if c.courier_type == 0:
                                algo5_num_active_couriers0 += 1
                                if c.speed > 4:
                                    algo5_count_overspeed0 += 1
                            else:
                                algo5_num_active_couriers1 += 1
                                if c.speed > 4:
                                    algo5_count_overspeed1 += 1
                                    
            self.eval_envs.eval_env_step()
            
        # Evaluation over periods
        for i in range(self.eval_envs.num_envs):
            if i == 0:
                platform_cost1 += self.eval_envs.envs_map[i].platform_cost
                for c in self.eval_envs.envs_map[i].couriers:
                    if c.courier_type == 0:
                        algo1_Hired_num += 1
                        if c.travel_distance > 0:
                            algo1_Hired_distance_per_episode.append(c.travel_distance)
                        algo1_Hired_finish_num.append(c.finish_order_num)
                        algo1_Hired_leisure_time.append(c.total_leisure_time)
                        algo1_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo1_Hired_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo1_Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    else:
                        algo1_Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            algo1_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo1_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo1_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo1_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo1_Crowdsourced_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo1_Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                        if c.state == 'active':
                            algo1_Crowdsourced_on += 1

                
                for o in self.eval_envs.envs_map[i].orders:
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
                platform_cost2 += self.eval_envs.envs_map[i].platform_cost
                for c in self.eval_envs.envs_map[i].couriers:
                    if c.courier_type == 0:
                        algo2_Hired_num += 1
                        if c.travel_distance > 0:
                            algo2_Hired_distance_per_episode.append(c.travel_distance)
                        algo2_Hired_finish_num.append(c.finish_order_num)
                        algo2_Hired_leisure_time.append(c.total_leisure_time)
                        algo2_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo2_Hired_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo2_Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    else:
                        algo2_Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            algo2_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo2_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo2_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo2_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo2_Crowdsourced_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo2_Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                        if c.state == 'active':
                            algo2_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_map[i].orders:
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
                platform_cost3 += self.eval_envs.envs_map[i].platform_cost
                for c in self.eval_envs.envs_map[i].couriers:
                    if c.courier_type == 0:
                        algo3_Hired_num += 1
                        if c.travel_distance > 0:
                            algo3_Hired_distance_per_episode.append(c.travel_distance)
                        algo3_Hired_finish_num.append(c.finish_order_num)
                        algo3_Hired_leisure_time.append(c.total_leisure_time)
                        algo3_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo3_Hired_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo3_Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    else:
                        algo3_Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            algo3_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo3_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo3_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo3_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo3_Crowdsourced_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo3_Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                        if c.state == 'active':
                            algo3_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_map[i].orders:
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
                platform_cost4 += self.eval_envs.envs_map[i].platform_cost
                for c in self.eval_envs.envs_map[i].couriers:
                    if c.courier_type == 0:
                        algo4_Hired_num += 1
                        if c.travel_distance > 0:
                            algo4_Hired_distance_per_episode.append(c.travel_distance)
                        algo4_Hired_finish_num.append(c.finish_order_num)
                        algo4_Hired_leisure_time.append(c.total_leisure_time)
                        algo4_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo4_Hired_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo4_Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    else:
                        algo4_Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            algo4_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo4_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo4_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo4_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo4_Crowdsourced_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo4_Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                        if c.state == 'active':
                            algo4_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_map[i].orders:
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
                platform_cost5 += self.eval_envs.envs_map[i].platform_cost
                for c in self.eval_envs.envs_map[i].couriers:
                    if c.courier_type == 0:
                        algo5_Hired_num += 1
                        if c.travel_distance > 0:
                            algo5_Hired_distance_per_episode.append(c.travel_distance)
                        algo5_Hired_finish_num.append(c.finish_order_num)
                        algo5_Hired_leisure_time.append(c.total_leisure_time)
                        algo5_Hired_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo5_Hired_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo5_Hired_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                    else:
                        algo5_Crowdsourced_num += 1
                        if c.travel_distance > 0:
                            algo5_Crowdsourced_distance_per_episode.append(c.travel_distance)
                        algo5_Crowdsourced_finish_num.append(c.finish_order_num)
                        algo5_Crowdsourced_leisure_time.append(c.total_leisure_time)
                        algo5_Crowdsourced_running_time.append(c.total_running_time)
                        if c.avg_speed > 0:
                            algo5_Crowdsourced_avg_speed.append(c.avg_speed)
                        if c.income > 0:
                            algo5_Crowdsourced_income.append(c.income / (c.total_running_time + c.total_leisure_time) * 3600)
                        if c.state == 'active':
                            algo5_Crowdsourced_on += 1
                
                for o in self.eval_envs.envs_map[i].orders:
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
            
            message += f"Rate of Late Orders for Evaluation in Algo1: Hired - {algo1_late_rate0}, Crowdsourced - {algo1_late_rate1}, Total - {algo1_late_rate} out of ({algo1_count_dropped_orders0+algo1_count_dropped_orders1})\n" + f"Rate of ETA Usage for Evaluation in Algo1: Hired - {algo1_ETA_usage_rate0} (Var: {algo1_var0_ETA}), Crowdsourced - {algo1_ETA_usage_rate1} (Var: {algo1_var1_ETA}), Total - {algo1_ETA_usage_rate} (Var: {algo1_var_ETA})\n"
        
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
        logger.success(message)
            
        print("\n")
        
        self.eval_envs.close()