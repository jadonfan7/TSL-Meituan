import numpy as np
from envs.env_data import Map

from geopy.distance import geodesic
import random
from scipy.spatial import KDTree

class EnvCore(object):
    
    def __init__(self, map_algo_index=0):        

        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
        self.num_speeds = 7 # 1-7 m/s, 1-4 normal, 0 stay put, in the model the multidiscrete is set [0, 7], but later I want to set it to four choice: 1,3,5,7, later I use 1, 2, 3 to represent low(1-3), normal(3-4) and high(4-7) speed range        
                 
    def reset(self, env_index):
        self.map.reset(env_index)
    
    def step(self):
        
        for agent in self.map.couriers:       
            if (agent.waybill != [] or agent.wait_to_pick != []) and agent.stay_duration == 0:

                agent.move(self.map)  
                agent.avg_speed = agent.travel_distance / agent.riding_time if agent.riding_time != 0 else 0
                
                for order in agent.wait_to_pick:
                    if agent.position == order.pick_up_point and self.map.clock >= order.meal_prepare_time: # picking up
                        order.wait_time = self.map.clock - order.meal_prepare_time
                        agent.pick_order(order)
                        
                    elif agent.position == order.pick_up_point and self.map.clock < order.meal_prepare_time:
                        agent.stay_duration = np.ceil((order.meal_prepare_time - self.map.clock) / self.map.interval)
                        
                for order in agent.waybill:
                    if agent.position == order.drop_off_point:  # dropping off
                        agent.drop_order(order)
                        
                        agent.finish_order_num += 1
                            
                        if self.map.clock > order.ETA:

                            agent.income += order.price * 0.7
                            order.is_late = 1
                                                    
                        else:
                            order.ETA_usage = (self.map.clock - order.order_create_time) / (order.ETA - order.order_create_time)
                            agent.income += order.price 
            else:
                agent.speed = 0
                if agent.stay_duration != 0:
                    agent.stay_duration -= 1
                                        
            if agent.waybill == [] and agent.wait_to_pick == []:
                agent.is_leisure = 1
            else:
                agent.is_leisure = 0
                agent.leisure_time = self.map.clock
                
    def get_map(self):
        return self.map