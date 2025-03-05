import numpy as np
from envs.env_data import Map
import copy

from geopy.distance import geodesic
import random
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor

class EnvCore(object):
    
    def __init__(self, map_algo_index):        

        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
        self.num_speeds = 7 # 1-7 m/s, 1-4 normal, 0 stay put, in the model the multidiscrete is set [0, 7], but later I want to set it to four choice: 1,3,5,7, later I use 1, 2, 3 to represent low(1-3), normal(3-4) and high(4-7) speed range        
                 
    def reset(self, env_index):
        self.map.reset(env_index)
    
    # def step(self):
    #     current_map = copy.copy(self.map)
        
    #     for agent in self.map.active_couriers:
            
    #         if agent.state == 'active':   
    #             if agent.current_waiting_time > 0:
    #                 if agent.current_waiting_time > self.map.interval:
    #                     agent.current_waiting_time -= self.map.interval
    #                 else:
    #                     agent.is_waiting = 0
    #                     self._pick_or_drop(agent)
                        
    #             if (agent.waybill != [] or agent.wait_to_pick != []) and agent.is_waiting == 0:
                    
    #                 agent.move(self.map, current_map)  
    #                 agent.actual_speed = agent.travel_distance / agent.total_riding_time if agent.total_riding_time != 0 else 0
                    
    #                 self._pick_or_drop(agent)
                                            
    #             if agent.waybill == [] and agent.wait_to_pick == []:
    #                 agent.is_leisure = 1
    #             else:
    #                 agent.is_leisure = 0
    #                 agent.leisure_time = self.map.clock

    def process_agent(self, agent, current_map):
        if agent.state == 'active':   
            if agent.current_waiting_time > 0:
                if agent.current_waiting_time > current_map.interval:
                    agent.current_waiting_time -= current_map.interval
                else:
                    agent.is_waiting = 0
                    self._pick_or_drop(agent)

            if (agent.waybill or agent.wait_to_pick) and agent.is_waiting == 0:
                agent.move(current_map)
                agent.actual_speed = agent.travel_distance / agent.total_riding_time if agent.total_riding_time != 0 else 0
                self._pick_or_drop(agent)

            if not agent.waybill and not agent.wait_to_pick:
                agent.is_leisure = 1
            else:
                agent.is_leisure = 0
                agent.leisure_time = current_map.clock
        
        return agent

    def step(self):
        current_map = copy.deepcopy(self.map)
        
        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda agent: self.process_agent(agent, current_map), 
                        self.map.active_couriers))

        self.map.grid = [[[] for _ in range(self.map.grid_size)] for _ in range(self.map.grid_size)]
        
        add_tasks = [
            (courier.position[0], courier.position[1], courier)
            for courier in self.map.active_couriers
        ]

        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda args: self.map.add_courier(*args), add_tasks))

    def get_map(self):
        return self.map
    
    def _pick_or_drop(self, agent):
        for order in agent.wait_to_pick:
            if agent.position == order.pick_up_point and self.map.clock >= order.meal_prepare_time: # picking up
                order.wait_time = self.map.clock - order.meal_prepare_time
                agent.pick_order(order)
                
            elif agent.position == order.pick_up_point and self.map.clock < order.meal_prepare_time and agent.is_waiting == 0:
                agent.current_waiting_time = order.meal_prepare_time - self.map.clock
                agent.total_waiting_time += order.meal_prepare_time - self.map.clock
                agent.is_waiting == 1
                
        for order in agent.waybill:
            if agent.position == order.drop_off_point:  # dropping off
                agent.drop_order(order)
                                    
                if self.map.clock > order.ETA:

                    agent.income += order.price * 0.7
                    order.is_late = 1
                                            
                else:
                    order.ETA_usage = (self.map.clock - order.order_create_time) / (order.ETA - order.order_create_time)
                    agent.income += order.price 
