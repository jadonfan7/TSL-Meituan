import numpy as np
from envs.env_data import Map


import random
from scipy.spatial import KDTree


class EnvCore(object):
    
    def __init__(self, map_algo_index):        

        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
                 
    def reset(self, env_index):
        self.map.reset(env_index)
    
    def step(self):
        
        for i, agent in enumerate(self.map.couriers):
            if agent.current_waiting_time > 0:
                if agent.current_waiting_time >= self.map.interval:
                    agent.current_waiting_time -= self.map.interval
                else:
                    agent.is_waiting = 0
            if (agent.waybill != [] or agent.wait_to_pick != []) and agent.is_waiting == 0:
                        
                agent.target_location = agent.order_sequence[0][0]
                agent.move(self.map)  
                
                agent.actual_speed = agent.travel_distance / agent.total_riding_time if agent.total_riding_time != 0 else 0
                                                                        
            if agent.waybill == [] and agent.wait_to_pick == []:
                agent.is_leisure = 1
            else:
                agent.is_leisure = 0
                agent.leisure_time = self.map.clock
                    
        self.map.step()
                    
    def get_map(self):
        return self.map
    
    def get_env_obs(self):
        predicted_orders = self.map.get_predicted_orders()
        tree = self._build_tree(predicted_orders)
        obs = []
        for i, agent in enumerate(self.map.couriers):
            obs.append(self._get_obs(agent, predicted_orders, tree))
        return obs
    
    def get_env_space(self):
        return self.action_space, self.observation_space