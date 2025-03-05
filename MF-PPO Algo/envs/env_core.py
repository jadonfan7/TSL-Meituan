import numpy as np
from envs.env_data import Map
from envs.observationspace import ObservationSpace

from envs.multi_discrete import MultiDiscrete
from gym.spaces import Box, Discrete

from geopy.distance import geodesic
import random
from scipy.spatial import KDTree
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

class EnvCore(object):
    
    def __init__(self, map_algo_index=0):        

        self.current_step = 0
        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
        self.num_speeds = 7 # 1-7 m/s, 1-4 normal, 0 stay put, in the model the multidiscrete is set [0, 7], but later I want to set it to four choice: 1,3,5,7, later I use 1, 2, 3 to represent low(1-3), normal(3-4) and high(4-7) speed range
        
        self.action_space = []
        self.obs_dim = (self.map.couriers[0].capacity * 6 + 6) * 11 # orders: pick_up_point, drop_off_point, prepare_time, estimate_arrive_time; couriers: position, speed, target_position; env: time
        
        self.observation_space = []
        self.epsilon = 0.05

        # shared_obs_dim = self.obs_dim * self.num_agent
        
        self.shared_obs_dim = 0
        self.share_observation_space = []
        
        for _ in range(self.num_agent):
            
            self.shared_obs_dim += self.obs_dim
            order_dim = self.map.couriers[0].capacity
            speed_dim = self.num_speeds

            action_space = MultiDiscrete([[0, order_dim - 1], [0, speed_dim - 1]])
            # action_space = MultiDiscrete([[0, 1], [0, speed_dim-1]])
            self.action_space.append(action_space)

            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32))
            
        self.share_observation_space.append(Box(low=0.0, high=1.0, shape=(self.shared_obs_dim,), dtype=np.float32))
                 
    def reset(self, env_index, eval=False):
        self.map.reset(env_index, eval)
    
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        
        share_obs = []
        
        # set action for each agent
        for i, agent in enumerate(self.map.couriers):
            if agent.state == 'active':
                reward = self._set_action(action_n[i], agent)
            else:
                reward = 0
            reward_n.append(reward)
                
        self.map.grid = [[[] for _ in range(self.map.grid_size)] for _ in range(self.map.grid_size)]
        
        for courier in self.map.active_couriers:
            self.map.add_courier(courier.position[0], courier.position[1], courier)
                    
        for i, agent in enumerate(self.map.couriers):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
                
        for i, agent in enumerate(self.map.couriers):
            share_obs.append(self._get_local_share_obs(agent, k=10))
        
        return np.stack(obs_n), np.array(reward_n), np.array(done_n), np.stack(share_obs)
    
    # set env action for a particular agent
    def _set_action(self, action, agent):
        if agent.state != 'active':
            return 0
        
        current_map = copy.deepcopy(self.map)
        
        reward = 0
        
        if agent.current_waiting_time > 0:
            if agent.current_waiting_time > self.map.interval:
                agent.current_waiting_time -= self.map.interval
            else:
                agent.is_waiting = 0
                reward += self._pick_or_drop(agent)   
        
        if (agent.waybill != [] or agent.wait_to_pick != []) and agent.is_waiting == 0:

            waybill_length = len(agent.waybill)
            wait_to_pick_length = len(agent.wait_to_pick)
            total_length = waybill_length + wait_to_pick_length
            index = self.map.couriers[0].capacity

            order_index = 0
            if np.argmax(action[:index]) > total_length - 1:
                reward -= 150
                agent.target_location = agent.order_sequence[0][0]
            else:
                order_index = np.argmax(action[:index])
                if order_index < waybill_length:
                    target_loc = agent.waybill[order_index].drop_off_point
                else:
                    target_loc = agent.wait_to_pick[order_index - waybill_length].pick_up_point
                
                if target_loc != agent.target_location:
                    dist1 = geodesic(agent.position, target_loc).meters
                    dist2 = geodesic(agent.position, agent.target_location).meters
                    reward += (dist2 - dist1) / 10
                
                agent.target_location = target_loc
            
            agent.speed =  np.argmax(action[index:]) + 1

            if agent.speed > 4:
                if agent.courier_type == 0:
                    reward -= (agent.speed - 4) ** 2 * 10
                else:
                    reward -= (agent.speed - 4) ** 2 * 5
            else:
                if agent.courier_type == 0:
                    reward -= (agent.speed - 4) ** 2 * 2
                else:
                    reward -= (agent.speed - 4) ** 2 * 3            
                
            # if order_index < waybill_length:
            #     if agent.target_location == None:
            #         agent.target_location = agent.waybill[order_index].drop_off_point
            #         agent.is_target_locked = True
            #     # elif not agent.is_target_locked and random.random() < self.epsilon:
            #     elif not agent.is_target_locked:
            #         agent.target_location = agent.waybill[order_index].drop_off_point
            #         agent.is_target_locked = True
            # elif order_index >= waybill_length and order_index < wait_to_pick_length + waybill_length:
            #     if agent.target_location == None:
            #         agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
            #         agent.is_target_locked = True
            #     # elif not agent.is_target_locked and random.random() < self.epsilon:
            #     elif not agent.is_target_locked:
            #         agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
            #         agent.is_target_locked = True
            
            agent.move(current_map)  
            agent.actual_speed = agent.travel_distance / agent.total_riding_time if agent.total_riding_time != 0 else 0
            
            reward -= agent.congestion_rate * 100
            reward += self._pick_or_drop(agent)
                                        
            if agent.waybill == [] and agent.wait_to_pick == []:
                agent.is_leisure = 1
            else:
                agent.is_leisure = 0
                agent.leisure_time = self.map.clock

            agent.reward += reward

        return reward
    
    def _get_obs(self, agent):
        if agent.state == 'inactive':
            return np.full((66,), -1)
        return ObservationSpace(self.map, agent).get_obs()
    
    def _get_done(self, agent):
        if agent.state == 'inactive':
            return True
        else:
            return False
                        
    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    
    def get_map(self):
        return self.map
    
    # def get_env_obs(self):
    #     obs = []
    #     for i, agent in enumerate(self.map.couriers):
    #         obs.append(self._get_obs(agent))
    #     return obs
    
    def get_env_obs(self):
        with ThreadPoolExecutor() as executor:
            obs = list(executor.map(self._get_obs, self.map.couriers))
        return obs

    def get_env_space(self):
        return self.action_space, self.observation_space
    
    def _get_local_share_obs(self, agent, k=10):
        agents_nearby = self.map.get_couriers_in_adjacent_grids(agent.position[0], agent.position[1])
        local_share_obs = []
        if agents_nearby == []:
            local_share_obs = np.full((10, 66), -1)
        elif len(agents_nearby) < k:
            for agent in agents_nearby:
                local_share_obs.append(self._get_obs(agent))
            local_share_obs = np.array(local_share_obs)
            local_share_obs = np.pad(local_share_obs, ((0, k - len(agents_nearby)), (0, 0)), 'constant', constant_values=-1)
        else:
            agent_positions = np.array([a.position for a in agents_nearby])
        
            tree = KDTree(agent_positions)

            _, neighbor_idx = tree.query(agent.position, k=k)
            for idx in neighbor_idx:
                local_share_obs.append(self._get_obs(agents_nearby[idx]))
            local_share_obs = np.array(local_share_obs)
        
        local_share_obs = local_share_obs.flatten().reshape(-1)
        return local_share_obs
    
    def _pick_or_drop(self, agent):
        reward = 0
        for order in agent.wait_to_pick:
            if agent.position == order.pick_up_point and self.map.clock >= order.meal_prepare_time: # picking up
                order.wait_time = self.map.clock - order.meal_prepare_time
                agent.pick_order(order)
                
                if agent.courier_type == 0:
                    reward += 40
                else:
                    reward += 60
                    
            elif agent.position == order.pick_up_point and self.map.clock < order.meal_prepare_time and agent.is_waiting == 0:
                waiting_time = order.meal_prepare_time - self.map.clock
                agent.current_waiting_time = waiting_time
                agent.total_waiting_time += waiting_time
                agent.is_waiting == 1
                reward -= waiting_time / 60 * 6
                
        for order in agent.waybill:
            if agent.position == order.drop_off_point:  # dropping off
                agent.drop_order(order)
                                    
                if self.map.clock > order.ETA:
                    if agent.courier_type == 0:
                        reward -= 30 + 80 * ((self.map.clock - order.order_create_time) / (order.ETA - order.order_create_time) - 1)
                    else:
                        reward -= 50 + 100 * ((self.map.clock - order.order_create_time) / (order.ETA - order.order_create_time) - 1)
                        
                    agent.income += order.price * 0.7
                    order.is_late = 1
                                            
                else:
                    order.ETA_usage = (self.map.clock - order.order_create_time) / (order.ETA - order.order_create_time)
                    if agent.courier_type == 0:
                        reward += 80 + 80 * (1 - order.ETA_usage)
                    else:
                        reward += 100 + 100 * (1 - order.ETA_usage)
                        
                    agent.income += order.price 
        return reward
    