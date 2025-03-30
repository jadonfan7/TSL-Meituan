import numpy as np
from envs.env_data import Map
from envs.observationspace import ObservationSpace

from envs.multi_discrete import MultiDiscrete
from gym.spaces import Box, Discrete

from geopy.distance import great_circle
import random
from scipy.spatial import KDTree

class EnvCore(object):
    
    def __init__(self, map_algo_index=0):        

        self.current_step = 0
        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
        
        self.action_space = []
        # self.obs_dim = self.map.couriers[0].capacity * 6 + 6 + 10 * 6 # orders: pick_up_point, drop_off_point, prepare_time, estimate_arrive_time; couriers: position, speed, target_position; env: time
        # self.obs_dim = self.map.couriers[0].capacity * 2 + 2 + 10 * 2
        self.obs_dim = self.map.couriers[0].capacity * 5 + 4 + 10 * 5

        self.observation_space = []
        
        self.shared_obs_dim = 0   
        
        for _ in range(self.num_agent):
            self.shared_obs_dim += self.obs_dim
            capacity = self.map.couriers[0].capacity

            action_space = MultiDiscrete([[0, 1], [0, 1], [0, capacity-1]])
            # action_space = MultiDiscrete([[0, 1], [0, speed_dim-1]])
            self.action_space.append(action_space)

            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32))
        
        self.share_observation_space = Box(low=0.0, high=1.0, shape=(self.shared_obs_dim,), dtype=np.float32)
                 
    def reset(self, env_index, eval=False):
        self.map.reset(env_index, eval)
    
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []

        for i, agent in enumerate(self.map.couriers):
            if agent.state == 'active':
                reward = self._set_action(action_n[i], agent)
            else:
                reward = 0
            reward_n.append(reward)
        
        total_cost = self.map.step()

        predicted_orders = self.map.get_predicted_orders()
        tree = self._build_tree(predicted_orders)
        
        for i, agent in enumerate(self.map.couriers):
            if reward_n[i] != 0:
                reward_n[i] -= total_cost
                
            if agent.state == 'active' and agent.waybill == [] and agent.wait_to_pick == []:
                reward_n[i] -= 10
            
            obs_n.append(self._get_obs(agent, predicted_orders, tree))
            done_n.append(self._get_done(agent)) 
            
        share_obs_n = np.concatenate(obs_n)

        return np.stack(obs_n), np.array(reward_n), np.array(done_n), share_obs_n
    
    # set env action for a particular agent
    def _set_action(self, action, agent):
        reward = 0
        
        if agent.current_waiting_time > 0:
            if agent.current_waiting_time >= self.map.interval:
                agent.current_waiting_time -= self.map.interval
            else:
                agent.is_waiting = 0
                   
        
        if (agent.waybill != [] or agent.wait_to_pick != []) and agent.is_waiting == 0:
            
            agent.save = np.argmax(action[:2])
            
            speed_index = np.argmax(action[2:5])
            if speed_index == 0:
                agent.speed = 4
            elif speed_index == 1:
                agent.speed = 7

            if agent.speed > 4:
                if agent.courier_type == 0:
                    reward -= 20
                else:
                    reward -= 15
                        
            waybill_length = len(agent.waybill)
            wait_to_pick_length = len(agent.wait_to_pick)
            total_length = waybill_length + wait_to_pick_length
            
            order_index = np.argmax(action[5:])
            if order_index > total_length - 1:
                reward -= 50
                agent.target_location = agent.order_sequence[0][0]
            else:
                reward += 10
                # if order_index < waybill_length:
                #     agent.target_location = agent.waybill[order_index].drop_off_point
                # else:
                #     agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
                if order_index < waybill_length:
                    target_loc = agent.waybill[order_index].drop_off_point
                else:
                    target_loc = agent.wait_to_pick[order_index - waybill_length].pick_up_point
                
                if target_loc != agent.order_sequence[0][0]:
                    dist1 = great_circle(agent.position, target_loc).meters
                    dist2 = great_circle(agent.position, agent.order_sequence[0][0]).meters
                    if dist1 - dist2 > 300:
                        agent.target_location = agent.order_sequence[0][0]
                    else:
                        agent.target_location = target_loc
                        reward += (dist2 - dist1) / 10
                else:
                    agent.target_location = agent.order_sequence[0][0]

            if agent.speed != 0:
                reward += agent.move(self.map)  
            
            agent.actual_speed = agent.travel_distance / agent.total_riding_time if agent.total_riding_time != 0 else 0
            
            agent.reward += reward
                                                    
        if agent.waybill == [] and agent.wait_to_pick == []:
            agent.is_leisure = 1
        else:
            agent.is_leisure = 0
            agent.leisure_time = self.map.clock

        return reward
    
    def _build_tree(self, predicted_orders):
        order_coords = np.array([order.pick_up_point for order in predicted_orders])
        tree = KDTree(order_coords)
        
        return tree
        
    def _get_obs(self, agent, predicted_orders, tree):
        if agent.state == 'inactive':
            return np.full((self.obs_dim,), -1)
        return ObservationSpace(self.map, agent).get_obs(predicted_orders, tree)
    
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
    
    def get_env_obs(self):
        predicted_orders = self.map.get_predicted_orders()
        tree = self._build_tree(predicted_orders)
        obs = []
        for i, agent in enumerate(self.map.couriers):
            obs.append(self._get_obs(agent, predicted_orders, tree))
        return obs
    
    def get_env_space(self):
        return self.action_space, self.observation_space