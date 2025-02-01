import numpy as np
from envs.env_data import Map
from envs.observationspace import ObservationSpace

from envs.multi_discrete import MultiDiscrete
from gym.spaces import Box, Discrete

from geopy.distance import geodesic
import random

class EnvCore(object):
    
    def __init__(self, map_algo_index=0):        

        self.current_step = 0
        self.map_algo_index = map_algo_index
        self.map = Map(algo_index=map_algo_index)
        self.num_agent = self.map.num_couriers
        self.num_speeds = 4 # 1-7 m/s, 1-4 normal, 0 stay put, in the model the multidiscrete is set [0, 7], but later I want to set it to four choice: 2,4,5.5,7, later I use 1, 2, 3 to represent low(1-3), normal(3-4) and high(4-7) speed range
        
        self.action_space = []
        self.obs_dim = self.map.couriers[0].capacity * 5 + 2 # orders: pick_up_point, drop_off_point, prepare_time, estimate_arrive_time; couriers: position, (num_waybill+num_wait_to_pick) * 2(distance_between_each_order + time_window)
        
        self.observation_space = []
        self.epsilon = 0.05
        
        for _ in range(self.num_agent):

            order_dim = self.map.couriers[0].capacity
            speed_dim = self.num_speeds

            # action_space = MultiDiscrete([[0, order_dim - 1], [0, speed_dim - 1]])
            action_space = MultiDiscrete([[0, 1], [0, speed_dim-1]])
            # action_space = Discrete(speed_dim)
            self.action_space.append(action_space)

            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32))
                 
    def reset(self, env_index, eval=False):
        self.map.reset(env_index, eval)
                        
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        # share_obs = []

        # set action for each agent
        for i, agent in enumerate(self.map.couriers):
            # self.action_dim = len(agent.waybill) + len(agent.wait_to_pick) + self.num_speeds

            reward = 0
            reward = self._set_action(action_n[i], agent)

            reward_n.append(reward)

        
        # self.update_action_space()

        for i, agent in enumerate(self.map.couriers):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))
        
        # self.num_agent = self.map.num_couriers
        # self.obs_dim = self.map.num_orders * 6 + 2
        

        # return obs_n, reward_n, done_n, info_n, share_obs
        return np.stack(obs_n), np.array(reward_n), np.array(done_n), info_n
    
    # set env action for a particular agent
    def _set_action(self, action, agent):
        
        reward = 0
        
        if (agent.waybill != [] or agent.wait_to_pick != []) and agent.stay_duration == 0:

            # waybill_length = len(agent.waybill)
            # wait_to_pick_length = len(agent.wait_to_pick)
            # total_length = waybill_length + wait_to_pick_length
            # index = self.map.couriers[0].capacity

            # if np.argmax(action[:index]) > total_length - 1:
            #     reward -= 200
            #     order_index = np.random.randint(0, total_length)
            # else:
            #     order_index = np.argmax(action[:index])
            
            # agent.speed =  np.argmax(action[index:]) + 1

            # # speed_index = np.argmax(action[index:])
            # # if speed_index == 0:
            # #     agent.speed = np.random.uniform(1, 2.5)
            # # elif speed_index == 1:
            # #     agent.speed = np.random.uniform(2.5, 4)
            # # else:
            # #     agent.speed = np.random.uniform(4, 7)
            
            policy = np.argmax(action[:2])
            speed_index = np.argmax(action[2:])
            if speed_index == 0:
                agent.speed = 2
            elif speed_index == 1:
                agent.speed = 4
            elif speed_index == 2:
                agent.speed = 5.5
            else:
                agent.speed = 7
            # agent.speed =  np.argmax(action) + 1
            
            if agent.speed > 4:
                if agent.courier_type == 0:
                    reward -= (agent.speed - 4) ** 2 * 10
                else:
                    reward -= (agent.speed - 4) ** 2 * 5
            else:
                if agent.courier_type == 0:
                    reward -= (agent.speed - 4) ** 2 * 5
                else:
                    reward -= (agent.speed - 4) ** 2 * 2

            # if order_index < waybill_length:
            #     if agent.target_location == None:
            #         agent.target_location = agent.waybill[order_index].drop_off_point
            #         agent.is_target_locked = True
            #     # elif not agent.is_target_locked and random.random() < self.epsilon:
            #     elif not agent.is_target_locked:
            #         agent.target_location = agent.waybill[order_index].drop_off_point
            #         agent.is_target_locked = True
            #     agent.move(self.map.interval)
            # elif order_index >= waybill_length and order_index < wait_to_pick_length + waybill_length:
            #     if agent.target_location == None:
            #         agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
            #         agent.is_target_locked = True
            #     # elif not agent.is_target_locked and random.random() < self.epsilon:
            #     elif not agent.is_target_locked:
            #         agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
            #         agent.is_target_locked = True
            #     agent.move(self.map.interval) 
            
            all_orders = agent.waybill + agent.wait_to_pick
            
            if policy == 0:
                def calculate_distance(courier_position, order):
                    if order.status == 'picked_up':
                        return geodesic(courier_position, order.drop_off_point).meters
                    elif order.status == 'wait_pick':
                        return geodesic(courier_position, order.pick_up_point).meters

                order_sequence = sorted(all_orders, key=lambda order: calculate_distance(agent.position, order))
            else:
                order_sequence = sorted(all_orders, key=lambda order: order.ETA)
            
            # agent.order_sequence = self._cal_sequence(all_orders, agent)
            
            if order_sequence[0].status == 'wait_pick':
                agent.target_location = agent.order_sequence[0].pick_up_point
            elif order_sequence[0].status == 'picked_up':
                agent.target_location = agent.order_sequence[0].drop_off_point
                
            agent.move(self.map.interval)  
            agent.avg_speed = agent.travel_distance / agent.riding_time if agent.riding_time != 0 else 0
            
            for order in agent.wait_to_pick:
                if agent.position == order.pick_up_point and self.map.clock >= order.meal_prepare_time: # picking up
                    agent.pick_order(order)
                    
                    if agent.courier_type == 0:
                        reward += 40
                    else:
                        reward += 60
                        
                elif agent.position == order.pick_up_point and self.map.clock < order.meal_prepare_time:
                    agent.stay_duration = np.ceil((order.meal_prepare_time - self.map.clock) / self.map.interval)
                    
            for order in agent.waybill:
                if agent.position == order.drop_off_point:  # dropping off
                    agent.drop_order(order)
                    
                    agent.finish_order_num += 1
                        
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
                            reward += 50 + 80 * (1 - order.ETA_usage)
                        else:
                            reward += 70 + 80 * (1 - order.ETA_usage)
                            
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

        
        agent.reward += reward

        return reward
            
    def _get_obs(self, agent):
        return ObservationSpace(self.map, agent).get_obs()
    
    def _get_done(self, agent):
        if agent.waybill == [] and agent.wait_to_pick == []:
            return True
        else:
            return False
                
    def _get_info(self, agent):
        return {
            'courier': agent,
            'order': self.map.orders
        }
        
    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)
    
    def get_map(self):
        return self.map
    
    def get_env_obs(self):
        obs = []
        for i, agent in enumerate(self.map.couriers):
            obs.append(self._get_obs(agent))
        return obs

    def adjust(self):
        for _ in range(self.map.add_new_couriers):

            order_dim = self.map.couriers[0].capacity
            speed_dim = self.num_speeds
            
            # action_space = MultiDiscrete([[0, order_dim - 1], [0, speed_dim - 1]])
            action_space = MultiDiscrete([[0, 1], [0, speed_dim-1]])
            # action_space = Discrete(speed_dim)
            self.action_space.append(action_space)

            self.observation_space.append(Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32))
        
        return self.action_space, self.observation_space