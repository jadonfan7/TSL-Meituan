import numpy as np
from envs.env_data import Map
from envs.observationspace import ObservationSpace

import random

class EnvCore(object):
    
    def __init__(self):        

        self.current_step = 0
        self.map = Map()
        self.num_agent = self.map.num_couriers
        self.num_speeds = 7 # 0-7 m/s, 1-4 normal, 0 stay put, in the model the multidiscrete is set [0, 7]
        self.action_space = []
        self.obs_dim = self.map.couriers[0].capacity * 6 + 2 # orders: pick_up_point, drop_off_point, prepare_time, estimate_arrive_time; couriers: position, (num_waybill+num_wait_to_pick) * 2(distance_between_each_order + time_window)

        self.epsilon = 0.05
                 
    def reset(self):
        self.map.reset()
        self.map.__init__()
                        
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        # share_obs = []

        # set action for each agent
        for i, agent in enumerate(self.map.couriers):
            self.action_dim = len(agent.waybill) + len(agent.wait_to_pick) + self.num_speeds

            reward = 0
            reward = self._set_action(action_n[i], agent)

            reward_n.append(reward)

        
        # self.update_action_space()

        for i, agent in enumerate(self.map.couriers):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))
        
        # share_obs = ObservationSpace(self.map).get_obs()     

        self.num_agent = self.map.num_couriers
        # self.obs_dim = self.map.num_orders * 6 + 2

        # return obs_n, reward_n, done_n, info_n, share_obs
        return obs_n, reward_n, done_n, info_n
    
    # set env action for a particular agent
    def _set_action(self, action, agent):
        reward = 0
        
        if agent.waybill != [] or agent.wait_to_pick != []:
            waybill_length = len(agent.waybill)
            wait_to_pick_length = len(agent.wait_to_pick)
            total_length = waybill_length + wait_to_pick_length
            index = self.map.couriers[0].capacity

            if np.argmax(action[:index]) > total_length - 1:
                reward -= 100
                order_index = np.random.randint(0, total_length)
            else:
                order_index = np.argmax(action[:index])

            agent.speed =  np.argmax(action[index:]) + 1

            if agent.speed > 4:
                reward -= (agent.speed - 4) ** 2 * 50
                # reward -= 100

            if order_index < waybill_length:
                if agent.target_location == None:
                    agent.target_location = agent.waybill[order_index].drop_off_point
                    agent.is_target_locked = True
                elif not agent.is_target_locked and random.random() < self.epsilon:
                    agent.target_location = agent.waybill[order_index].drop_off_point
                    agent.is_target_locked = True
                agent.move(self.map.interval)
            elif order_index >= waybill_length and order_index < wait_to_pick_length + waybill_length:
                if agent.target_location == None:
                    agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
                    agent.is_target_locked = True
                elif not agent.is_target_locked and random.random() < self.epsilon:
                    agent.target_location = agent.wait_to_pick[order_index - waybill_length].pick_up_point
                    agent.is_target_locked = True
                agent.move(self.map.interval) 
        else:
            agent.speed = 0
        

        # reward -= dist        
        
        for order in agent.wait_to_pick:
            if agent.position == order.pick_up_point: # picking up
                agent.pick_order(order)
                reward += 200
                            
        for order in agent.waybill:
            if agent.position == order.drop_off_point:  # dropping off
                agent.drop_order(order)
                if self.map.clock - order.pair_time > order.ETA:
                    reward -= 250 * ((self.map.clock - order.pair_time) / order.ETA - 1)
                    order.is_late = 1
                else:
                    order.ETA_usage = (self.map.clock - order.pair_time) / order.ETA
                    reward += 200 * order.ETA_usage

        return reward
            
    def _get_obs(self, agent):
        return ObservationSpace(self.map, agent).get_obs()
    
    def _get_done(self, agent):
        if agent.waybill == [] and agent.wait_to_pick == []:
            return True
        else:
            return False
        # if self.map.current_index < 654344: # num of the data
        #     return False
        
        # all_dropped = all(order.status == 'dropped' for order in self.orders)

        # if agent.waybill == [] and agent.wait_to_pick == [] and all_dropped:
        #     return True
        # else:
        #     return False
        
    def _get_info(self, agent):
        return {
            'courier': agent,
            'order': self.map.orders
        }
    
    # def update_action_space(self):
    #     self.action_space = []
    #     for agent_idx in range(self.num_agent):
    #         order_dim = len(self.agents[agent_idx].waybill) + len(self.agents[agent_idx].wait_to_pick)
    #         speed_dim = self.num_speeds
    #         if order_dim == 0:
    #             action_space = MultiDiscrete([[0, 0], [0, speed_dim]]) # [0, 0] just for the requirement of the form
    #         else:
    #             action_space = MultiDiscrete([[0, order_dim - 1], [0, speed_dim]])
    #         self.action_space.append(action_space)
