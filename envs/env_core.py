import numpy as np
from envs.gridmap import GridMap
from envs.observationspace import ObservationSpace
from agent.courier import Courier
from agent.order import Order


class EnvCore(object):
    
    def __init__(self):        

        self.current_step = 0
        self.gridmap = GridMap()
        self.agent_num = self.gridmap.num_couriers
        self.obs_dim = self.gridmap.num_orders * 4 + 2 # orders: pick_up_point, drop_off_point, prepare_time, estimate_arrive_time; couriers: position, (num_waybill+num_wait_to_pick) * 2(distance_between_each_order + time_window)
        self.action_dim = 4
        self.agents = self.gridmap.couriers
                 
    def reset(self):
        self.gridmap.reset()
        
                        
    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        share_obs = []

        self.agents = self.gridmap.couriers
        
        # set action for each agent
        for i, agent in enumerate(self.agents):
            reward = 0

            self._set_action(action_n[i], agent)

            for orderid in agent.wait_to_pick:
                for order in self.gridmap.orders:
                    if order.orderid == orderid and agent.position == order.pick_up_point: # picking up
                            agent.pick_order(order)
                            reward += 10
                            
            for orderid in agent.waybill:
                for order in self.gridmap.orders:
                    if order.orderid == orderid and agent.position == order.drop_off_point:  # dropping off
                        agent.drop_order(order)
                        reward += 10
            reward -= 1
            reward_n.append(reward)

        # add new couriers or orders
        # if self.current_step == 10:
        #     self.gridmap.step()

        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            done_n.append(self._get_done(agent))
            info_n.append(self._get_info(agent))
        
        share_obs = ObservationSpace(self.gridmap, share=True).get_obs()


        return obs_n, reward_n, done_n, info_n, share_obs
    
    
    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        
        if agent.waybill != [] or agent.wait_to_pick != []:
            # 获取 one-hot 编码对应的动作索引
            action_index = np.argmax(action)
            if action_index == 0:  # Move Up
                agent.move('up')
            elif action_index == 1:  # Move Down
                agent.move('down')
            elif action_index == 2:  # Move Left
                agent.move('left')
            elif action_index == 3:  # Move Right
                agent.move('right')
            else:
                raise ValueError(f"Invalid action: {action_index}")
            
    def _get_obs(self, agent):
        return ObservationSpace(self.gridmap, agent).get_obs()
    
    def _get_done(self, agent):
        if agent.travel_distance >= self.gridmap.world_length and agent.waybill == [] and agent.wait_to_pick == []:
            return True
        else:
            return False
        
    def _get_info(self, agent):
        return {
            'courier': agent,
            'order': self.gridmap.orders
        }

    def return_gridmap(self):
        return self.gridmap