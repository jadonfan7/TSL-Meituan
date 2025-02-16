import math
from geopy.distance import geodesic
from agent.courier import Courier
from agent.order import Order
import pandas as pd
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import joblib


class Map:
    def __init__(self, env_index=0, algo_index=0, eval=False):
        self.orders_id = set()
        self.couriers_id = set()
        self.orders = []
        self.couriers = []
        
        self.num_couriers1 = 0
        self.num_couriers2 = 0
        
        self.env_index = env_index
        self.algo_index = algo_index
        self.current_index = 0
    
        self.platform_cost = 0
        
        df = pd.read_csv('../all_waybill_info_meituan_0322.csv')
        # df = pd.read_csv('all_waybill_info_meituan_0322.csv')
        
        order_num_estimate = pd.read_csv('MF-PPO Algo/order_prediction/order_num_estimation.csv')
        # order_num_estimate = pd.read_csv('/share/home/tj23028/TSL/test/order_prediction/order_num_estimation.csv')

        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665982800},
        #     1: {'date': 20221017, 'start_time': 1665997200, 'end_time': 1666004400},
        #     2: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666069200},
        #     3: {'date': 20221018, 'start_time': 1666083600, 'end_time': 1666090800},
        #     4: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666155600},
        #     5: {'date': 20221019, 'start_time': 1666170000, 'end_time': 1666177200},
        #     6: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666242000},
        #     7: {'date': 20221020, 'start_time': 1666256400, 'end_time': 1666263600},
        #     8: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666328400},
        #     9: {'date': 20221021, 'start_time': 1666342800, 'end_time': 1666350000},
        #     10: {'date': 20221022, 'start_time': 1666407600, 'end_time': 1666414800},
        # } # 11:00-13:00, 17:00-19:00
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665975600 + 3600},
        #     1: {'date': 20221017, 'start_time': 1666000800, 'end_time': 1666000800 + 3600},
        #     2: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666062000 + 3600},
        #     3: {'date': 20221018, 'start_time': 1666087200, 'end_time': 1666087200 + 3600},
        #     4: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666148400 + 3600},
        #     5: {'date': 20221019, 'start_time': 1666173600, 'end_time': 1666173600 + 3600},
        #     6: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666234800 + 3600},
        #     7: {'date': 20221020, 'start_time': 1666260000, 'end_time': 1666260000 + 3600},
        #     8: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666321200 + 3600},
        #     9: {'date': 20221021, 'start_time': 1666346400, 'end_time': 1666346400 + 3600},
        # } # 1h
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665977400},
        #     1: {'date': 20221017, 'start_time': 1666000800, 'end_time': 1666002600},
        #     2: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666063800},
        #     3: {'date': 20221018, 'start_time': 1666087200, 'end_time': 1666089000},
        #     4: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666150200},
        #     5: {'date': 20221019, 'start_time': 1666173600, 'end_time': 1666175400},
        #     6: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666236600},
        #     7: {'date': 20221020, 'start_time': 1666260000, 'end_time': 1666261800},
        #     8: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666323000},
        #     9: {'date': 20221021, 'start_time': 1666346400, 'end_time': 1666348200},
        #     10: {'date': 20221022, 'start_time': 1666407600, 'end_time': 1666409400},
        # } # half an hour
                        
        config_mapping = {
            0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665976200},
            1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666062600},
            2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666149000},
            3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666235400},
            4: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666321800},
        } # 10 min
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665977400},
        #     1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666063800},
        #     2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666150200},
        #     3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666236600},
        #     4: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666323000},
        # } # half an hour
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665982800},
        #     1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666069200},
        #     2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666155600},
        #     3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666242000},
        #     4: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666328400},
        # } # 11:00-13:00

        self.existing_courier_algo = 0
        # 根据 env_index 获取相应的日期和时间范围
        if self.env_index in config_mapping:
            config = config_mapping[self.env_index]
            date_value = config['date']
            self.start_time = config['start_time']
            self.end_time = config['end_time']
            
            # 筛选和排序数据
            df = df[(df['dispatch_time'] > 0) & (df['dt'] == date_value)] # do not define in one area
            df = df.sort_values(by=['platform_order_time'], ascending=True)
            df = df[(df['platform_order_time'] >= self.start_time) & (df['platform_order_time'] < self.end_time)]
            self.order_data = df.reset_index(drop=True)
            
            self.predicted_count = order_num_estimate[order_num_estimate['dt'] == date_value]['predicted_count']
                     
        lat_values = self.order_data[['sender_lat', 'recipient_lat', 'grab_lat']]
        lat_values_non_zero = lat_values[lat_values > 0].dropna()

        self.lat_min = lat_values_non_zero.min().min() / 1e6 # 取所有列的最小值
        self.lat_max = lat_values_non_zero.max().max() / 1e6 # 取所有列的最大值

        lng_values = self.order_data[['sender_lng', 'recipient_lng', 'grab_lng']]
        lng_values_non_zero = lng_values[lng_values > 0].dropna()

        self.lng_min = lng_values_non_zero.min().min() / 1e6 # 取所有列的最小值
        self.lng_max = lng_values_non_zero.max().max() / 1e6 # 取所有列的最大值
        
        order_time = self.order_data[['estimate_arrived_time', 'dispatch_time', 'fetch_time', 'arrive_time', 'estimate_meal_prepare_time', 'order_push_time', 'platform_order_time']]
        order_time_non_zero = order_time[order_time > 0].dropna()

        self.time_min = order_time_non_zero.min().min() / 1e6 # 取所有列的最小值
        self.time_max = order_time_non_zero.max().max() / 1e6 # 取所有列的最大值

        self.grid_size = 20
        
        self.lat_step = (self.lat_max - self.lat_min) / self.grid_size
        self.lng_step = (self.lng_max - self.lng_min) / self.grid_size

        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.interval = 20 # allocation for every 20 seconds

        self.clock = self.start_time + self.interval # self.order_data['platform_order_time'][0]
        
        self.da_frequency = pd.read_csv('MF-PPO Algo/order_prediction/order_da_frequency.csv')
        self.location_estimation_data = pd.read_csv('MF-PPO Algo/order_prediction/noon_peak_hour_data.csv')
        # self.da_frequency = pd.read_csv('/share/home/tj23028/TSL/test/order_prediction/order_da_frequency.csv')
        # self.location_estimation_data = pd.read_csv('/share/home/tj23028/TSL/test/order_prediction/noon_peak_hour_data.csv')
        
        self.max_num_couriers = 1100 # 2250
        for index, dt in self.order_data.iterrows():
            if len(self.couriers) >= self.max_num_couriers:
                break
            
            courier_id = dt['courier_id']
            if courier_id not in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                self.couriers_id.add(courier_id)
                courier_type = 1 if random.random() > 0.7 else 0 # 0.3众包, 0.7专送
                if courier_type == 0:
                    self.num_couriers1 += 1
                else:
                    self.num_couriers2 += 1
                courier_location = (dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6)
                courier = Courier(courier_type, courier_id, courier_location, None)
                courier.state = 'inactive'
                self.couriers.append(courier)
        
        if eval == False:
            self.step(first_time=1)
        else:
            self.eval_step(first_time=1)
    
    def reset(self, env_index, eval=False):
        self.orders = []
        self.couriers = []
        self.__init__(env_index, self.algo_index, eval)

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + '\n'
        for c in self.couriers:
            message += repr(c) + '\n'
        for p in self.orders:
            message += repr(p) + '\n'
        return message                

    def step(self, first_time=0):
        
        # self.add_new_couriers = 0
        
        if not first_time:
            if self.clock < self.end_time:
                self.clock += self.interval 

        orders_failed = [order for order in self.orders if order.status == "wait_pair"]
        orders_new = []

        while(self.current_index < self.order_data.shape[0] and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
            dt = self.order_data.iloc[self.current_index]
            order_id = dt['order_id']
            
            if order_id not in self.orders_id and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0:                
        
                self.orders_id.add(order_id)
                
                order_create_time = dt['platform_order_time']
                pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                meal_prepare_time = dt['estimate_meal_prepare_time']
                estimate_arrived_time = dt['estimate_arrived_time']
                
                order = Order(order_id, dt['da_id'], dt['poi_id'], order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                orders_new.append(order)

            courier_id = dt['courier_id']
            if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                for courier in self.couriers:
                    if courier.courier_id == courier_id and courier.state == 'inactive':
                        courier.state = 'active'
                        courier.start_time = self.clock
                        courier.leisure_time = self.clock
                        self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                        break
            
            self.current_index += 1
            
        # if a courier does not get an order for a period of a time, he will quit the system.
        for courier in self.couriers:
            if courier.is_leisure == 1 and courier.state == 'active':
                courier.total_leisure_time += self.interval
            elif courier.is_leisure == 0 and courier.state == 'active':
                courier.total_running_time += self.interval

            if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                courier.state = 'inactive'
                self.remove_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)

            
            if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                salary_per_interval = 15 / 3600 * self.interval
                courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                self.platform_cost += salary_per_interval

        orders_pair = orders_failed + orders_new
        
        if orders_pair != []:
            
            self.orders += orders_new

            if self.algo_index == 0:
                self._Delay_allocation(orders_pair)
            elif self.algo_index == 1:
                self._Efficiency_allocation(orders_pair)     
            elif self.algo_index == 2:
                self._MaxMin_fairness_allocation(orders_pair)   
            elif self.algo_index == 3:
                self._Greedy_allocation(orders_pair)
            # self.algo_index == 4 is the origin allocation in the dataset  
        
        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)
        
    def eval_step(self, first_time=0):
        if self.algo_index == 4:
            # self.add_new_couriers = 0
            
            if not first_time:
                if self.clock < self.end_time:
                    self.clock += self.interval 
            
            while(self.current_index < self.order_data.shape[0] and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
                dt = self.order_data.iloc[self.current_index]
                order_id = dt['order_id']
                
                if dt['courier_id'] not in self.couriers_id:
                    self.current_index += 1
                    continue
            
                if order_id not in self.orders_id and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0 and dt['is_courier_grabbed'] == 1 and self.existing_courier_algo < self.max_num_couriers:                        
                    order_create_time = dt['platform_order_time']
                    pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                    meal_prepare_time = dt['estimate_meal_prepare_time']
                    estimate_arrived_time = dt['estimate_arrived_time']
                
                    order = Order(order_id, dt['da_id'], dt['poi_id'], order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)

                    courier_id = dt['courier_id']
                    courier = None
                    for candidate in self.couriers:
                        if candidate.courier_id == courier_id:
                            courier = candidate
                            if courier.state == 'inactive':
                                courier.state = 'active'
                                courier.start_time = self.clock
                                courier.leisure_time = self.clock
                                self.existing_courier_algo += 1
                                self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                                break
                        
                        # self.add_new_couriers += 1
                    
                    if len(courier.waybill) + len(courier.wait_to_pick) == courier.capacity or courier == None:
                        self.current_index += 1
                        continue
                    
                    self.orders_id.add(order_id)
                    self.orders.append(order)

                    if courier.courier_type == 0:
                        order.price = self._wage_response_model(order, courier)
                        self.platform_cost += order.price
                    else:
                        order.price = self._wage_response_model(order, courier) * 1.5
                        self.platform_cost += order.price

                    courier.wait_to_pick.append(order)
                    courier.order_sequence, courier.current_wave_dist, courier.current_risk = self._cal_wave_info(None, courier)
                    order.pair_courier = courier
                    order.pair_time = self.clock
                    order.status = 'wait_pick'
            
                    if courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:
                        courier.pick_order(order)

                        if courier.position == order.drop_off_point:
                            courier.drop_order(order)
                                   
                self.current_index += 1
            
            # if a courier does not get an order for a period of a time, he will quit the system.
            for courier in self.couriers:
                if courier.is_leisure == 1 and courier.state == 'active':
                    courier.total_leisure_time += self.interval
                elif courier.is_leisure == 0 and courier.state == 'active':
                    courier.total_running_time += self.interval

                if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                    courier.state = 'inactive'
                
                if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                    salary_per_interval = 15 / 3600 * self.interval
                    courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                    self.platform_cost += salary_per_interval
        
        else:
            # self.add_new_couriers = 0
            
            if not first_time:
                if self.clock < self.end_time:
                    self.clock += self.interval 

            orders_failed = [order for order in self.orders if order.status == "wait_pair"]
            orders_new = []

            while(self.current_index < self.order_data.shape[0] and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
                dt = self.order_data.iloc[self.current_index]
                order_id = dt['order_id']
                
                if dt['courier_id'] not in self.couriers_id:
                    self.current_index += 1
                    continue
                                    
                if order_id not in self.orders_id and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0:                
            
                    self.orders_id.add(order_id)
                    
                    order_create_time = dt['platform_order_time']
                    pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                    meal_prepare_time = dt['estimate_meal_prepare_time']
                    estimate_arrived_time = dt['estimate_arrived_time']
                    
                    order = Order(order_id, dt['da_id'], dt['poi_id'], order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                    orders_new.append(order)

                courier_id = dt['courier_id']                        
                if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                    for courier in self.couriers:
                        if courier.courier_id == courier_id and courier.state == 'inactive':
                            courier.state = 'active'
                            courier.start_time = self.clock
                            courier.leisure_time = self.clock
                            self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                            break

                self.current_index += 1
                
            # if a courier does not get an order for a period of a time, he will quit the system.
            for courier in self.couriers:
                if courier.is_leisure == 1 and courier.state == 'active':
                    courier.total_leisure_time += self.interval
                elif courier.is_leisure == 0 and courier.state == 'active':
                    courier.total_running_time += self.interval

                if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                    courier.state = 'inactive'
                
                if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                    salary_per_interval = 15 / 3600 * self.interval
                    courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                    self.platform_cost += salary_per_interval

            orders_pair = orders_failed + orders_new
            
            if orders_pair != []:
                
                self.orders += orders_new

                if self.algo_index == 0:
                    self._Delay_allocation(orders_pair)
                elif self.algo_index == 1:
                    self._Efficiency_allocation(orders_pair)     
                elif self.algo_index == 2:
                    self._MaxMin_fairness_allocation(orders_pair)   
                elif self.algo_index == 3:
                    self._Greedy_allocation(orders_pair)
                # self.algo_index == 4 is the origin allocation in the dataset  
            
        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)
    
    ##################
    # grid
    def get_grid_index(self, lat, lng):
        lat_index = int((lat - self.lat_min) / self.lat_step)
        lng_index = int((lng - self.lng_min) / self.lng_step)
        
        lat_index = min(max(lat_index, 0), self.grid_size - 1)
        lng_index = min(max(lng_index, 0), self.grid_size - 1)
        
        return lat_index, lng_index
    
    def add_courier(self, lat, lng, courier):
        lat_index, lng_index = self.get_grid_index(lat, lng)
        self.grid[lat_index][lng_index].append(courier)
    
    def get_courier_in_grid(self, lat, lng):
        lat_index, lng_index = self.get_grid_index(lat, lng)
        return self.grid[lat_index][lng_index]
    
    def remove_courier(self, lat, lng, courier):
        lat_index, lng_index = self.get_grid_index(lat, lng)
        if courier in self.grid[lat_index][lng_index]:
            self.grid[lat_index][lng_index].remove(courier)
    
    def update_courier_position(self, old_lat, old_lng, new_lat, new_lng, courier):
        self.remove_courier(old_lat, old_lng, courier)
        self.add_courier(new_lat, new_lng, courier)
        
    def get_subgrid_position(self, lat, lng):    
        lat_in_grid = (lat - self.lat_min) % self.lat_step
        lng_in_grid = (lng - self.lng_min) % self.lng_step
        
        half_lat_step = self.lat_step / 2
        half_lng_step = self.lng_step / 2
        
        # (0, 0)为左下，(1, 0)为右下，(0, 1)为左上，(1, 1)为右上
        subgrid_pos = (lat_in_grid >= half_lat_step, lng_in_grid >= half_lng_step)
        
        return subgrid_pos  
        
    def get_adjacent_grids(self, lat, lng):
        lat_index, lng_index = self.get_grid_index(lat, lng)    
        position_flag = self.get_subgrid_position(lat, lng)
        adjacent_grids = [(lat_index, lng_index)]
        
        if position_flag == (0, 0):
            for di, dj in [(-1, 0), (0, -1), (-1, -1)]:
                ni, nj = lat_index + di, lng_index + dj
                if ni >= 0 and nj >= 0:
                    adjacent_grids.append((ni, nj))  
        elif position_flag == (1, 0): 
            for di, dj in [(1, 0), (0, -1), (1, -1)]:
                ni, nj = lat_index + di, lng_index + dj
                if ni < self.grid_size and nj >= 0:
                    adjacent_grids.append((ni, nj))  
        elif position_flag == (0, 1):
            for di, dj in [(-1, 0), (0, 1), (-1, 1)]:
                ni, nj = lat_index + di, lng_index + dj
                if ni >= 0 and nj < self.grid_size:
                    adjacent_grids.append((ni, nj))  
        else:
            for di, dj in [(1, 0), (0, 1), (1, 1)]:
                ni, nj = lat_index + di, lng_index + dj
                if ni < self.grid_size and nj < self.grid_size:
                    adjacent_grids.append((ni, nj))  
        return adjacent_grids
    
    def get_couriers_in_adjacent_grids(self, lat, lng):
        adjacent_grids = self.get_adjacent_grids(lat, lng)
        couriers = []
        
        for grid in adjacent_grids:
            lat_index, lng_index = grid
            couriers.extend(self.grid[lat_index][lng_index])
        
        return couriers

            
    ##################
    # courier
    def _accept_or_reject(self, order, courier):
        
        decision = True if random.random() < 0.9 else False
        return decision
    
    ##################
    # platform
    def _get_predicted_orders(self):
            
        index = (self.clock - self.start_time) // self.interval - 1
        predicted_count = int(self.predicted_count.iloc[index])

        predicted_orders = []
        da_frequency_row = self.da_frequency[self.da_frequency['time_interval'] == index].iloc[0]
        
        da_ids = da_frequency_row.index[1:]
        frequencies = da_frequency_row.values[1:]

        frequencies_normalized = frequencies / np.sum(frequencies)

        from collections import Counter
        assigned_da_ids = np.random.choice(
            da_ids,
            size=predicted_count,
            p=frequencies_normalized
        )

        
        da_order_count = dict(Counter(assigned_da_ids))

        for da_id, num in da_order_count.items():
            da_id = int(da_id)
            model_data = self.location_estimation_data[(self.location_estimation_data['time_interval'] == index) & (self.location_estimation_data['da_id'] == da_id)].reset_index(drop=True)
            
            mean_eta = int(np.mean(model_data['estimate_arrived_time'] - model_data['platform_order_time']))
            
            from sklearn.cluster import KMeans

            coordinates = model_data[['sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng']].values / 1e6
            if len(coordinates) > num:
                kmeans = KMeans(n_clusters=num, random_state=42, n_init='auto')
                kmeans.fit(coordinates)

                predicted_coords = kmeans.cluster_centers_

                labels = kmeans.labels_

                for label in np.unique(labels):
                    
                    cluster_center = predicted_coords[label]
                    
                    pickup_point = (cluster_center[0], cluster_center[1])
                    dropoff_point = (cluster_center[2], cluster_center[3])
                    
                    eta = mean_eta + self.clock

                    order_create_time = self.clock            
                    order = Order(-1, da_id, -1, order_create_time, pickup_point, dropoff_point, 0, eta)
                    predicted_orders.append(order)
            else:
                for i in range(len(coordinates)):
                    pickup_point = (coordinates[i][0], coordinates[i][1])
                    dropoff_point = (coordinates[i][2], coordinates[i][3])

                    eta = mean_eta + self.clock
                    order_create_time = self.clock
                    order = Order(-1, da_id, -1, order_create_time, pickup_point, dropoff_point, 0, eta)
                    predicted_orders.append(order)
                                
        return predicted_orders
        
    # give order to the nearest guy                        
    def _Efficiency_allocation(self, orders):
        
        for order in orders:
            min_dist = math.inf
            nearest_courier = None
            for courier in self.couriers:
                if courier.state == 'active' and len(courier.waybill) + len(courier.wait_to_pick) < courier.capacity:
                    dist = geodesic(courier.position, order.pick_up_point).meters
                    if min_dist > dist:
                        min_dist = dist
                        nearest_courier = courier
            
            if (self.clock - order.order_create_time > 120) and (nearest_courier.courier_type == 0 and nearest_courier.reject_order_num > 5):
                if nearest_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, nearest_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, nearest_courier) * 1.5
                    self.platform_cost += order.price
                
                nearest_courier.wait_to_pick.append(order)
                nearest_courier.order_sequence, nearest_courier.current_wave_dist, nearest_courier.current_risk = self._cal_wave_info(None, nearest_courier)
                order.pair_courier = nearest_courier
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if nearest_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    nearest_courier.pick_order(order)

                    if nearest_courier.position == order.drop_off_point:  # dropping off
                        nearest_courier.drop_order(order)
            
            elif (self.clock - order.order_create_time <= 120) and ((nearest_courier.courier_type == 1) or (nearest_courier.courier_type == 0 and nearest_courier.reject_order_num <= 5)):
                decision = self._accept_or_reject(order, courier)
                if decision == True:
                    order.price = self._wage_response_model(order, nearest_courier) 
                    self.platform_cost += order.price           
                            
                    nearest_courier.wait_to_pick.append(order)
                    nearest_courier.order_sequence, nearest_courier.current_wave_dist, nearest_courier.current_risk = self._cal_wave_info(None, nearest_courier)
                    order.pair_courier = nearest_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if nearest_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        nearest_courier.pick_order(order)

                        if nearest_courier.position == order.drop_off_point:  # dropping off
                            nearest_courier.drop_order(order)                    
                else:
                    order.reject_count += 1
                    courier.reject_order_num += 1
    
    # give the order to the poorest guy                            
    def _fair_allocation(self, orders):

        for i, order in enumerate(orders):
            min_income = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(order, 1500)
            for courier in nearby_couriers:
                sequence, dist, risk = self._cal_wave_info(order, courier)
                avg_income = courier.income / (self.clock - courier.start_time) if (self.clock - courier.start_time) != 0 else courier.income
                if min_income > avg_income and risk:
                    min_income = avg_income
                    assigned_courier = courier
            
            if assigned_courier is not None:
                if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                    if assigned_courier.courier_type == 0:
                        order.price = self._wage_response_model(order, assigned_courier)
                        self.platform_cost += order.price
                    else:
                        order.price = self._wage_response_model(order, assigned_courier) * 1.5
                        self.platform_cost += order.price
                    
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)
                            
                elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                    decision = self._accept_or_reject(order, courier)
                    if decision == True:
                        order.price = self._wage_response_model(order, assigned_courier)
                        self.platform_cost += order.price
                                            
                        assigned_courier.wait_to_pick.append(order)
                        assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                        order.pair_courier = assigned_courier
                        order.status = 'wait_pick'
                        order.pair_time = self.clock
                        
                        if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                            assigned_courier.pick_order(order)

                            if assigned_courier.position == order.drop_off_point:  # dropping off
                                assigned_courier.drop_order(order)                    
                    else:
                        order.reject_count += 1
                        courier.reject_order_num += 1
    
    # consider the matching degree but match one by one           
    def _Greedy_allocation(self, orders):
        for i, order in enumerate(orders):
            min_cost = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(order, 1500)
            for courier in nearby_couriers:
                order_sequence, distance, risk = self._cal_wave_info(order, courier)
                detour = distance - courier.current_wave_dist

                price = self._wage_response_model(order, courier)
                
                cost = detour / price
                
                if min_cost > cost and not risk:
                    min_cost = cost
                    assigned_courier = courier
            
            if assigned_courier is not None:
                if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                    if assigned_courier.courier_type == 0:
                        order.price = self._wage_response_model(order, assigned_courier)
                        self.platform_cost += order.price
                    else:
                        order.price = self._wage_response_model(order, assigned_courier) * 1.5
                        self.platform_cost += order.price
                    
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)
                            
                elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                    decision = self._accept_or_reject(order, courier)
                    if decision == True:
                        order.price = self._wage_response_model(order, assigned_courier)      
                        self.platform_cost += order.price
                                      
                        assigned_courier.wait_to_pick.append(order)
                        assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                        order.pair_courier = assigned_courier
                        order.status = 'wait_pick'
                        order.pair_time = self.clock
                        
                        if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                            assigned_courier.pick_order(order)

                            if assigned_courier.position == order.drop_off_point:  # dropping off
                                assigned_courier.drop_order(order)                    
                    else:
                        order.reject_count += 1
                        courier.reject_order_num += 1
    
    def _is_bipartite_solvable(self, matrix):
        has_isolated_row = np.any(np.all(np.isinf(matrix), axis=1))
        has_isolated_col = np.any(np.all(np.isinf(matrix), axis=0))

        if has_isolated_row or has_isolated_col:
            return False
        
        finite_matrix = np.where(np.isinf(matrix), 1e9, matrix)
        rank = np.linalg.matrix_rank(finite_matrix)
        rows, cols = matrix.shape
        
        if rank < min(rows, cols):
            return False
        
        return True
    
    # Bipartitie matching with estimation
    def _Delay_allocation(self, orders):
        
        predicted_orders = self._get_predicted_orders()
        all_orders = orders + predicted_orders
        
        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        for order in orders:
            nearby_couriers = self._get_nearby_couriers(order)
            couriers.update(nearby_couriers)
        
        couriers = list(couriers)
        
        M = 1e9
        min_cost = 0
        for order in all_orders:
            row = []
            for courier in couriers:
                sequence, dist, risk = self._cal_wave_info(order, courier)
                if not risk:
                    price = self._wage_response_model(order, courier)
                    detour = dist - courier.current_wave_dist
                    
                    cost =  detour / price
                    if cost < min_cost:
                        min_cost = cost
                    row.append(cost)
                else:
                    row.append(float(M))  # Set an infinite cost if the assignment is unreasonable
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix)
        cost_matrix += abs(min_cost)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)        

        # Assign orders to couriers based on the optimal matching
        for order_index, courier_index in zip(row_ind, col_ind):
            
            order = all_orders[order_index]
            if order in predicted_orders:
                continue

            if cost_matrix[order_index][courier_index] == float(M):
                order.reject_count += 1
                continue  # Skip infeasible matches
            
            assigned_courier = couriers[courier_index]
                        
            if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                if assigned_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, assigned_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, assigned_courier) * 1.5
                    self.platform_cost += order.price
                
                assigned_courier.wait_to_pick.append(order)
                assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                order.pair_courier = assigned_courier
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    assigned_courier.pick_order(order)

                    if assigned_courier.position == order.drop_off_point:  # dropping off
                        assigned_courier.drop_order(order)
                        
            elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                decision = self._accept_or_reject(order, courier)
                if decision == True:
                    order.price = self._wage_response_model(order, assigned_courier)     
                    self.platform_cost += order.price               
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)                    
                else:
                    order.reject_count += 1
                    courier.reject_order_num += 1
    
    # matching with a fairness threshold: first match, then set a threshold and first match the "valuable" courier without cost exceeding the threshold
    def _fairness_threshold_allocation(self, orders):
        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        predicted_orders = self._get_predicted_orders()
        all_orders = orders + predicted_orders

        for order in orders:
            nearby_couriers = self._get_nearby_couriers(order)
            couriers.update(nearby_couriers)
        
        couriers = list(couriers)
        
        M = 1e9
        min_cost = 0
        for order in all_orders:
            row = []
            for courier in couriers:
                sequence, dist, risk = self._cal_wave_info(order, courier)
                if not risk:
                    price = self._wage_response_model(order, courier)
                    detour = dist - courier.current_wave_dist
                    
                    cost =  detour / price
                    if cost < min_cost:
                        min_cost = cost
                    row.append(cost)
                else:
                    row.append(float(M))  # Set an infinite cost if the assignment is unreasonable
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix)
        cost_matrix += abs(min_cost)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)   
        
        fairness_factor = 1.5
        filtered_costs = cost_matrix[row_ind, col_ind]
        filtered_costs = filtered_costs[filtered_costs != M]
        initial_max_cost = np.max(filtered_costs) if len(filtered_costs) > 0 else M
        
        matched_orders = set()
        courier_avg_costs = []
        
        if initial_max_cost != M: 
            row_ind = []
            col_ind = []
         
            threshold = fairness_factor * initial_max_cost  
        
            for col in range(cost_matrix.shape[1]):
                courier_costs = cost_matrix[:, col]
                courier_costs = courier_costs[courier_costs != M]
                if len(courier_costs) > 0:
                    courier_avg_cost = np.mean(courier_costs)
                    courier_avg_costs.append((couriers[col], courier_avg_cost))

            sorted_courier_list = sorted(courier_avg_costs, key=lambda x: x[1], reverse=True)
            sorted_couriers = [courier for courier, _ in sorted_courier_list]

            for courier in sorted_couriers:
                courier_index = couriers.index(courier)
                courier_costs = cost_matrix[:, courier_index]
                valid_costs = courier_costs[~np.isin(np.arange(cost_matrix.shape[0]), list(matched_orders))]
                
                if len(valid_costs) == 0:
                    continue
                
                min_cost = np.min(valid_costs)
                best_order_index_in_valid = np.argmin(valid_costs)
                best_order_index = np.where(courier_costs == valid_costs[best_order_index_in_valid])[0][0]
                
                if min_cost < threshold: 
                    matched_orders.add(best_order_index)
                    row_ind.append(best_order_index)
                    col_ind.append(courier_index)
                    
        # Assign orders to couriers based on the optimal matching
        for order_index, courier_index in zip(row_ind, col_ind):
            if cost_matrix[order_index][courier_index] == float(M):
                order.reject_count += 1
                continue  # Skip infeasible matches
            order = all_orders[order_index]
            assigned_courier = couriers[courier_index]
                        
            if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                if assigned_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, assigned_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, assigned_courier) * 1.5
                    self.platform_cost += order.price
                
                assigned_courier.wait_to_pick.append(order)
                assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                order.pair_courier = assigned_courier
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    assigned_courier.pick_order(order)

                    if assigned_courier.position == order.drop_off_point:  # dropping off
                        assigned_courier.drop_order(order)
                        
            elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                decision = self._accept_or_reject(order, courier)
                if decision == True:
                    order.price = self._wage_response_model(order, assigned_courier)     
                    self.platform_cost += order.price               
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)                    
                else:
                    order.reject_count += 1
                    courier.reject_order_num += 1

    # Maxmin the cost
    def _MaxMin_fairness_allocation(self, orders):
        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        for order in orders:
            nearby_couriers = self._get_nearby_couriers(order)
            couriers.update(nearby_couriers)
        
        couriers = list(couriers)
        
        M = 1e9
        min_cost = 0
        for order in orders:
            row = []
            for courier in couriers:
                sequence, dist, risk = self._cal_wave_info(order, courier)
                if not risk:
                    price = self._wage_response_model(order, courier)
                    detour = dist - courier.current_wave_dist
                    
                    cost =  detour / price
                    if cost < min_cost:
                        min_cost = cost
                    row.append(cost)
                else:
                    row.append(float('inf'))  # Set an infinite cost if the assignment is unreasonable
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix)
        if self._is_bipartite_solvable(cost_matrix):
            cost_matrix += abs(min_cost)
            filtered_elements = cost_matrix[~np.isinf(cost_matrix)]
            sorted_elements = np.sort(filtered_elements)[::-1]
            sorted_list = sorted_elements.tolist()
            for element in sorted_list:
                cost_matrix_copy = cost_matrix.copy()
                cost_matrix_copy[cost_matrix_copy >= element] = np.inf
                if self._is_bipartite_solvable(cost_matrix_copy):
                    continue
                else:
                    cost_matrix[cost_matrix > element] = M
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    break
        else:
            cost_matrix[cost_matrix == np.inf] = M
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
        for order_index, courier_index in zip(row_ind, col_ind):
            if cost_matrix[order_index][courier_index] == float(M):
                order.reject_count += 1
                continue  # Skip infeasible matches
            
            order = orders[order_index]
            assigned_courier = couriers[courier_index]
                        
            if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                if assigned_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, assigned_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, assigned_courier) * 1.5
                    self.platform_cost += order.price
                
                assigned_courier.wait_to_pick.append(order)
                assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                order.pair_courier = assigned_courier
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    assigned_courier.pick_order(order)

                    if assigned_courier.position == order.drop_off_point:  # dropping off
                        assigned_courier.drop_order(order)
                        
            elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                decision = self._accept_or_reject(order, courier)
                if decision == True:
                    order.price = self._wage_response_model(order, assigned_courier)     
                    self.platform_cost += order.price               
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)                    
                else:
                    order.reject_count += 1
                    courier.reject_order_num += 1
                    
    # minimize the bound between the worst and the best
    def _Pairwise_fairness_allocation(self, orders):
        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        for order in orders:
            nearby_couriers = self._get_nearby_couriers(order)
            couriers.update(nearby_couriers)
        
        couriers = list(couriers)
        
        M = 1e9
        min_cost = 0
        for order in orders:
            row = []
            for courier in couriers:
                sequence, dist, risk = self._cal_wave_info(order, courier)
                if not risk:
                    price = self._wage_response_model(order, courier)
                    detour = dist - courier.current_wave_dist
                    
                    cost =  detour / price
                    if cost < min_cost:
                        min_cost = cost
                    row.append(cost)
                else:
                    row.append(float('inf'))  # Set an infinite cost if the assignment is unreasonable
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix)
        if not self._is_bipartite_solvable(cost_matrix):
            cost_matrix[cost_matrix == np.inf] = M
            row_ind, col_ind = linear_sum_assignment(cost_matrix)   
        else:            
            cost_matrix += abs(min_cost)
            filtered_elements = cost_matrix[~np.isinf(cost_matrix)]
            sorted_elements = np.sort(filtered_elements)
            sorted_list = sorted_elements.tolist()
            
            previous_lower_bound = sorted_list[0]
            previous_upper_bound = sorted_list[-1]
            origin_list = sorted_list.copy()
            
            while len(sorted_list) > 1:
                first_diff = sorted_list[1] - sorted_list[0]
                last_diff = sorted_list[-1] - sorted_list[-2]
                if first_diff <= last_diff:
                    cost_matrix_copy_first = cost_matrix.copy()
                    cost_matrix_copy_first[cost_matrix_copy_first <= sorted_list[0]] = np.inf
                    if not self._is_bipartite_solvable(cost_matrix_copy_first):
                        cost_matrix_copy_last = cost_matrix.copy()
                        cost_matrix_copy_last[cost_matrix_copy_last >= sorted_list[-1]] = np.inf
                        if not self._is_bipartite_solvable(cost_matrix_copy_last):
                            break
                        else:
                            previous_upper_bound = sorted_list[-1]
                            sorted_list = sorted_list[:-1]
                    else:
                        sorted_list = sorted_list[1:]
                else:
                    cost_matrix_copy_last = cost_matrix.copy()
                    cost_matrix_copy_last[cost_matrix_copy_last >= sorted_list[-1]] = np.inf
                    if not self._is_bipartite_solvable(cost_matrix_copy_last):
                        cost_matrix_copy_first = cost_matrix.copy()
                        cost_matrix_copy_first[cost_matrix_copy_first <= sorted_list[0]] = np.inf
                        if not self._is_bipartite_solvable(cost_matrix_copy_first):
                            break
                        else:
                            previous_lower_bound = sorted_list[0]
                            sorted_list = sorted_list[1:]
                    else:
                        sorted_list = sorted_list[:-1]
            if previous_lower_bound != origin_list[0] or previous_upper_bound != origin_list[-1]:            
                cost_matrix[(cost_matrix <= previous_lower_bound) | (cost_matrix >= previous_upper_bound)] = np.inf
            
            cost_matrix[cost_matrix == np.inf] = M
            row_ind, col_ind = linear_sum_assignment(cost_matrix)   

        # Assign orders to couriers based on the optimal matching
        for order_index, courier_index in zip(row_ind, col_ind):
            if cost_matrix[order_index][courier_index] == float(M):
                order.reject_count += 1
                continue  # Skip infeasible matches
            
            order = orders[order_index]
            assigned_courier = couriers[courier_index]
                        
            if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                if assigned_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, assigned_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, assigned_courier) * 1.5
                    self.platform_cost += order.price
                
                assigned_courier.wait_to_pick.append(order)
                assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                order.pair_courier = assigned_courier
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    assigned_courier.pick_order(order)

                    if assigned_courier.position == order.drop_off_point:  # dropping off
                        assigned_courier.drop_order(order)
                        
            elif (self.clock - order.order_create_time <= 120) and ((assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num <= 5)):
                decision = self._accept_or_reject(order, courier)
                if decision == True:
                    order.price = self._wage_response_model(order, assigned_courier)     
                    self.platform_cost += order.price               
                    assigned_courier.wait_to_pick.append(order)
                    assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self._cal_wave_info(None, assigned_courier)
                    order.pair_courier = assigned_courier
                    order.status = 'wait_pick'
                    order.pair_time = self.clock
                    
                    if assigned_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                        assigned_courier.pick_order(order)

                        if assigned_courier.position == order.drop_off_point:  # dropping off
                            assigned_courier.drop_order(order)                    
                else:
                    order.reject_count += 1
                    courier.reject_order_num += 1
                        
    def _get_nearby_couriers(self, order, dist_range=1500):
        nearby_couriers = []

        min_dist = math.inf
        nearest_courier = None
        
        couriers = self.get_couriers_in_adjacent_grids(order.pick_up_point[0], order.pick_up_point[1])
        
        for courier in couriers:
            if courier.state == 'active' and len(courier.waybill) + len(courier.wait_to_pick) < courier.capacity:
                dist = geodesic(courier.position, order.pick_up_point).meters
                if min_dist > dist:
                    min_dist = dist
                    nearest_courier = courier
                if dist < dist_range: # paper from Hai Wang, but the average distance in the data is 608m
                    nearby_couriers.append(courier)
                
        if nearby_couriers == []:
            nearby_couriers.append(nearest_courier)

        return nearby_couriers

    def _cal_wave_info(self, order, courier):
        order_sequence = self._cal_sequence(order, courier)
        last_loc = courier.position
        total_dist = 0
        
        risk = 0
        
        for order in order_sequence:
            total_dist += geodesic(last_loc, order[0]).meters
            last_loc = order[0]
            
            if order[1] == 'dropped':
                speed = total_dist / (order[2] - self.clock)
                if speed > 4:
                    risk = 1
                else:
                    risk = 0

        return order_sequence, total_dist, risk
    
    def _cal_sequence(self, order, courier):
        #################
        # first stage
        orders = (courier.waybill + courier.wait_to_pick).copy()
        if order is not None:
            orders.append(order)
            
        points = []
        waybill_points = []

        for o in orders:
            if o in courier.waybill:
                points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
                waybill_points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
            else:
                points.append((o.pick_up_point, 'pick_up', o.ETA, o.orderid))
                points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
            
        # ETA reveals the sequence of the appearance of orders on the platform
        orders = sorted(orders, key=lambda o: o.ETA)
        order_sequence = []
        if orders[0] in courier.waybill:
            order_sequence.append((orders[0].drop_off_point, 'dropped', orders[0].ETA, orders[0].orderid))
            points.remove((orders[0].drop_off_point, 'dropped', orders[0].ETA, orders[0].orderid))
        else:
            order_sequence.append((orders[0].pick_up_point, 'pick_up', o.ETA, orders[0].orderid))
            points.remove((orders[0].pick_up_point, 'pick_up', orders[0].ETA, orders[0].orderid))
            
        while points:
            last_point = order_sequence[-1][0]
            closest_points = []

            for point in points:
                distance = geodesic(last_point, point[0]).meters
                closest_points.append((distance, point))
                
            closest_points.sort(key=lambda x: (x[0], x[1]))
            flag = True
            for dist, closest_point in closest_points:
                if flag == False:
                    break
                
                if closest_point[1] == 'dropped' and closest_point not in waybill_points:
                    for seq_point in order_sequence:
                        if seq_point[3] == closest_point[3] and seq_point[1] == 'pick_up':
                            order_sequence.append(closest_point)
                            points.remove(closest_point)
                            flag = False
                            break
                else:
                    order_sequence.append(closest_point)
                    points.remove(closest_point)
                    break        
                
        #################
        # second stage
        while(True):
            dist = 0
            last_location = courier.position
            i = 0
            while i < len(order_sequence):
                point = order_sequence[i]
                if point[1] == 'dropped':
                    temp_dist = dist + geodesic(last_location, point[0]).meters
                    if 4 * (point[2] - self.clock) < temp_dist and i != 0 and order_sequence[i-1][3] != point[3]:
                        order_sequence[i], order_sequence[i-1] = order_sequence[i-1], order_sequence[i]
                        break
                    else:
                        dist = temp_dist
                        last_location = point[0]
                        i += 1
                else:
                    dist += geodesic(last_location, point[0]).meters
                    last_location = point[0]
                    i += 1
            break
        
        return order_sequence
        
    def _wage_response_model(self, order, courier):
        courier_total_time = self.clock - courier.start_time
        if courier_total_time == 0 or (courier.income == 0 and len(courier.waybill) + len(courier.wait_to_pick) == 0):
            return 10 # as a incentive for a new courier, also 10 is the average price of an order
        elif courier.income == 0 and courier.courier_type == 1 and len(courier.waybill) + len(courier.wait_to_pick) > 0:
            wm = 15 / 3600
            v = 3 # 3 m/s
            r = geodesic(order.pick_up_point, courier.position).meters
            d = geodesic(order.pick_up_point, order.drop_off_point).meters
            wage = wm * (r + d) / v
            if len(courier.waybill) + len(courier.wait_to_pick) == 0:
                wage = wage * 1.5
            elif len(courier.waybill) + len(courier.wait_to_pick) > 0 and len(courier.waybill) + len(courier.wait_to_pick) <= 3:
                wage = wage * 1
            else:
                wage = wage * 0.6
            return 1.5 * wage
        else:
            wm = courier.income / courier_total_time
            v = 3 # 3 m/s
            r = geodesic(order.pick_up_point, courier.position).meters
            d = geodesic(order.pick_up_point, order.drop_off_point).meters
            wage = wm * (r + d) / v
            
            if len(courier.waybill) + len(courier.wait_to_pick) == 0:
                wage = wage * 1.5
            elif len(courier.waybill) + len(courier.wait_to_pick) > 0 and len(courier.waybill) + len(courier.wait_to_pick) <= 3:
                wage = wage * 1
            else:
                wage = wage * 0.6
                
            if courier.courier_type == 0:
                return wage
            else:
                return 1.5 * wage