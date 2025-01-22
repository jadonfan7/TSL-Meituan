import math
from geopy.distance import geodesic
from agent.courier import Courier
from agent.order import Order
import pandas as pd
from utils.gorubi_solver import gorubi_solver
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
        self.env_index = env_index
        self.algo_index = algo_index
        self.current_index = 0
    
        self.platform_cost = 0
        
        df = pd.read_csv('../all_waybill_info_meituan_0322.csv')
        # df = pd.read_csv('all_waybill_info_meituan_0322.csv')
        
        order_estimate_30min = pd.read_csv('/Users/jadonfan/Documents/TSL/data exploration/predictions/30min_result.csv')
        # order_estimate_30min = pd.read_csv('/share/home/tj23028/TSL/PPO_based/predictions/30min_result.csv')

        
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
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665975900},
        #     1: {'date': 20221017, 'start_time': 1666000800, 'end_time': 1666001100},
        #     2: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666062300},
        #     3: {'date': 20221018, 'start_time': 1666087200, 'end_time': 1666087500},
        #     4: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666148700},
        #     5: {'date': 20221019, 'start_time': 1666173600, 'end_time': 1666173900},
        #     6: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666235100},
        #     7: {'date': 20221020, 'start_time': 1666260000, 'end_time': 1666260300},
        #     8: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666321500},
        #     9: {'date': 20221021, 'start_time': 1666346400, 'end_time': 1666346700},
        #     10: {'date': 20221022, 'start_time': 1666407600, 'end_time': 1666407900},
        # } # 5 min
                
        config_mapping = {
            0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665976200},
            1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666062600},
            2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666149000},
            3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666235400},
            4: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666235400},
            5: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666321800},
        } # 10 min
        
        # config_mapping = {
        #     0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665977400},
        #     1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666063800},
        #     2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666150200},
        #     3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666236600},
        #     4: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666323000},
        # } # half an hour
        
        self.max_num_couriers = 2200
        self.existing_courier_algo = 0
        # 根据 env_index 获取相应的日期和时间范围
        if self.env_index in config_mapping:
            config = config_mapping[self.env_index]
            date_value = config['date']
            self.start_time = config['start_time']
            self.end_time = config['end_time']
            
            # 筛选和排序数据
            df = df[(df['dispatch_time'] > 0) & (df['dt'] == date_value)]
            df = df.sort_values(by=['platform_order_time'], ascending=True)
            df = df[(df['platform_order_time'] >= self.start_time) & (df['platform_order_time'] < self.end_time)]
            self.order_data = df.reset_index(drop=True)
            
            self.predicted_count = order_estimate_30min[order_estimate_30min['dt'] == date_value]['predicted_count']
         
        for index, dt in self.order_data.iterrows():
            if len(self.couriers) > self.max_num_couriers:
                break
            
            courier_id = dt['courier_id']
            if courier_id not in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                self.couriers_id.add(courier_id)
                courier_type = 1 if random.random() > 0.7 else 0 # 0.3众包, 0.7专送
                courier_location = (dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6)
                courier = Courier(courier_type, courier_id, courier_location)
                courier.state = 'inactive'
                self.couriers.append(courier)
            
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

        self.interval = 20 # allocation for every 20 seconds

        self.clock = self.start_time + self.interval # self.order_data['platform_order_time'][0]

        # self.add_new_couriers = 0
        # self.scaler = joblib.load('/share/home/tj23028/TSL/PPO_based/envs/courier behavior model/scaler.pkl')
        # self.best_logreg = joblib.load('/share/home/tj23028/TSL/PPO_based/envs/courier behavior model/logistic_regression_model.joblib')
        # self.scaler = joblib.load('/Users/jadonfan/Documents/TSL/courier_accept_reject_behavior/scaler.pkl')
        # self.best_logreg = joblib.load('/Users/jadonfan/Documents/TSL/courier_accept_reject_behavior/logistic_regression_model.joblib')
        
        self.poi_frequency = pd.read_csv('/Users/jadonfan/Documents/TSL/data exploration/predictions/poi_frequency.csv')
        # self.poi_frequency = pd.read_csv('/share/home/tj23028/TSL/PPO_based/predictions/poi_frequency.csv')
        if eval == False:
            self.step(first_time=1)
        else:
            self.eval_step(agent_num=math.inf, first_time=1)
    
    def reset(self, env_index, eval=False):
        self.orders = []
        self.couriers = []
        self.__init__(env_index, self.algo_index, eval)

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + ', size:' + str(self.size) + '\n'
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
                
                is_in_the_same_da_and_poi = 1 if dt['da_id'] == dt['poi_id'] else 0
                order_create_time = dt['platform_order_time']
                pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                meal_prepare_time = dt['estimate_meal_prepare_time']
                estimate_arrived_time = dt['estimate_arrived_time']
                
                order = Order(order_id, is_in_the_same_da_and_poi, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                orders_new.append(order)

            courier_id = dt['courier_id']
            if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                for courier in self.couriers:
                    if courier.courierid == courier_id and courier.state == 'inactive':
                        courier.state = 'active'
                        courier.start_time = self.clock
                        courier.leisure_time = self.clock
                        break
            
            self.current_index += 1
            
        # if a courier does not get an order for a period of a time, he will quit the system.
        for courier in self.couriers:
            if courier.is_leisure == 1 and courier.state == 'active':
                courier.total_leisure_time += self.interval
            elif courier.is_leisure == 0 and courier.state == 'active':
                courier.total_running_time += self.interval

            if courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                courier.state = 'inactive'
            
            if courier.start_time != self.clock and courier.courier_type == 0:
                salary_per_interval = 15 / 3600 * self.interval
                courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                self.platform_cost += salary_per_interval

        orders_pair = orders_failed + orders_new
        
        if orders_pair != []:
            
            self.orders += orders_new

            if self.algo_index == 0:
                self._EEtradeoff_bipartite_allocation(orders_pair)
            # else:
            #     nearby_couriers = None
            #     for i, p in enumerate(orders):
            #         nearby_couriers = self._get_nearby_couriers(p, 1500)
            #     gorubi_solver(nearby_couriers, orders, self.clock)
            elif self.algo_index == 1:
                self._Efficiency_allocation(orders_pair)     
            elif self.algo_index == 2:
                self._MaxMin_fairness_allocation(orders_pair)   
            elif self.algo_index == 3:
                self._EEtradeoff_greedy_allocation(orders_pair)
            # self.algo_index == 4 is the origin allocation in the dataset  
        
        # if orders_pair != []:
            
        #     self.orders += orders_new

        #     if self.algo_index == 0:
        #         self._fairness_threshold_allocation(orders_pair)
        #     elif self.algo_index == 1:
        #         self._MaxMin_fairness_allocation(orders_pair)     
        #     elif self.algo_index == 2:
        #         self._Pairwise_fairness_allocation(orders_pair)   
        #     elif self.algo_index == 3:
        #         self._Efficiency_allocation(orders_pair)     
        #     elif self.algo_index == 4:
        #         self._fair_allocation(orders_pair)               
        
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
            
                if order_id not in self.orders_id and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0 and dt['is_courier_grabbed'] == 1 and self.existing_courier_algo <= self.max_num_couriers:                        
                    is_in_the_same_da_and_poi = 1 if dt['da_id'] == dt['poi_id'] else 0
                    order_create_time = dt['platform_order_time']
                    pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                    meal_prepare_time = dt['estimate_meal_prepare_time']
                    estimate_arrived_time = dt['estimate_arrived_time']
                
                    order = Order(order_id, is_in_the_same_da_and_poi, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)

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

                if courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                    courier.state = 'inactive'
                
                if courier.start_time != self.clock and courier.courier_type == 0:
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
                    
                    is_in_the_same_da_and_poi = 1 if dt['da_id'] == dt['poi_id'] else 0
                    order_create_time = dt['platform_order_time']
                    pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                    meal_prepare_time = dt['estimate_meal_prepare_time']
                    estimate_arrived_time = dt['estimate_arrived_time']
                    
                    order = Order(order_id, is_in_the_same_da_and_poi, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                    orders_new.append(order)

                courier_id = dt['courier_id']                        
                if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                    for courier in self.couriers:
                        if courier.courier_id == courier_id and courier.state == 'inactive':
                            courier.state = 'active'
                            courier.start_time = self.clock
                            courier.leisure_time = self.clock
                            break

                self.current_index += 1
                
            # if a courier does not get an order for a period of a time, he will quit the system.
            for courier in self.couriers:
                if courier.is_leisure == 1 and courier.state == 'active':
                    courier.total_leisure_time += self.interval
                elif courier.is_leisure == 0 and courier.state == 'active':
                    courier.total_running_time += self.interval

                if courier.is_leisure == 1 and self.clock - courier.leisure_time > 300: # 5 minutes
                    courier.state = 'inactive'
                
                if courier.start_time != self.clock and courier.courier_type == 0:
                    salary_per_interval = 15 / 3600 * self.interval
                    courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                    self.platform_cost += salary_per_interval

            orders_pair = orders_failed + orders_new
            
            if orders_pair != []:
                
                self.orders += orders_new

                if self.algo_index == 0:
                    self._EEtradeoff_bipartite_allocation(orders_pair)
                elif self.algo_index == 1:
                    self._Efficiency_allocation(orders_pair)     
                elif self.algo_index == 2:
                    self._MaxMin_fairness_allocation(orders_pair)   
                elif self.algo_index == 3:
                    self._EEtradeoff_greedy_allocation(orders_pair)
                # self.algo_index == 4 is the origin allocation in the dataset  
            
        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)

        
    def _accept_or_reject(self, order, courier):
        
        decision = True if random.random() < 0.9 else False
        return decision
        
        # _, _, max_speed = self._cal_speed(order, courier)
        # # reward = courier.speed - avg_speed_fair
        # # fairness = abs(avg_speed_fair - avg_speed)
        
        # num_waybill = len(courier.waybill + courier.wait_to_pick)
        # potential_overspeed_risk = 1 if max_speed > 4 else 0
        # rejection_history_count = order.reject_count
        # is_in_the_same_da_and_poi = order.is_in_the_same_da_and_poi
        # pick_up_distance = geodesic(courier.position, order.pick_up_point).meters
        # drop_off_distance = geodesic(order.pick_up_point, order.drop_off_point).meters
        # estimate_arrived_time = order.ETA - self.clock if order.ETA - self.clock > 0 else 0
        # estimate_meal_prepare_time = order.meal_prepare_time - self.clock if order.meal_prepare_time - self.clock > 0 else 0
        # order_push_time = self.clock - order.order_create_time

        # # feature_names = [
        # #     'reward', 'fairness', 'num_waybill', 'potential_overspeed_risk', 
        # #     'rejection history count', 'is_in_the_same_da_and_poi', 
        # #     'pick_up_distance', 'drop_off_distance', 'estimate_arrived_time', 
        # #     'estimate_meal_prepare_time', 'order_push_time'
        # # ]
        # feature_names = [
        #     'num_waybill', 'potential_overspeed_risk', 
        #     'rejection history count', 'is_in_the_same_da_and_poi', 
        #     'pick_up_distance', 'drop_off_distance', 'estimate_arrived_time', 
        #     'estimate_meal_prepare_time', 'order_push_time'
        # ]
        # # X = (reward, fairness, num_waybill, potential_overspeed_risk, rejection_history_count, is_in_the_same_da_and_poi, pick_up_distance, drop_off_distance, estimate_arrived_time, estimate_meal_prepare_time, order_push_time)
        # X = (num_waybill, potential_overspeed_risk, rejection_history_count, is_in_the_same_da_and_poi, pick_up_distance, drop_off_distance, estimate_arrived_time, estimate_meal_prepare_time, order_push_time)
        
        # X_df = pd.DataFrame([X], columns=feature_names)
        
        # # Load the scaler
        # # columns_to_standardize = ['reward', 'fairness', 'num_waybill', 'rejection history count', 'pick_up_distance', 'drop_off_distance', 'estimate_arrived_time', 'estimate_meal_prepare_time', 'order_push_time']
        # columns_to_standardize = ['num_waybill', 'rejection history count', 'pick_up_distance', 'drop_off_distance', 'estimate_arrived_time', 'estimate_meal_prepare_time', 'order_push_time']
        # # Standardize new data
        # X_df[columns_to_standardize] = self.scaler.transform(X_df[columns_to_standardize])

        # y_pred_new = self.best_logreg.predict(X_df)
        
        # # if y_pred_new[0] == False:
        # #     order.reject_count += 1
        # #     courier.reject_order_num += 1

        # return y_pred_new[0]
        
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
                                
    def _fair_allocation(self, orders):
        speed_upper_bound = 4

        for i, order in enumerate(orders):
            min_income = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(order, 1500)
            for courier in nearby_couriers:
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                avg_income = courier.income / (self.clock - courier.start_time) if (self.clock - courier.start_time) != 0 else courier.income
                if min_income > avg_income and max_speed < speed_upper_bound:
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
                                 
                # if (assigned_courier.courier_type == 1) or (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num < 10):
                #     decision = self._accept_or_reject(p, assigned_courier)
                #     if decision == True:
                #         p.price = self._wage_response_model(p, assigned_courier)
                #         # courier.income += p.price
                        
                #         assigned_courier.wait_to_pick.append(p)
                #         p.pair_courier = assigned_courier
                #         p.status = 'wait_pick'
                #         p.pair_time = self.clock
                        
                #         if assigned_courier.position == p.pick_up_point and self.clock >= p.meal_prepare_time:  # picking up
                #             assigned_courier.pick_order(p)

                #             if assigned_courier.position == p.drop_off_point:  # dropping off
                #                 assigned_courier.drop_order(p)
                #     else:
                #         p.reject_count += 1
                #         assigned_courier.reject_order_num += 1
                                
                # else:
                #     p.price = self._wage_response_model(p, assigned_courier)
                #     # courier.income += p.price
                                        
                #     assigned_courier.wait_to_pick.append(p)
                #     p.pair_courier = assigned_courier
                #     p.status = 'wait_pick'
                #     p.pair_time = self.clock
                    
                #     if assigned_courier.position == p.pick_up_point and self.clock >= p.meal_prepare_time:  # picking up
                #         assigned_courier.pick_order(p)

                #         if assigned_courier.position == p.drop_off_point:  # dropping off
                #             assigned_courier.drop_order(p)
                
    def _EEtradeoff_greedy_allocation(self, orders):
        speed_upper_bound = 4

        for i, order in enumerate(orders):
            min_cost = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(order, 1500)
            for courier in nearby_couriers:
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if len(courier.waybill) + len(courier.wait_to_pick) > 0:
                    formal_speed_fair, formal_speed, formal_max_speed = self._cal_speed(None, courier)  
                else:
                    formal_speed_fair = 0

                price = self._wage_response_model(order, courier)
                
                cost = (avg_speed_fair - formal_speed_fair) / price
                
                if min_cost > cost and max_speed < speed_upper_bound:
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
    
    def _EEtradeoff_bipartite_allocation(self, orders):
        
        def get_predicted_orders():
            
            index = (self.clock - self.start_time) // self.interval - 1
            predicted_count = int(self.predicted_count.iloc[index])

            predicted_orders = []

            assigned_poi_ids = np.random.choice(
                self.poi_frequency['poi_id'],
                size=predicted_count,
                p=self.poi_frequency['frequency_ratio']
            )
            order_id_index = 0
            for poi_id in assigned_poi_ids:
                data = self.poi_frequency[self.poi_frequency['poi_id'] == poi_id]
                
                eta = data['avg_delivery_time'].values[0] + self.clock

                order_create_time = self.clock
                pickup_point = (data['sender_lat'].values[0] / 1e6, data['sender_lng'].values[0] / 1e6)
                dropoff_point = (data['recipient_lat'].values[0] / 1e6, data['recipient_lng'].values[0] / 1e6)
        
                order = Order(order_id_index, 0, order_create_time, pickup_point, dropoff_point, 0, eta)
                predicted_orders.append(order)
                
                order_id_index += 1

            return predicted_orders

        speed_upper_bound = 4
        
        predicted_orders = get_predicted_orders()
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
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if max_speed < speed_upper_bound:
                    price = self._wage_response_model(order, courier)
                    if len(courier.waybill) + len(courier.wait_to_pick) > 0:
                        formal_speed_fair, formal_speed, formal_max_speed = self._cal_speed(None, courier)  
                    else:
                        formal_speed_fair = 0
                    speed_variation = avg_speed_fair - formal_speed_fair
                    cost =  speed_variation / price
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
            if cost_matrix[order_index][courier_index] == float(M):
                order.reject_count += 1
                continue  # Skip infeasible matches
            
            order = all_orders[order_index]
            assigned_courier = couriers[courier_index]
            
            if order in predicted_orders:
                continue
            
            if (self.clock - order.order_create_time > 120) and (assigned_courier.courier_type == 0 and assigned_courier.reject_order_num > 5):
                if assigned_courier.courier_type == 0:
                    order.price = self._wage_response_model(order, assigned_courier)
                    self.platform_cost += order.price
                else:
                    order.price = self._wage_response_model(order, assigned_courier) * 1.5
                    self.platform_cost += order.price
                
                assigned_courier.wait_to_pick.append(order)
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
    
    def _fairness_threshold_allocation(self, orders):
        
        def get_predicted_orders():
            
            index = (self.clock - self.start_time) // self.interval - 1
            predicted_count = int(self.predicted_count.iloc[index])

            predicted_orders = []

            assigned_poi_ids = np.random.choice(
                self.poi_frequency['poi_id'],
                size=predicted_count,
                p=self.poi_frequency['frequency_ratio']
            )
            order_id_index = 0
            for poi_id in assigned_poi_ids:
                data = self.poi_frequency[self.poi_frequency['poi_id'] == poi_id]
                
                eta = data['avg_delivery_time'].values[0] + self.clock

                order_create_time = self.clock
                pickup_point = (data['sender_lat'].values[0] / 1e6, data['sender_lng'].values[0] / 1e6)
                dropoff_point = (data['recipient_lat'].values[0] / 1e6, data['recipient_lng'].values[0] / 1e6)
        
                order = Order(order_id_index, 0, order_create_time, pickup_point, dropoff_point, 0, eta)
                predicted_orders.append(order)
                
                order_id_index += 1

            return predicted_orders

        speed_upper_bound = 4
        
        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        predicted_orders = get_predicted_orders()
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
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if max_speed < speed_upper_bound:
                    price = self._wage_response_model(order, courier)
                    if len(courier.waybill) + len(courier.wait_to_pick) > 0:
                        formal_speed_fair, formal_speed, formal_max_speed = self._cal_speed(None, courier)  
                    else:
                        formal_speed_fair = 0
                    speed_variation = avg_speed_fair - formal_speed_fair
                    cost =  speed_variation / price
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

    def _MaxMin_fairness_allocation(self, orders):
        speed_upper_bound = 4
        
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
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if max_speed < speed_upper_bound:
                    price = self._wage_response_model(order, courier)
                    if len(courier.waybill) + len(courier.wait_to_pick) > 0:
                        formal_speed_fair, formal_speed, formal_max_speed = self._cal_speed(None, courier)  
                    else:
                        formal_speed_fair = 0
                    speed_variation = avg_speed_fair - formal_speed_fair
                    cost =  speed_variation / price
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
                    
    def _Pairwise_fairness_allocation(self, orders):
        speed_upper_bound = 4
        
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
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if max_speed < speed_upper_bound:
                    price = self._wage_response_model(order, courier)
                    if len(courier.waybill) + len(courier.wait_to_pick) > 0:
                        formal_speed_fair, formal_speed, formal_max_speed = self._cal_speed(None, courier)  
                    else:
                        formal_speed_fair = 0
                    speed_variation = avg_speed_fair - formal_speed_fair
                    cost =  speed_variation / price
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

        for courier in self.couriers:
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

    def _cal_speed(self, order, courier):
        order_sequence = self._cal_sequence(order, courier)
        order_speed = {}
        visited_orders = set()  
        orders = (courier.waybill + courier.wait_to_pick).copy()
        if order is not None:
            orders.append(order)
        
        total_dist = geodesic(courier.position, order_sequence[0][0]).meters
        order_id = order_sequence[0][2]

        if order_sequence[0][1] == 'dropped':
            visited_orders.add(order_sequence[0][2])
            matched_order = next((o for o in orders if o.orderid == order_id), None)
            order_speed[order_id] = total_dist / (matched_order.ETA - matched_order.order_create_time)
        
        for i in range(1, len(order_sequence)):
            prev_location = order_sequence[i-1][0]
            current_location = order_sequence[i][0]
            total_dist += geodesic(prev_location, current_location).meters

            order_status = order_sequence[i][1]
            order_id = order_sequence[i][2]
            
            if order_status == 'dropped' and order_id not in visited_orders:
                visited_orders.add(order_id)

                matched_order = next((o for o in orders if o.orderid == order_id), None)

                if matched_order:
                    # 单位为m/s
                    order_speed[order_id] = total_dist / (matched_order.ETA - matched_order.order_create_time - 2*np.clip(np.random.normal(180, 40), 0, 300)) if matched_order.ETA - matched_order.order_create_time - 2*np.clip(np.random.normal(180, 40), 0, 300) > 0 else total_dist / (matched_order.ETA - matched_order.order_create_time)
                        
        max_eta = max([o.ETA for o in orders])
        time_window = max_eta - self.clock
                    
        avg_speed_fair = sum(order_speed.values()) / len(order_speed.values())
        avg_speed = total_dist / time_window
        max_speed = max(order_speed.values())

        return avg_speed_fair, avg_speed, max_speed
    
    def _cal_sequence(self, order, courier):
        orders = (courier.waybill + courier.wait_to_pick).copy()
        if order is not None:
            orders.append(order)
        # ETA reveals the sequence of the appearance of orders on the platform
        orders = sorted(orders, key=lambda o: o.ETA)
        order_sequence = []
        drop_off_point = []
        
        if len(orders) == 1:
            order_sequence.append((orders[0].pick_up_point, 'pick_up', orders[0].orderid))
            order_sequence.append((orders[0].drop_off_point, 'dropped', orders[0].orderid))

        else:
            i = 0
            while(i < len(orders)):
                if orders[i] in courier.waybill:
                    order_sequence.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                    i += 1
                else:
                    order_sequence.append((orders[i].pick_up_point, 'pick_up', orders[i].orderid))
                    drop_off_point.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                    i += 1
                    break
            
            last_location = order_sequence[-1][0]
            distance = []
            for point in drop_off_point:
                distance.append((geodesic(last_location, point[0]).meters, point))
            distance.sort(key=lambda x: x[0])
            
            while(i < len(orders)):
                if orders[i] in courier.waybill:
                    dist = geodesic(last_location, orders[i].drop_off_point).meters
                    if drop_off_point == []:
                        order_sequence.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                        distance.clear()
                        last_location = orders[i].drop_off_point
                        for point in drop_off_point:
                            distance.append((geodesic(last_location, point[0]).meters, point))
                        distance.sort(key=lambda x: x[0])
                        i += 1
                    else:
                        for d, point in distance:
                            if d < dist:
                                order_sequence.append(point)
                                drop_off_point.remove(point)
                            else:
                                order_sequence.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                                distance.clear()
                                last_location = orders[i].drop_off_point
                                for point in drop_off_point:
                                    distance.append((geodesic(last_location, point[0]).meters, point))
                                distance.sort(key=lambda x: x[0])
                                i += 1
                                break

                else:
                    dist = geodesic(last_location, orders[i].pick_up_point).meters
                    if drop_off_point == []:
                        order_sequence.append((orders[i].pick_up_point, 'pick_up', orders[i].orderid))
                        drop_off_point.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                        distance.clear()
                        last_location = orders[i].pick_up_point
                        for point in drop_off_point:
                            distance.append((geodesic(last_location, point[0]).meters, point))
                        distance.sort(key=lambda x: x[0])
                        i += 1
                    else:
                        for d, point in distance:
                            if d < dist:
                                order_sequence.append(point)
                                drop_off_point.remove(point)
                            else:
                                order_sequence.append((orders[i].pick_up_point, 'pick_up', orders[i].orderid))
                                drop_off_point.append((orders[i].drop_off_point, 'dropped', orders[i].orderid))
                                distance.clear()
                                last_location = orders[i].pick_up_point
                                for point in drop_off_point:
                                    distance.append((geodesic(last_location, point[0]).meters, point))
                                distance.sort(key=lambda x: x[0])
                                i += 1
                                break

            for _, point in distance:
                order_sequence.append(point)

        return order_sequence
        
    def _wage_response_model(self, order, courier):
        courier_total_time = self.clock - courier.start_time
        if courier_total_time == 0 or (courier.income == 0 and len(courier.waybill) + len(courier.wait_to_pick) == 0):
            return 10 # as a incentive for a new courier, also 10 is the average price of an order
        elif courier.income == 0 and courier.courier_type == 1 and len(courier.waybill) + len(courier.wait_to_pick) > 0:
            wm = 15 / 3600
            v = 4 # 4 m/s
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
            v = 4 # 4 m/s
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

            # m = 1 / courier.total_leisure_time
            # n = 1 / (courier_total_time - courier.total_leisure_time)
            
            # wm = courier.income / courier_total_time # income_per_sec
            # v = 4 # 4 m/s
                    
            # r = geodesic(order.pick_up_point, courier.position).meters
            # d = geodesic(order.pick_up_point, order.drop_off_point).meters
            
            # wp = (m * wm) / (n * (m + n)) * (n * courier_total_time + m / (m + n))
            # wage = wp * (r + d) / v
            
            # if courier.courier_type == 1:
            #     return wage + 3 # flexible+fixed
            # else:
            #     return wage
    
    # def get_actions(self):
    #     num_agents = len(self.couriers)
    #     capacity = self.couriers[0].capacity
    #     available_actions = [[0] * capacity for _ in range(num_agents)]
    #     for i, courier in enumerate(self.couriers):
    #         length = len(courier.waybill) + len(courier.wait_to_pick) - 1
    #         for j in range(min(length, capacity)):
    #             available_actions[i][j] = 1
                
    #     return available_actions