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
    def __init__(self, env_index=0, algo_index=0):
        self.orders_id = set()
        self.couriers_id = set()
        self.orders = []
        self.couriers = []
        self.env_index = env_index
        self.algo_index = algo_index
        self.current_index = 0
        
        
        df = pd.read_csv('../all_waybill_info_meituan_0322.csv')
        # df_waybill = pd.read_csv('../dispatch_rider_meituan.csv')
        
        config_mapping = {
            0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665977400},
            1: {'date': 20221017, 'start_time': 1666000800, 'end_time': 1666002600},
            2: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666063800},
            3: {'date': 20221018, 'start_time': 1666087200, 'end_time': 1666089000},
            4: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666150200},
            5: {'date': 20221019, 'start_time': 1666173600, 'end_time': 1666175400},
            6: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666236600},
            7: {'date': 20221020, 'start_time': 1666260000, 'end_time': 1666261800},
            8: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666323000},
            9: {'date': 20221021, 'start_time': 1666346400, 'end_time': 1666348200},
            10: {'date': 20221022, 'start_time': 1666407600, 'end_time': 1666409400},
        } # half an hour

        # 根据 env_index 获取相应的日期和时间范围
        if self.env_index in config_mapping:
            config = config_mapping[self.env_index]
            date_value = config['date']
            start_time = config['start_time']
            end_time = config['end_time']
            
            # 筛选和排序数据
            df = df[(df['dispatch_time'] > 0) & (df['dt'] == date_value) & (df['da_id'] == 0)]
            df = df.sort_values(by=['platform_order_time'], ascending=True)
            df = df[(df['platform_order_time'] >= start_time) & (df['platform_order_time'] < end_time)]
            self.order_data = df.reset_index(drop=True)
            
            # self.waybill_data = df_waybill[(df_waybill['dt'] == date_value)]


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

        self.clock = self.order_data['platform_order_time'][0]

        self.interval = 10

        self.add_new_couriers = 0

        self.step(first_time=1)

    def reset(self, env_index):
        self.orders = []
        self.couriers = []
        self.__init__(env_index, self.algo_index)

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + ', size:' + str(self.size) + '\n'
        for c in self.couriers:
            message += repr(c) + '\n'
        for p in self.orders:
            message += repr(p) + '\n'
        return message                

    def step(self, first_time=0):
        
        self.add_new_couriers = 0
        
        if not first_time:
            self.clock += self.interval

        orders = [order for order in self.orders if order.status == "wait_to_pick"]

        while(self.current_index < self.order_data.shape[0] and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
            dt = self.order_data.iloc[self.current_index]
            order_id = dt['order_id']
            
            if order_id not in self.orders_id and dt['is_courier_grabbed'] == 1 and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0:
                
                self.orders_id.add(order_id)
                
                is_prebook = dt['is_prebook']
                is_in_the_same_da_and_poi = 1 if dt['da_id'] == dt['poi_id'] else 0
                order_create_time = dt['platform_order_time']
                pickup_point = (dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6)
                dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                meal_prepare_time = dt['estimate_meal_prepare_time']
                estimate_arrived_time = dt['estimate_arrived_time']
                
                order = Order(order_id, is_prebook, is_in_the_same_da_and_poi, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                orders.append(order)

                courier_id = dt['courier_id']
                if courier_id not in self.couriers_id:
                    self.couriers_id.add(courier_id)
                    courier_type = 1 if random.random() > 0.2 else 0 # 0.8众包, 0.2专送
                    courier_location = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    courier = Courier(courier_type, courier_id, courier_location, self.clock)
                    courier.state = 'active'
                    courier.start_time = self.clock
                    
                    # if self.algo_index == 1:
                    #     courier.wait_to_pick.append(order)
                    #     order.pair_time = self.clock
                    self.couriers.append(courier)
                    self.add_new_couriers += 1
                    
            
            self.current_index += 1
            
        
        if orders != []:
            # self.current_index = count
            self.orders += orders
            # self.couriers += couriers
            if self.algo_index == 0:
                self._equitable_allocation(orders)   
            # else:
            #     nearby_couriers = None
            #     for i, p in enumerate(orders):
            #         nearby_couriers = self._get_nearby_couriers(p, 1500)
            #     gorubi_solver(nearby_couriers, orders, self.clock)
            else:
                self._greedy_allocation(orders)
                
        for courier in self.couriers:
            if self.clock - self.last_pick_up_time > 300:
                courier.state = 'inactive'
                self.couriers.

        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)
        
    def _accept_or_reject(self, order, courier):
                
        avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
        reward = courier.speed - avg_speed_fair
        fairness = abs(avg_speed_fair - avg_speed)
        
        num_waybill = len(courier.waybill + courier.wait_to_pick)
        potential_overspeed_risk = 1 if max_speed > 4 else 0
        rejection_history_count = order.reject_count
        is_in_the_same_da_and_poi = order.is_in_the_same_da_and_poi
        is_prebooked = order.is_prebooked
        pick_up_distance = geodesic(courier.position, order.pick_up_point).meters
        drop_off_distance = geodesic(order.pick_up_point, order.drop_off_point).meters
        estimate_arrived_time = order.ETA - self.clock if order.ETA - self.clock > 0 else 0
        estimate_meal_prepare_time = order.meal_prepare_time - self.clock if order.meal_prepare_time - self.clock > 0 else 0
        order_push_time = self.clock - order.order_create_time

        X = (reward, fairness, num_waybill, potential_overspeed_risk, rejection_history_count, is_in_the_same_da_and_poi, is_prebooked, pick_up_distance, drop_off_distance, estimate_arrived_time, estimate_meal_prepare_time, order_push_time)
        best_logreg = joblib.load('/Users/jadonfan/Documents/TSL/best_logreg_model.joblib')
        y_pred_new = best_logreg.predict(X)
        
        if y_pred_new == 0:
            order.reject_count += 1

        return y_pred_new
        
    def _greedy_allocation(self, orders):
        
        min_dist = math.inf
        nearest_courier = None
        
        for order in orders:
            for courier in self.couriers:
                if courier.state == 'active' and len(courier.waybill) + len(courier.wait_to_pick) < courier.capacity:
                    dist = geodesic(courier.position, order.pick_up_point).meters
                if min_dist > dist:
                    min_dist = dist
                    nearest_courier = courier
            
            decision = self._accept_or_reject(order, courier)
            if decision == 1:
                nearest_courier.wait_to_pick.append(order)
                order.status = 'wait_pick'
                order.pair_time = self.clock
                
                if nearest_courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:  # picking up
                    nearest_courier.pick_order(order)

                    if nearest_courier.position == order.drop_off_point:  # dropping off
                        nearest_courier.drop_order(order)
    
    def _bipartite_allocation(self, orders):
        speed_upper_bound = 4

        # Create a cost matrix
        cost_matrix = []
        couriers = set()
        
        for order in orders:
            nearby_couriers = self._get_nearby_couriers(order)
            couriers.update(nearby_couriers)
        
        couriers = list(couriers)
        
        for order in orders:
            row = []
            for courier in couriers:
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(order, courier)
                if max_speed < speed_upper_bound:
                    avg_reward = courier.reward / (self.clock - courier.start_time) if (self.clock - courier.appear_time) != 0 else courier.reward
                    cost = avg_reward # Define the cost based on factors you consider reasonable
                    row.append(cost)
                else:
                    row.append(np.inf)  # Set an infinite cost if the assignment is unreasonable
            cost_matrix.append(row)

        cost_matrix = np.array(cost_matrix)
        
        # Solve the bipartite matching problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Assign orders to couriers based on the optimal matching
        for order_index, courier_index in zip(row_ind, col_ind):
            order = orders[order_index]
            courier = couriers[courier_index]
            
            decision = self._accept_or_reject(order, courier)
            if decision == 1:
                courier.wait_to_pick.append(order)
                order.status = 'wait_pick'
                order.pair_time = self.clock

                # Pick up or drop off orders as needed
                if courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:
                    courier.pick_order(order)
                    if courier.position == order.drop_off_point:
                        courier.drop_order(order)

    def _equitable_allocation(self, orders):
        speed_upper_bound = 4

        for i, p in enumerate(orders):
            min_reward = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(p, 1500)
            for courier in nearby_couriers:
                avg_speed_fair, avg_speed, max_speed = self._cal_speed(p, courier)
                avg_reward = courier.reward / (self.clock - courier.start_time) if (self.clock - courier.appear_time) != 0 else courier.reward
                if min_reward > avg_reward and max_speed < speed_upper_bound:
                    min_reward = avg_reward
                    assigned_courier = courier
            
            if assigned_courier is not None:
                
                decision = self._accept_or_reject(p, assigned_courier)
                
                if decision == 1:
                    assigned_courier.wait_to_pick.append(p)
                    p.status = 'wait_pick'
                    p.pair_time = self.clock

                    if assigned_courier.position == p.pick_up_point and self.clock >= p.prepare_time:  # picking up
                        assigned_courier.pick_order(p)

                        if assigned_courier.position == p.drop_off_point:  # dropping off
                            assigned_courier.drop_order(p)


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
                    order_speed[order_id] = total_dist / (matched_order.ETA - matched_order.order_create_time)
                    
        waybill_etas = [o.ETA for o in courier.waybill]
        wait_to_pick_etas = [o.ETA for o in courier.wait_to_pick]
    
        max_eta = max(waybill_etas + wait_to_pick_etas)
        time_window = max_eta - self.clock
                    
        avg_speed_fair = sum(order_speed.values()) / len(order_speed.values())
        avg_speed = total_dist / time_window
        max_speed = max(order_speed.values())

        return avg_speed_fair, avg_speed, max_speed
    
    def _cal_sequence(self, order, courier):
        orders = (courier.waybill + courier.wait_to_pick).copy()
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
        
    def get_actions(self):
        num_agents = len(self.couriers)
        available_actions = [[0] * 10 for _ in range(num_agents)]
        for i, courier in enumerate(self.couriers):
            length = len(courier.waybill) + len(courier.wait_to_pick) - 1
            for j in range(min(length, 10)):
                available_actions[i][j] = 1
                
        return available_actions