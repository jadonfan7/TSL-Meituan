import math
from geopy.distance import great_circle
from agent.courier import Courier
from agent.order import Order
import pandas as pd
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import hdbscan
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

class Map:
    def __init__(self, env_index=0, algo_index=0, eval=False):
        self.eval = eval
        
        self.orders_id = set()
        self.couriers_id = set()
        self.orders = []
        self.couriers = []
        self.active_couriers = []
        
        self.num_couriers1 = 0
        self.num_couriers2 = 0
        
        self.env_index = env_index
        self.algo_index = algo_index
        self.current_index = 0
    
        self.platform_cost = 0
        
        df = pd.read_csv('all_waybill_info_meituan_0322.csv')
        
        order_num_estimate = pd.read_csv('order_num_estimation.csv')

        config_mapping = {
            0: {'date': 20221017, 'start_time': 1665975600, 'end_time': 1665982800},
            1: {'date': 20221018, 'start_time': 1666062000, 'end_time': 1666069200},
            2: {'date': 20221019, 'start_time': 1666148400, 'end_time': 1666155600},
            3: {'date': 20221020, 'start_time': 1666234800, 'end_time': 1666242000},
            4: {'date': 20221021, 'start_time': 1666321200, 'end_time': 1666328400},
        } # 11:00-13:00
        
        Cluster_0_Lng_Range = (174414242, 174685447)
        Cluster_0_Lat_Range = (45744563, 45959787)
        
        self.existing_courier_algo = 0
        if self.env_index in config_mapping:
            config = config_mapping[self.env_index]
            date_value = config['date']
            self.start_time = config['start_time']
            self.end_time = config['end_time']
            
            df = df[(df['dispatch_time'] > 0) & (df['dt'] == date_value) & (df['da_id'] == 0) & (df['is_courier_grabbed'] == 1) & (df['estimate_arrived_time'] > df['estimate_meal_prepare_time'])]
            df = df.sort_values(by=['platform_order_time'], ascending=True)
            df = df[(df['platform_order_time'] >= self.start_time) & (df['platform_order_time'] < self.end_time)]
            
            df = df[(df['sender_lat'] >= Cluster_0_Lat_Range[0]) & (df['sender_lat'] <= Cluster_0_Lat_Range[1])]
            df = df[(df['sender_lng'] >= Cluster_0_Lng_Range[0]) & (df['sender_lng'] <= Cluster_0_Lng_Range[1])]
            
            self.order_data = df.reset_index(drop=True)
            
            self.predicted_count = order_num_estimate[order_num_estimate['dt'] == date_value]['predicted_count']
                     
        lat_values = self.order_data[['sender_lat', 'recipient_lat', 'grab_lat']]
        lat_values_non_zero = lat_values[lat_values > 0].dropna()

        self.lat_min = lat_values_non_zero.min().min() / 1e6
        self.lat_max = lat_values_non_zero.max().max() / 1e6

        lng_values = self.order_data[['sender_lng', 'recipient_lng', 'grab_lng']]
        lng_values_non_zero = lng_values[lng_values > 0].dropna()

        self.lng_min = lng_values_non_zero.min().min() / 1e6
        self.lng_max = lng_values_non_zero.max().max() / 1e6
        
        order_time = self.order_data[['estimate_arrived_time', 'dispatch_time', 'fetch_time', 'arrive_time', 'estimate_meal_prepare_time', 'order_push_time', 'platform_order_time']]
        order_time_non_zero = order_time[order_time > 0].dropna()

        self.time_min = order_time_non_zero.min().min()
        self.time_max = order_time_non_zero.max().max()

        self.grid_size = 20
        
        self.lat_step = (self.lat_max - self.lat_min) / self.grid_size
        self.lng_step = (self.lng_max - self.lng_min) / self.grid_size

        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.interval = 30 # matching for every 30 seconds

        self.clock = self.start_time + self.interval
        
        self.da_frequency = pd.read_csv('order_da_frequency.csv')
        self.location_estimation_data = pd.read_csv('noon_peak_hour_data.csv')

        self.max_num_couriers = 906
        random.seed(42)
        for index, dt in self.order_data.iterrows():
            if len(self.couriers) >= self.max_num_couriers:
                break
            
            courier_id = dt['courier_id']
            if courier_id not in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                self.couriers_id.add(courier_id)
                courier_type = 1 if random.random() > 0.7 else 0 # 0.3 for crowdsourced, 0.7 for company-hired
                if courier_type == 0:
                    self.num_couriers1 += 1
                else:
                    self.num_couriers2 += 1
                courier_location = (dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6)
                courier = Courier(courier_type, courier_id, courier_location, None)
                courier.state = 'inactive'
                self.couriers.append(courier)
                
        self.num_couriers = len(self.couriers)
        self.num_orders = 0
    
    def reset(self, env_index, eval=False):
        self.orders = []
        self.couriers = []
        self.__init__(env_index, self.algo_index, eval)
        
        if eval == False:
            self.step(first_time=1)
        else:
            self.eval_step(first_time=1)

    def step(self, first_time=0):
        if self.eval:
            total_cost = self.eval_step(first_time)
        else:
            total_cost = self.train_step(first_time)
        
        return total_cost

    def train_step(self, first_time=0):
                
        if not first_time:
            if self.clock < self.end_time:
                self.clock += self.interval 

        total_cost = 0
        # if a courier does not get an order for a period of a time, he will quit the system.
        for courier in self.active_couriers:
            if courier.is_leisure == 1 and courier.state == 'active':
                courier.total_leisure_time += self.interval
            elif courier.is_leisure == 0 and courier.state == 'active':
                courier.total_running_time += self.interval

            # if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 600: # 10 minutes
            #     courier.state = 'inactive'
            #     self.active_couriers.remove(courier)
            #     self.remove_courier(courier.position[0], courier.position[1], courier)

            if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                salary_per_interval = 15 / 3600 * self.interval
                courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                self.platform_cost += salary_per_interval

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
                
                order = Order(order_id, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                orders_new.append(order)

            courier_id = dt['courier_id']
            if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                for courier in self.couriers:
                    if courier.courier_id == courier_id and courier.state == 'inactive':
                        courier.state = 'active'
                        self.active_couriers.append(courier)
                        courier.start_time = self.clock
                        courier.leisure_time = self.clock
                        self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                        break
            
            self.current_index += 1
            
        orders_pair = orders_failed + orders_new
        
        if orders_pair != []:
            
            self.orders += orders_new

            if self.algo_index == 0:
                total_cost = self._Delay_allocation(orders_pair)
        
        self.num_orders = len(self.orders)
        
        return total_cost    
    
    def eval_step(self, first_time=0):
        total_cost = 0
        if self.algo_index == 3:
            
            if not first_time:
                if self.clock < self.end_time:
                    self.clock += self.interval 
                    
            # if a courier does not get an order for a period of a time, he will quit the system.
            for courier in self.active_couriers:
                if courier.is_leisure == 1 and courier.state == 'active':
                    courier.total_leisure_time += self.interval
                elif courier.is_leisure == 0 and courier.state == 'active':
                    courier.total_running_time += self.interval

                # if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 600: # 10 minutes
                #     courier.state = 'inactive'
                #     self.active_couriers.remove(courier)
                #     self.remove_courier(courier.position[0], courier.position[1], courier)

                if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                    salary_per_interval = 15 / 3600 * self.interval
                    courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                    self.platform_cost += salary_per_interval
            
            while(self.current_index < self.order_data.shape[0] and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
                dt = self.order_data.iloc[self.current_index]
                order_id = dt['order_id']
                
                if dt['courier_id'] not in self.couriers_id:
                    self.current_index += 1
                    continue
            
                if order_id not in self.orders_id and dt['estimate_arrived_time'] - dt['platform_order_time'] > 0 and dt['is_courier_grabbed'] == 1:                        
                    order_create_time = dt['platform_order_time']
                    pickup_point = (dt['sender_lat'] / 1e6, dt['sender_lng'] / 1e6)
                    dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                    meal_prepare_time = dt['estimate_meal_prepare_time']
                    estimate_arrived_time = dt['estimate_arrived_time']
                
                    order = Order(order_id, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)

                    courier_id = dt['courier_id']
                    courier = None
                    for candidate in self.couriers:
                        if candidate.courier_id == courier_id:
                            courier = candidate
                            if courier.state == 'inactive':
                                courier.state = 'active'
                                self.active_couriers.append(courier)
                                courier.start_time = self.clock
                                courier.leisure_time = self.clock
                                self.existing_courier_algo += 1
                                self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                                break
                        
                    if len(courier.waybill) + len(courier.wait_to_pick) >= courier.capacity or courier == None:
                        self.current_index += 1
                        continue
                    
                    self.orders_id.add(order_id)
                    self.orders.append(order)

                    order.price = self._wage_response_model(order, courier)
                    self.platform_cost += order.price

                    courier.wait_to_pick.append(order)
                    courier.order_sequence, courier.current_wave_dist, courier.current_risk = self.cal_wave_info(None, courier)
                    order.pair_courier = courier
                    order.pair_time = self.clock
                    order.status = 'wait_pick'
            
                    if courier.position == order.pick_up_point and self.clock >= order.meal_prepare_time:
                        courier.pick_order(order)

                        if courier.position == order.drop_off_point:
                            courier.drop_order(order)
                                   
                self.current_index += 1
            
        else:
            
            if not first_time:
                if self.clock < self.end_time:
                    self.clock += self.interval 

            orders_failed = [order for order in self.orders if order.status == "wait_pair"]
            orders_new = []
            
            # if a courier does not get an order for a period of a time, he will quit the system.
            for courier in self.active_couriers:
                if courier.is_leisure == 1 and courier.state == 'active':
                    courier.total_leisure_time += self.interval
                elif courier.is_leisure == 0 and courier.state == 'active':
                    courier.total_running_time += self.interval

                # if courier.state == 'active' and courier.is_leisure == 1 and self.clock - courier.leisure_time > 600: # 10 minutes
                #     courier.state = 'inactive'
                #     self.active_couriers.remove(courier)
                #     self.remove_courier(courier.position[0], courier.position[1], courier)

                if courier.state == 'active' and courier.start_time != self.clock and courier.courier_type == 0:
                    salary_per_interval = 15 / 3600 * self.interval
                    courier.income += salary_per_interval # 15 is from the paper "The Meal Delivery Routing Problem", 26.4 is the least salary per hour in Beijing
                    self.platform_cost += salary_per_interval

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
                    
                    order = Order(order_id, order_create_time, pickup_point, dropoff_point, meal_prepare_time, estimate_arrived_time)
                    orders_new.append(order)

                courier_id = dt['courier_id']                        
                if courier_id in self.couriers_id and dt['grab_lat'] != 0 and dt['grab_lng'] != 0:
                    for courier in self.couriers:
                        if courier.courier_id == courier_id and courier.state == 'inactive':
                            courier.state = 'active'
                            self.active_couriers.append(courier)
                            courier.start_time = self.clock
                            courier.leisure_time = self.clock
                            self.add_courier(dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6, courier)
                            break

                self.current_index += 1
                
            orders_pair = orders_failed + orders_new
            
            if orders_pair != []:
                
                self.orders += orders_new

                if self.algo_index == 0:
                    total_cost = self._Delay_allocation(orders_pair)
                elif self.algo_index == 1:
                    self._Efficiency_allocation(orders_pair)     
                # elif self.algo_index == 2:
                #     self._Greedy_allocation(orders_pair)
                # self.algo_index == 4 is the origin allocation in the dataset  
            
        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)

        return total_cost
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
             
    def get_adjacent_grids(self, lat, lng):
        lat_index, lng_index = self.get_grid_index(lat, lng)

        adjacent_offsets = [(-1, 0), (1, 0), (0, 0), (0, -1), (0, 1),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        adjacent_grids = [(lat_index + offset[0], lng_index + offset[1]) for offset in adjacent_offsets]

        valid_adjacent_grids = [
            (lat, lng) for lat, lng in adjacent_grids
            if lat >= 0 and lat < self.grid_size and lng >= 0 and lng < self.grid_size
        ]

        return valid_adjacent_grids
   
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
        return True
        # if isinstance(order, list):
        #     for o in order:
        #         if o.reject_count >= 4:
        #             return True
        # else:
        #     if order.reject_count >= 4:
        #         return True

        # decision = True if np.random.rand() < 0.9 else False
        # return decision
    
    ##################
    # platform
    def get_predicted_orders(self):
            
        index = (self.clock - self.start_time) // self.interval - 1
        max_time_interval = self.location_estimation_data['time_interval'].max()
        next_index = min(index + 1, max_time_interval)
                                            
        predicted_count = int(self.predicted_count.iloc[index])

        predicted_orders = []
        da_frequency_row = self.da_frequency[self.da_frequency['time_interval'] == index].iloc[0]
        
        predicted_count = int(self.predicted_count.iloc[index])
        next_predicted_count = int(self.predicted_count.iloc[next_index])
    
        da_frequency_row = self.da_frequency[self.da_frequency['time_interval'] == index].iloc[0]
        next_da_frequency_row = self.da_frequency[self.da_frequency['time_interval'] == next_index].iloc[0]

        da_ids = da_frequency_row.index[1:]
        frequencies = da_frequency_row.values[1:]
        next_frequencies = next_da_frequency_row.values[1:]
        
        combined_frequencies = (frequencies + next_frequencies) / 2
        frequencies_normalized = combined_frequencies / np.sum(combined_frequencies)
        
        ############
        if '0' in da_ids:
            da_id_0_index = da_ids.get_loc('0')
            frequencies = frequencies_normalized[da_id_0_index]
        else:
            frequencies = 0

        num = int((predicted_count + next_predicted_count) * frequencies)
        da_id = 0
        
        model_data = self.location_estimation_data[
            (self.location_estimation_data['time_interval'].isin([index, next_index])) &
            (self.location_estimation_data['da_id'] == da_id) &
            (self.location_estimation_data['is_courier_grabbed'] == 1)
        ].reset_index(drop=True)

        poi_order_counts = model_data['poi_id'].value_counts().reset_index()
        poi_order_counts.columns = ['poi_id', 'order_count']
        
        mean_eta = int(np.mean(model_data['estimate_arrived_time'] - model_data['platform_order_time'])) + self.clock
        mean_mpt = int(np.mean(model_data['estimate_meal_prepare_time'] - model_data['platform_order_time'])) + self.clock
        
        coordinates = model_data[['sender_lat', 'sender_lng', 'recipient_lat', 'recipient_lng']].values / 1e6

        if len(coordinates) > num:
            gmm = GaussianMixture(n_components=num, random_state=42)
            gmm.fit(coordinates)
            predicted_coords = gmm.sample(num)[0]

            for coord in predicted_coords:
                pickup_point = (coord[0], coord[1])
                dropoff_point = (coord[2], coord[3])

                order_create_time = self.clock
                order = Order(-1, da_id, -1, order_create_time, pickup_point, dropoff_point, mean_mpt, mean_eta)
                predicted_orders.append(order)
        else:
            for i in range(len(coordinates)):
                pickup_point = (coordinates[i][0], coordinates[i][1])
                dropoff_point = (coordinates[i][2], coordinates[i][3])

                order_create_time = self.clock
                order = Order(-1, da_id, -1, order_create_time, pickup_point, dropoff_point, mean_mpt, mean_eta)
                predicted_orders.append(order)
                                
        return predicted_orders
        
    # give order to the nearest guy                        
    def _Efficiency_allocation(self, orders):
        nearest_courier = None
        courier_coords = np.array([courier.position for courier in self.active_couriers])
        tree = KDTree(courier_coords)

        for order in orders:
            indices = tree.query(order.pick_up_point, k=10)[1]
            nearest_courier = None
            for index in indices:
                candidate_courier = self.active_couriers[index]
                if len(candidate_courier.waybill) + len(candidate_courier.wait_to_pick) < candidate_courier.capacity:
                    nearest_courier = candidate_courier
                    break
            
            if nearest_courier is not None:
                self._courier_order_matching(order, nearest_courier)

    # Bipartitie matching with estimation
    def _Delay_allocation(self, orders):
        clustered_orders = self._cluster_orders(orders)
        batch_size = 50
        total_cost = 0
        count = 0
        
        for i in range(0, len(clustered_orders), batch_size):
            batch_orders = clustered_orders[i:i + batch_size]
            couriers = set()
            
            for order in batch_orders:
                if isinstance(order, list):
                    for o in order:
                        nearby_couriers = self._get_nearby_couriers(o)
                        couriers.update(nearby_couriers)
                else:
                    nearby_couriers = self._get_nearby_couriers(order)
                    couriers.update(nearby_couriers)

            couriers = list(couriers)
        
            M = 1e9
            cost_matrix = []
            for order in batch_orders:           

                row = []
                for courier in couriers:

                    if courier.save:
                        row.append(float(M))
                        continue
                    
                    unmatch = False
                    if isinstance(order, list):
                        if len(courier.waybill) + len(courier.wait_to_pick) + len(order) > courier.capacity:
                            unmatch = True
                        for o in order:
                            if great_circle(o.pick_up_point, courier.position).meters > 1500:
                                unmatch = True
                    else:
                        if len(courier.waybill) + len(courier.wait_to_pick) + 1 > courier.capacity or great_circle(order.pick_up_point, courier.position).meters > 1500:
                            unmatch = True   

                    if unmatch:
                        row.append(float(M))
                        continue
                    
                    sequence, dist, risk = self.cal_wave_info(order, courier)
                    if isinstance(order, list):
                        price = 0
                        for task in order:
                            price += self._wage_response_model(task, courier)
                    else:
                        price = self._wage_response_model(order, courier)
                    
                    detour = dist - courier.current_wave_dist
                    cost =  detour / price * 2 if risk == 1 else detour / price
                    row.append(cost)

                cost_matrix.append(row)

            cost_matrix = np.array(cost_matrix) 
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for order_index, courier_index in zip(row_ind, col_ind):
                order = batch_orders[order_index]
                assigned_courier = couriers[courier_index]
                
                if cost_matrix[order_index][courier_index] != float(M):
                    count += 1
                    total_cost += cost_matrix[order_index][courier_index] / 100
                
                    self._courier_order_matching(order, assigned_courier)
                    
        total_cost += 20 * (len(clustered_orders) - count)
        if count > 0:
            return total_cost / count
        else:
            return total_cost
    
    def _cluster_orders(self, orders):
        def great_circle_distance(point1, point2):
            sender1 = (point1[0], point1[1])
            sender2 = (point2[0], point2[1])
            recipient1 = (point1[2], point1[3])
            recipient2 = (point2[2], point2[3])

            sender_dist = great_circle(sender1, sender2).meters
            recipient_dist = great_circle(recipient1, recipient2).meters
            if sender_dist > 1000 or recipient_dist > 1000:
                return np.inf

            return sender_dist + recipient_dist
        
        order_features = []
        for order in orders:
            order_features.append([order.pick_up_point[0], order.pick_up_point[1], order.drop_off_point[0], order.drop_off_point[1]])
        
        order_features = np.array(order_features)
        
        dbscan = DBSCAN(eps=1000, min_samples=2, metric=great_circle_distance)
        dbscan_labels = dbscan.fit_predict(order_features)
        
        dbscan_clusters = defaultdict(list)
        dbscan_clustered_orders = []
        
        for i, label in enumerate(dbscan_labels):
            if label != -1:
                dbscan_clusters[label].append(orders[i])
            else:
                dbscan_clustered_orders.append(orders[i])

        new_clusters = []
        capacity = self.couriers[0].capacity
        for label, order_list in dbscan_clusters.items():
            while len(order_list) > capacity:
                new_clusters.append(order_list[:capacity])
                order_list = order_list[capacity:]
            
            if order_list:
                new_clusters.append(order_list)
        
        for cluster in new_clusters:
            dbscan_clustered_orders.append(cluster)
        
        return dbscan_clustered_orders
            
    def _courier_order_matching(self, orders, assigned_courier):
        if orders is not None:
            if isinstance(orders, list):
                pass
            else:
                orders = [orders]               
            
        if assigned_courier.courier_type == 0:
            for order in orders:
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
            assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self.cal_wave_info(None, assigned_courier)
                
        else:
            decision = self._accept_or_reject(orders, assigned_courier)
            if decision == True:
                for order in orders:
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
                assigned_courier.order_sequence, assigned_courier.current_wave_dist, assigned_courier.current_risk = self.cal_wave_info(None, assigned_courier)
                                               
            else:
                for order in orders:
                    order.reject_count += 1
                assigned_courier.reject_order_num += 1
        
    def _get_nearby_couriers(self, order):
        nearby_couriers = self.get_couriers_in_adjacent_grids(order.pick_up_point[0], order.pick_up_point[1])
        return nearby_couriers

    def cal_wave_info(self, order, courier):
        ################# 
        # first stage
        orders = (courier.waybill + courier.wait_to_pick).copy()
        if order is not None:
            if isinstance(order, list):
                orders.extend(order)
            else:
                orders.append(order) 
            
        points = []
        waybill_points = []

        for o in orders:
            if o in courier.waybill:
                points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
                waybill_points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
            else:
                points.append((o.pick_up_point, 'pick_up', o.meal_prepare_time, o.orderid))
                points.append((o.drop_off_point, 'dropped', o.ETA, o.orderid))
        
        # ETA reveals the sequence of the appearance of orders on the platform
        points = sorted(points, key=lambda o: o[2])
        order_sequence = []
        order_sequence.append(points[0])
        points.pop(0)
            
        while points:
            last_point = order_sequence[-1][0]
            closest_points = []

            for point in points:
                distance = great_circle(last_point, point[0]).meters
                closest_points.append((distance, point))
                
            closest_points.sort(key=lambda x: (x[0], x[1][2]))
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
                elif closest_point[1] == 'dropped' and closest_point in waybill_points:
                    order_sequence.append(closest_point)
                    points.remove(closest_point)
                    break
                else:
                    if all(closest_point[2] <= p[2] for dist, p in closest_points):
                        order_sequence.append(closest_point)
                        points.remove(closest_point)
                        break
                
        #################
        # second stage
        risk = 0
        dist = 0
        while(True):
            risk = 0
            dist = 0
            
            re = 0
            
            last_location = courier.position
            dist_between_grab_time = 0
            time = self.clock
            i = 0
            while i < len(order_sequence):
                point = order_sequence[i]
                dist += great_circle(last_location, point[0]).meters
                if point[1] == 'dropped':
                    dist_between_grab_time += great_circle(last_location, point[0]).meters
                    if 4 * (point[2] - time) < dist_between_grab_time and i != 0 and order_sequence[i-1][3] != point[3] and order_sequence[i-1][2] > point[2]:
                        order_sequence[i], order_sequence[i-1] = order_sequence[i-1], order_sequence[i]
                        re = 1
                        break
                    else:
                        if 4 * (point[2] - time) < dist_between_grab_time:
                            risk = 1
                        last_location = point[0]
                        i += 1
                else:
                    grab_dist = great_circle(last_location, point[0]).meters + dist_between_grab_time
                    grab_time = int(grab_dist / 4 + time)
                    if grab_time < point[2]:
                        time = point[2]
                    else:
                        time = grab_time
                    
                    last_location = point[0]
                    dist_between_grab_time = 0
                    i += 1
                    
            if re == 0:
                break
            else:
                continue
        
        return order_sequence, dist, risk
        
    def _wage_response_model(self, order, courier):
        courier_total_time = self.clock - courier.start_time
        if courier_total_time == 0 or (courier.income == 0 and len(courier.waybill) + len(courier.wait_to_pick) == 0):
            return 10 # as a incentive for a new courier, also 10 is the average price of an order
        else:
            wm = 15 / 3600
            v = 4 # 4 m/s
            r = great_circle(order.pick_up_point, courier.position).meters
            d = great_circle(order.pick_up_point, order.drop_off_point).meters
            wage = wm * (r + d) / v

            return wage
