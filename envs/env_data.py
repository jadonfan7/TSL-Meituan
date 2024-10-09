import math
# from utils.distance import haversine
from geopy.distance import geodesic
from agent.courier import Courier
from agent.order import Order
import pandas as pd
from utils.gorubi_solver import gorubi_solver

class Map:
    def __init__(self):
        self.orders_id = set()
        self.couriers_id = set()
        self.orders = []
        self.couriers = []

        self.available_couriers = []

        self.current_index = 0
        
        
        df = pd.read_csv('../all_waybill_info_meituan_0322.csv')
        df = df[(df['dispatch_time'] > 0) & (df['dt'] == 20221017)]

        df = df.sort_values(by=['platform_order_time'], ascending=[True])
        
        df = df[(df['platform_order_time'] >= 1665936000) & (df['platform_order_time'] <= 1666022371)]
        self.order_data = df.reset_index(drop=True)

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

        self.interval = 60

        courierid_set = set()
        for _, row in self.order_data[:120].iterrows():
            courier_id = row['courier_id']
            if courier_id not in courierid_set:
                courierid_set.add(courier_id)
                courier_location = (row['sender_lat'] / 1e6, row['sender_lng'] / 1e6)
                courier = Courier(courier_id, courier_location)
                self.couriers.append(courier)

        self.step(first_time=1)

    def reset(self):
        self.orders = []
        self.couriers = []
        self.__init__()

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + ', size:' + str(self.size) + '\n'
        for c in self.couriers:
            message += repr(c) + '\n'
        for p in self.orders:
            message += repr(p) + '\n'
        return message                

    def step(self, first_time=0):
        
        if not first_time:
            self.clock += self.interval

        # count = 0

        # orders = []
        # couriers = []
        orders = [order for order in self.orders if order.status == "wait_to_pick"]
        self.available_couriers = [courier for courier in self.couriers if len(courier.waybill) + len(courier.wait_to_pick) < courier.capacity and courier.state == 'active']

        
        # for index, row in self.order_data[self.current_index:].iterrows():
        #     platform_order_time = row['platform_order_time']

        #     if platform_order_time is not None and platform_order_time <= self.clock:
        #         order_id = row['order_id']

        #         if order_id not in self.orders_id and row['is_courier_grabbed'] == 1 and row['estimate_arrived_time'] - row['order_push_time'] > 0:

        #             self.orders_id.add(order_id)
        #             pickup_point = (row['grab_lat'] / 1e6, row['grab_lng'] / 1e6)
        #             dropoff_point = (row['recipient_lat'] / 1e6, row['recipient_lng'] / 1e6)
        #             # prepare_time = row['estimate_meal_prepare_time']
        #             estimate_arrived_time = row['estimate_arrived_time'] - row['order_push_time']
                    
        #             order = Order(order_id, pickup_point, dropoff_point, estimate_arrived_time)
        #             orders.append(order)
                
        #         target_courier_id = row['courier_id']
        #         courier = next((c for c in self.couriers if c.courierid == target_courier_id), None)
        #         courier.state = 'active'
        #         self.available_couriers.append(courier)
                

        #         # if courier_id not in self.couriers_id:
        #         #     self.couriers_id.add(courier_id)
        #         #     courier_location = (row['sender_lat'], row['sender_lng'])

        #         #     courier = Courier(courier_id, courier_location)
        #         #     couriers.append(courier)
        #     else:
        #         count = index
        #         break
        
        while(self.current_index <= 100 and self.order_data.iloc[self.current_index]['platform_order_time'] <= self.clock):
            dt = self.order_data.iloc[self.current_index]
            order_id = dt['order_id']
            
            if order_id not in self.orders_id and dt['is_courier_grabbed'] == 1 and dt['estimate_arrived_time'] - dt['order_push_time'] > 0:
                
                self.orders_id.add(order_id)
                pickup_point = (dt['grab_lat'] / 1e6, dt['grab_lng'] / 1e6)
                dropoff_point = (dt['recipient_lat'] / 1e6, dt['recipient_lng'] / 1e6)
                estimate_arrived_time = dt['estimate_arrived_time'] - dt['order_push_time']
                
                order = Order(order_id, pickup_point, dropoff_point, estimate_arrived_time)
                orders.append(order)
            
            target_courier_id = dt['courier_id']
            courier = next((c for c in self.couriers if c.courierid == target_courier_id), None)
            courier.state = 'active'
            self.available_couriers.append(courier)
            
            self.current_index += 1
            
        
        if orders != []:
            # self.current_index = count
            self.orders += orders
            # self.couriers += couriers
            self._equitable_allocation(orders)   
            
            # gorubi_solver(self.available_couriers, orders, self.clock)

        self.num_orders = len(self.orders)
        self.num_couriers = len(self.couriers)


    def _equitable_allocation(self, orders):
        # allocation = [0] * len(orders)
        speed_upper_bound = 4

        for i, p in enumerate(orders):
            min_speed = math.inf
            assigned_courier = None

            nearby_couriers = self._get_nearby_couriers(p)
            for courier in nearby_couriers:
                avg_speed, max_speed = self._cal_speed(p, courier)

                if min_speed > avg_speed and max_speed < speed_upper_bound:
                    min_speed = avg_speed
                    assigned_courier = courier
            
            if assigned_courier is not None:
                # allocation[i] = assigned_courier.courierid
                # 插入订单到骑手的waybill
                assigned_courier.wait_to_pick.append(p)
                p.status = 'wait_pick'
                p.pair_time = self.clock

                if assigned_courier.position == p.pick_up_point:  # picking up
                    assigned_courier.pick_order(p)

                    if assigned_courier.position == p.drop_off_point:  # dropping off
                        assigned_courier.drop_order(p)

        # return allocation

    def _get_nearby_couriers(self, order):
        nearby_couriers = []

        min_dist = math.inf
        nearest_courier = None

        for courier in self.available_couriers:
            if courier.state == 'active' and len(courier.waybill) + len(courier.wait_to_pick) < courier.capacity:
                dist = geodesic(courier.position, order.pick_up_point).meters
                if min_dist > dist:
                    min_dist = dist
                    nearest_courier = courier
                if dist < 1500: # paper from Hai Wang, but the average distance in the data is 608m
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
            order_speed[order_id] = total_dist / matched_order.ETA

        
        for i in range(1, len(order_sequence)):
            prev_location = order_sequence[i-1][0]
            current_location = order_sequence[i][0]
            total_dist += geodesic(prev_location, current_location).meters

            order_type = order_sequence[i][1]
            order_id = order_sequence[i][2]
            
            if order_type == 'dropped' and order_id not in visited_orders:
                visited_orders.add(order_id)

                matched_order = next((o for o in orders if o.orderid == order_id), None)

                if matched_order:
                    # 单位为m/s
                    order_speed[order_id] = total_dist / matched_order.ETA

        avg_speed = sum(order_speed.values()) / len(order_speed.values())
        max_speed = max(order_speed.values())

        return avg_speed, max_speed

                
    
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
        
    