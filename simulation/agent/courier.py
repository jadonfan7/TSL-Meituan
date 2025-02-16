from geopy.distance import geodesic
import numpy as np
import random

class Courier:
    def __init__(self, courier_type, CourierID, position, time, state='inactive'):
        self.courier_type = courier_type # 0专送，1众包
    
        self.courier_id = CourierID
        self.start_time = time
        self.state = state # inactive, active
        
        self.is_leisure = None
        self.leisure_time = time
        self.total_leisure_time = 0
        self.riding_time = 0
        self.total_running_time = 0
        
        self.stay_duration = 0
        
        self.reject_order_num = 0
        self.finish_order_num = 0
        
        self.position = position
        self.waybill = []
        self.travel_distance = 0
        self.wait_to_pick = []
        self.target_location = None
        
        self.order_sequence = []
        self.current_wave_dist = 0
        self.current_risk = 0
        
        self.speed = 3
        self.avg_speed = 0
        
        self.capacity = 10
        self.reward = 0
        
        self.da_count = dict()
        self.poi_count = dict()
        
        self.income = 0
                
    def __repr__(self):
        message = 'cls: ' + type(self).__name__ + ', courier_id: ' + str(self.courier_id) + ', type: ' + str(self.courier_type) + ', state: ' + self.state + ', position: ' + str(self.position) + ', travel_distance: ' + str(round(self.travel_distance, 2)) + ', income: ' + str(round(self.income, 2)) + ', total_leisure_time: ' + str(self.total_leisure_time) + ', total_running_time: ' + str(self.total_running_time)

        if self.waybill != []:
            orderid = [o.orderid for o in self.waybill]
            message += ', waybill: ' + str(orderid)
        else:
            message += ', waybill: None'
        
        if self.wait_to_pick != []:
            orderid = [o.orderid for o in self.wait_to_pick]
            message += ', wait_to_pick: ' + str(orderid)
        else:
            message += ', wait_to_pick: None'

        message += ', head_to: ' + str(self.target_location)
        message += ', speed: ' + str(self.speed)

        return message

    def pair_order(self, order):
        order.status = 'wait_pick'
        self.wait_to_pick.append(order)
        
        if order.da in self.da_count:
            self.da_count[order.da] += 1
        else:
            self.da_count[order.da] = 1

        if order.poi in self.poi_count:
            self.poi_count[order.poi] += 1
        else:
            self.poi_count[order.poi] = 1


    def pick_order(self, order):
        self.waybill.append(order)
        self.wait_to_pick.remove(order)
        order.status = 'picked_up'

    def drop_order(self, order):
        interval = 20
        self.waybill.remove(order)
        order.status = 'dropped'
        self.finish_order_num += 1
        self.stay_duration = np.ceil(np.clip(np.random.normal(180, 40), 0, 300) / interval)

    def move(self, map, current_map):
        
        # congestion_rate = random.uniform(1, 1.5)
        # speed = self.speed / congestion_rate
    
        self.target_location = self.order_sequence[0][0]
        travel_distance = map.interval * self.speed
        distance_to_target = geodesic(self.target_location, self.position).meters

        if travel_distance >= distance_to_target:
            new_latitude, new_longitude = self.target_location
            
        else:
            ratio = travel_distance / distance_to_target
            
            new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
            new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                        
        congestion_rate = self.get_congestion_rate(current_map, new_latitude, new_longitude)
        speed = congestion_rate * self.speed
        # speed = self.speed - 1.5
        
        # self.riding_time += interval / congestion_rate
        self.riding_time += map.interval
        
        travel_distance = map.interval * speed
        
        if travel_distance >= distance_to_target:
            map.update_courier_position(self.position[0], self.position[1], self.target_location[0], self.target_location[1], self)
            self.position = self.target_location
            self.travel_distance += distance_to_target
            
        else:
            ratio = travel_distance / distance_to_target
            
            new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
            new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
            
            map.update_courier_position(self.position[0], self.position[1], new_latitude, new_longitude, self)
            
            self.position = (new_latitude, new_longitude)

            self.travel_distance += travel_distance
            
                
    def get_congestion_rate(self, map, latitude, longitude, radius=200):
        points_on_line = [(self.position[0], self.position[1])]
        distance_to_target = geodesic(self.position, self.target_location).meters
        num_steps = int(distance_to_target // 100)  # 以100m为步长

        for i in range(1, num_steps + 1):
            ratio = i / num_steps
            new_latitude = self.position[0] + ratio * (latitude - self.position[0])
            new_longitude = self.position[1] + ratio * (longitude - self.position[1])
            points_on_line.append((new_latitude, new_longitude))

        nearby_couriers_count = 0
        nearby_couriers = map.get_couriers_in_adjacent_grids(self.position[0], self.position[1])
        for courier in nearby_couriers:
            if courier == self:
                continue
            for point in points_on_line:
                distance_to_point = geodesic(courier.position, point).meters
                if distance_to_point <= radius:
                    nearby_couriers_count += 1
                    break
        
        congestion_rate = 400 / nearby_couriers_count if nearby_couriers_count > 400 else 1
        return congestion_rate