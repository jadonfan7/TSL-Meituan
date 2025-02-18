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
        self.total_riding_time = 0
        self.total_congestion_time = 0
        self.total_running_time = 0
        
        self.total_waiting_time = 0
        self.current_waiting_time = 0
        self.is_waiting = 0

        self.reject_order_num = 0
        self.finish_order_num = 0
        
        self.position = position
        self.waybill = []
        self.travel_distance = 0
        self.wait_to_pick = []
        self.target_location = None
        self.is_target_locked = False
        
        self.order_sequence = []
        self.current_wave_dist = 0
        self.current_risk = 0

        self.speed = 0
        self.actual_speed = 0
        
        self.congestion_rate = 0
        
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
        self.waybill.remove(order)
        order.status = 'dropped'
        self.finish_order_num += 1

    def move(self, map, current_map):
        
        if self.speed != 0 and self.current_waiting_time < map.interval:
            # default_congestion_rate = 0.8
            # speed = self.speed * congestion_rate
            
            if self.current_waiting_time > 0:
                time = self.current_waiting_time - map.interval
                self.current_waiting_time = 0
            else:
                time = map.interval
            
            travel_distance = time * self.speed
            distance_to_target = geodesic(self.target_location, self.position).meters

            if travel_distance >= distance_to_target:
                new_latitude, new_longitude = self.target_location
                
            else:
                ratio = travel_distance / distance_to_target
                
                new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
                new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                            
            self.congestion_rate = self.get_congestion_rate(current_map, new_latitude, new_longitude, travel_distance)
            
            if self.congestion_rate != 1:
                speed = self.speed / (1 + self.congestion_rate ** 4)
                self.total_congestion_time += time * (self.congestion_rate ** 4)
            else:
                speed = self.speed
            
            self.total_riding_time += time # congestion has showed on speed
            
            travel_distance = time * speed
            
            if travel_distance >= distance_to_target:
                map.update_courier_position(self.position[0], self.position[1], self.target_location[0], self.target_location[1], self)
                self.position = self.target_location
                self.is_target_locked = False
                self.travel_distance += distance_to_target
                
            else:
                ratio = travel_distance / distance_to_target
                
                new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
                new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                
                map.update_courier_position(self.position[0], self.position[1], new_latitude, new_longitude, self)
                
                self.position = (new_latitude, new_longitude)

                self.travel_distance += travel_distance
            
            
                
    def get_congestion_rate(self, map, latitude, longitude, travel_distance, radius=30):
        points_on_line = [(self.position[0], self.position[1])]
        num_steps = int(travel_distance // 60) # 60m as a step

        for i in range(1, num_steps + 1):
            ratio = i / num_steps
            new_latitude = self.position[0] + ratio * (latitude - self.position[0])
            new_longitude = self.position[1] + ratio * (longitude - self.position[1])
            points_on_line.append((new_latitude, new_longitude))

        nearby_couriers_count = 0
        nearby_couriers = set()
        
        nearby_couriers.update(map.get_couriers_in_adjacent_grids(self.position[0], self.position[1]))
        nearby_couriers.update(map.get_couriers_in_adjacent_grids(latitude, longitude))
        
        for courier in nearby_couriers:
            for point in points_on_line:
                distance_to_point = geodesic(courier.position, point).meters
                if distance_to_point <= radius:
                    nearby_couriers_count += 1
                    break
        
        flow_capacity = 13 * travel_distance
        congestion_rate =  nearby_couriers_count / flow_capacity if nearby_couriers_count > flow_capacity else 1
        return congestion_rate