from geopy.distance import great_circle
import numpy as np
import random

class Courier:
    def __init__(self, courier_type, CourierID, position, time, state='inactive'):
        self.save = 0
        
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
        
        self.order_sequence = []
        self.current_wave_dist = 0
        self.current_risk = 0

        self.speed = 0
        self.actual_speed = 0
                
        self.capacity = 10
        self.reward = 0
        
        
        self.income = 0
                
    def __repr__(self):
        message = 'cls: ' + type(self).__name__ + ', courier_id: ' + str(self.courier_id) + ', type: ' + str(self.courier_type) + ', state: ' + self.state + ', reservation: ' + str(self.save) + ', position: ' + str(self.position) + ', travel_distance: ' + str(round(self.travel_distance, 2)) + ', income: ' + str(round(self.income, 2)) + ', total_leisure_time: ' + str(self.total_leisure_time) + ', total_running_time: ' + str(self.total_running_time)

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

    def move(self, map):
        reward = 0
        time = map.interval
        old_lat, old_lng = self.position
        if self.speed != 0 and self.current_waiting_time < time and self.order_sequence != []:

            if self.current_waiting_time > 0:
                time -= self.current_waiting_time
                self.current_waiting_time = 0
            
            travel_distance = time * self.speed * 0.8 # congestion rate
            distance_to_target = great_circle(self.target_location, self.position).meters
            
            if travel_distance > distance_to_target:
                self.travel_distance += distance_to_target   
                self.total_riding_time += distance_to_target / (self.speed * 0.8)
                self.position = self.target_location
                reward += self._pick_or_drop(map)
                if self.speed > 4:
                    reward += 30
            else:
                ratio = travel_distance / distance_to_target
                new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
                new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                self.position = (new_latitude, new_longitude)
                
                self.travel_distance += travel_distance       
                self.total_riding_time += time
        
        map.update_courier_position(old_lat, old_lng, self.position[0], self.position[1], self)
        return reward
        
        # reward = 0
        # time = map.interval
        # old_lat, old_lng = self.position
        # while self.speed != 0 and self.current_waiting_time <= time and self.order_sequence != []:

        #     if self.current_waiting_time > 0:
        #         time -= self.current_waiting_time
        #         self.current_waiting_time = 0
            
        #     if time == 0:
        #         return
            
        #     travel_distance = time * self.speed * 0.8 # congestion rate
        #     distance_to_target = great_circle(self.target_location, self.position).meters
            
        #     if travel_distance > distance_to_target :
        #         travel_distance -= distance_to_target
        #         self.travel_distance += distance_to_target   
        #         self.total_riding_time += distance_to_target / (self.speed * 0.8)
        #         time -= distance_to_target / (self.speed * 0.8)
                
        #         self.position = self.target_location
        #         reward += self._pick_or_drop(map.clock)
        #         self.order_sequence.pop(0)
        #         continue
            
        #     ratio = travel_distance / distance_to_target
            
        #     new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
        #     new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
        #     self.position = (new_latitude, new_longitude)
            
        #     self.travel_distance += travel_distance       
        #     self.total_riding_time += time
        #     break
        
        # map.update_courier_position(old_lat, old_lng, self.position[0], self.position[1], self)
        # return reward
            
    def _pick_or_drop(self, map):
        reward = 0
        for order in self.wait_to_pick:
            if self.position == order.pick_up_point and map.clock >= order.meal_prepare_time: # picking up
                order.wait_time = map.clock - order.meal_prepare_time if order.meal_prepare_time > 0 else 0
                reward -= int(order.wait_time / 60)
                self.pick_order(order)
                self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                
                if self.courier_type == 0:
                    reward += 150
                else:
                    reward += 200
                    
            elif self.position == order.pick_up_point and map.clock < order.meal_prepare_time and self.is_waiting == 0:
                waiting_time = order.meal_prepare_time - map.clock
                reward -= int(waiting_time / 60)
                self.current_waiting_time = waiting_time
                self.total_waiting_time += waiting_time
                self.is_waiting == 1
                
                self.pick_order(order)
                self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                
                if self.courier_type == 0:
                    reward += 150
                else:
                    reward += 200
                
        for order in self.waybill:
            if self.position == order.drop_off_point:  # dropping off
                self.drop_order(order)
                if len(self.waybill + self.wait_to_pick) > 0:
                    self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                else:
                    self.order_sequence = []
                    self.current_wave_dist = 0
                    self.current_risk = 0
                                    
                if map.clock > order.ETA:
                    if self.courier_type == 0:
                        reward -= 150 + 80 * ((map.clock - order.order_create_time) / (order.ETA - order.order_create_time) - 1)
                    else:
                        reward -= 150 + 100 * ((map.clock - order.order_create_time) / (order.ETA - order.order_create_time) - 1)
                        
                    self.income += order.price * 0.7
                    order.is_late = 1
                                            
                else:
                    order.ETA_usage = (map.clock - order.order_create_time) / (order.ETA - order.order_create_time)
                    if self.courier_type == 0:
                        reward += 200 + 80 * (1 - order.ETA_usage)
                    else:
                        reward += 300 + 100 * (1 - order.ETA_usage)
                        
                    self.income += order.price 
        return reward
    