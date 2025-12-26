from geopy.distance import great_circle
import numpy as np
import random

class Courier:
    def __init__(self, courier_type, CourierID, position, time, state='inactive'):
        self.save = 0
        
        self.courier_type = courier_type # 0: dedicated delivery, 1: crowdsourced
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
        self.last_position = position  # For calculating progress rewards
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
                        
    def __repr__(self):
        message = 'cls: ' + type(self).__name__ + ', courier_id: ' + str(self.courier_id) + ', type: ' + str(self.courier_type) + ', state: ' + self.state + ', reservation: ' + str(self.save) + ', position: ' + str(self.position) + ', travel_distance: ' + str(round(self.travel_distance, 2)) + ', total_leisure_time: ' + str(self.total_leisure_time) + ', total_running_time: ' + str(self.total_running_time)

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
            
            travel_distance = time * self.speed
            distance_to_target = great_circle(self.target_location, self.position).meters
            
            if travel_distance > distance_to_target:
                self.travel_distance += distance_to_target   
                self.total_riding_time += distance_to_target / self.speed
                self.position = self.target_location
                reward += self._pick_or_drop(map)
                if self.speed > 4:
                    reward += 10
            else:
                ratio = travel_distance / distance_to_target
                new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
                new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                self.position = (new_latitude, new_longitude)
                
                self.travel_distance += travel_distance       
                self.total_riding_time += time
        
        map.update_courier_position(old_lat, old_lng, self.position[0], self.position[1], self)
        return reward
        
    def _pick_or_drop(self, map):
        reward = 0
        for order in self.wait_to_pick:
            if self.position == order.pick_up_point and map.clock >= order.meal_prepare_time: # picking up
                order.wait_time = map.clock - order.meal_prepare_time if order.meal_prepare_time > 0 else 0
                self.pick_order(order)
                self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                
                # Reduce pickup reward for better balance
                base_pickup_reward = 60 if self.courier_type == 0 else 80
                reward += base_pickup_reward
                    
            elif self.position == order.pick_up_point and map.clock < order.meal_prepare_time and self.is_waiting == 0:
                waiting_time = order.meal_prepare_time - map.clock
                self.current_waiting_time = waiting_time
                self.total_waiting_time += waiting_time
                self.is_waiting == 1
                
                # Reduce waiting penalty
                wait_penalty = -min(10, waiting_time / 120)  # 1 point penalty per 2 minutes, max 10 points
                reward += wait_penalty
                
                self.pick_order(order)
                self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                
                # Base pickup reward
                base_pickup_reward = 60 if self.courier_type == 0 else 80
                reward += base_pickup_reward
                
        for order in self.waybill:
            if self.position == order.drop_off_point:  # dropping off
                self.drop_order(order)
                if len(self.waybill + self.wait_to_pick) > 0:
                    self.order_sequence, self.current_wave_dist, self.current_risk = map.cal_wave_info(None, self)
                else:
                    self.order_sequence = []
                    self.current_wave_dist = 0
                    self.current_risk = 0
                
                # Calculate delivery time ratio
                time_usage = (map.clock - order.order_create_time) / (order.ETA - order.order_create_time)
                                    
                if map.clock > order.ETA:  # Late delivery
                    delay_ratio = time_usage - 1
                    # Reduce delay penalty to be more reasonable
                    base_penalty = -40 if self.courier_type == 0 else -50
                    delay_penalty = -min(30, 20 * delay_ratio)  # Delay penalty cap at 30 points
                    reward += base_penalty + delay_penalty                        
                    order.is_late = 1
                                            
                else:  # On-time or early delivery
                    # Base delivery reward
                    base_delivery_reward = 80 if self.courier_type == 0 else 100
                    # Efficiency reward: higher reward for earlier delivery
                    efficiency_bonus = min(15, 30 * (1 - time_usage))
                    reward += base_delivery_reward + efficiency_bonus
                    order.ETA_usage = time_usage
        return reward