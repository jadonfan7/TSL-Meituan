from agent.order import Order
# from utils.distance import haversine
from geopy.distance import geodesic
import math

class Courier:
    def __init__(self, courier_type, CourierID, position, time, state='inactive'):
        self.courier_type = courier_type # 0专送，1众包
        self.courierid = CourierID
        self.start_time = time
        
        # self.status = 'idle' # idle, picking_up, dropping_off
        self.position = position
        self.waybill = []
        self.travel_distance = 0
        self.wait_to_pick = []
        self.target_location = None
        self.is_target_locked = False
        self.speed = 0
        self.state = state # inactive, active
        self.capacity = 10
        self.reward = 0
        
        self.income = 0
        
        self.waiting_list = []
        
    def __repr__(self):
        message = 'cls: ' + type(self).__name__ + ', courier_id: ' + str(self.courierid) + ', position: ' + str(self.position) + ', travel_distance: ' + str(self.travel_distance) + ', income: ' + str(self.income)

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
        # self.status = 'picking_up'
        self.wait_to_pick.append(order)


    def pick_order(self, order):
        self.waybill.append(order)
        self.wait_to_pick.remove(order)
        order.status = 'picked_up'

        # self.status = 'dropping_off'

    def drop_order(self, order):
        self.waybill.remove(order)
        order.status = 'dropped'

        # if self.waybill==[]:
        #     self.status = 'idle'

    def reject_order(self, order):
        pass

    def move(self, interval):
        if self.speed != 0:
            travel_distance = interval * self.speed
            distance_to_target = geodesic(self.target_location, self.position).meters
            
            if travel_distance >= distance_to_target:
                self.position = self.target_location
                self.is_target_locked = False
                self.travel_distance += distance_to_target
                
            else:
                ratio = travel_distance / distance_to_target

                new_latitude = self.position[0] + ratio * (self.target_location[0] - self.position[0])
                new_longitude = self.position[1] + ratio * (self.target_location[1] - self.position[1])
                
                self.position = (new_latitude, new_longitude)

                self.travel_distance += travel_distance
        
