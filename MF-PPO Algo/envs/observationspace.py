import numpy as np

class ObservationSpace:
    def __init__(self, map, courier=None):

        self.orders = map.orders
        self.num_orders = len(self.orders)
        # self.share = share
        if courier is not None:
            self.courier = courier

        self.lng_min = map.lng_min
        self.lng_max = map.lng_max
        
        self.lat_min = map.lat_min
        self.lat_max = map.lat_max

        self.time_min = map.time_min
        self.time_max = map.time_max
             
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def get_obs(self):

        order_obs = []

        # 归一化订单信息
        for order in self.courier.waybill:

            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)

            prepare_time = self.normalize(order.meal_prepare_time, self.time_min, self.time_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([-1, -1, drop_off_x, drop_off_y, prepare_time, ETA])

        for order in self.courier.wait_to_pick:
            pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
            pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)

            prepare_time = self.normalize(order.meal_prepare_time, self.time_min, self.time_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y, prepare_time, ETA])
            
        orders_array = np.array(order_obs).flatten()
        
        courier_obs = []
            
        courier_pos_x = self.normalize(self.courier.position[0], self.lat_min, self.lat_max)
        courier_pos_y = self.normalize(self.courier.position[1], self.lng_min, self.lng_max)
        speed = self.normalize(self.courier.speed, 1, 7) if self.courier.speed > 0 else 0
        if self.courier.target_location is not None:
            target_x = self.normalize(self.courier.target_location[0], self.lat_min, self.lat_max)
            target_y = self.normalize(self.courier.target_location[1], self.lat_min, self.lat_max)
        else:
            target_x = -1
            target_y = -1
        courier_obs.append([courier_pos_x, courier_pos_y, target_x, target_y, speed])
   
        couriers_array = np.array(courier_obs).flatten()

        combined_obs = np.concatenate((orders_array, couriers_array))
        
        obs_dim = 6 * self.courier.capacity + 5
        if combined_obs.size < obs_dim:
            combined_obs = np.pad(combined_obs, (0, obs_dim - combined_obs.size), 'constant', constant_values=-1)
            
        return combined_obs
        