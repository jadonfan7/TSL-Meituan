import numpy as np

class ObservationSpace:
    def __init__(self, map, courier=None):
    # def __init__(self, map, courier=None, share=False):

        self.orders = map.orders
        self.num_orders = len(self.orders)
        # self.share = share
        if courier is not None:
            self.courier = courier
        # if self.share:
        #     self.couriers = map.couriers
        #     self.num_couriers = len(self.couriers)

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

            # prepare_time = self.normalize(order.prepare_time, self.time_min, self.time_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([-1, -1, drop_off_x, drop_off_y, ETA])

        for order in self.courier.wait_to_pick:
            pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
            pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)

            # prepare_time = self.normalize(order.prepare_time, self.time_min, self.time_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y, ETA])

        orders_array = np.array(order_obs).flatten()
        
        courier_obs = []
        # if self.share:
        #     # normalization
        #     for courier in self.couriers:
        #         courier_pos_x = self.normalize(courier.position[0], self.lng_min, self.lng_max)
        #         courier_pos_y = self.normalize(courier.position[1], self.lat_min, self.lat_max)
        #         courier_obs.append([courier_pos_x, courier_pos_y])
        # else:
            # courier_pos_x = self.normalize(self.courier.position[0], self.lng_min, self.lng_max)
            # courier_pos_y = self.normalize(self.courier.position[1], self.lat_min, self.lat_max)
            # courier_obs.append([courier_pos_x, courier_pos_y])
            
        courier_pos_x = self.normalize(self.courier.position[0], self.lat_min, self.lat_max)
        courier_pos_y = self.normalize(self.courier.position[1], self.lng_min, self.lng_max)
        courier_obs.append([courier_pos_x, courier_pos_y])

            
        couriers_array = np.array(courier_obs).flatten()

        # 合并订单和骑手数据
        combined_obs = np.concatenate((orders_array, couriers_array))
        
        if combined_obs.size < 52:
            combined_obs = np.pad(combined_obs, (0, 52 - combined_obs.size), 'constant', constant_values=-1)

        # 返回订单和骑手信息的Box空间
        return combined_obs
    