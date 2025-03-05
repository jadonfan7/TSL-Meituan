# import numpy as np

# class ObservationSpace:
#     def __init__(self, map, courier=None):

#         self.orders = map.orders
#         self.clock = map.clock
#         self.num_orders = len(self.orders)
#         # self.share = share
#         if courier is not None:
#             self.courier = courier

#         self.lng_min = map.lng_min
#         self.lng_max = map.lng_max
        
#         self.lat_min = map.lat_min
#         self.lat_max = map.lat_max

#         self.time_min = map.time_min
#         self.time_max = map.time_max
             
#     def normalize(self, value, min_val, max_val):
#         return (value - min_val) / (max_val - min_val)

#     def get_obs(self, predicted_orders, tree):

#         order_obs = []

#         for order in self.courier.waybill:

#             drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
#             drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)

#             order_obs.append([drop_off_x, drop_off_y])

#         # for order in self.courier.wait_to_pick:
#         #     pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
#         #     pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
#         #     drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
#         #     drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)

#         #     order_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y])
#         for order in self.courier.wait_to_pick:
#             pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
#             pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)

#             order_obs.append([pick_up_x, pick_up_y])
        
#         if order_obs == []:
#             order_array = np.full((10, 2), -1).flatten()
#         else:
#             order_array = np.array(order_obs)

#             if len(order_array) < 10:
#                 padding = np.full((10 - len(order_array), 2), -1)
#                 order_array = np.vstack([order_array, padding])
                
#             order_array = order_array.flatten()

#         courier_obs = []
            
#         courier_pos_x = self.normalize(self.courier.position[0], self.lat_min, self.lat_max)
#         courier_pos_y = self.normalize(self.courier.position[1], self.lng_min, self.lng_max)
#         courier_obs.append([courier_pos_x, courier_pos_y])
   
#         couriers_array = np.array(courier_obs).flatten()

#         share_obs = []
#         indices = tree.query(self.courier.position, k=10)[1]
#         nearest_orders = [predicted_orders[i] for i in indices]
#         for order in nearest_orders:
#             pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
#             pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
#             # drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
#             # drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)
            
#             share_obs.append([pick_up_x, pick_up_y])
            
#         if len(share_obs) < 10:
#             padding = np.full((10 - len(share_obs), 2), -1)
#             share_obs = np.vstack([share_obs, padding])

#         obs = np.concatenate((order_array, couriers_array, np.array(share_obs).flatten()))
        
#         return obs


import numpy as np

class ObservationSpace:
    def __init__(self, map, courier=None):

        self.orders = map.orders
        self.clock = map.clock
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

    def get_obs(self, predicted_orders, tree):

        order_obs = []

        for order in self.courier.waybill:

            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([-1, -1, drop_off_x, drop_off_y, ETA])
        
        for order in self.courier.wait_to_pick:
            pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
            pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            order_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y, ETA])
        
        if order_obs == []:
            # order_array = np.full((10, 2), -1).flatten()
            order_array = np.full((10, 5), -1).flatten()
        else:
            order_array = np.array(order_obs)

            if len(order_array) < 10:
                # padding = np.full((10 - len(order_array), 2), -1)
                padding = np.full((10 - len(order_array), 5), -1)
                order_array = np.vstack([order_array, padding])
                
            order_array = order_array.flatten()

        courier_obs = []
            
        courier_pos_x = self.normalize(self.courier.position[0], self.lat_min, self.lat_max)
        courier_pos_y = self.normalize(self.courier.position[1], self.lng_min, self.lng_max)
        # courier_obs.append([courier_pos_x, courier_pos_y])

        speed = self.normalize(self.courier.speed, 0, 7) if self.courier.speed > 0 else 0
        clock = self.normalize(self.clock, self.time_min, self.time_max)
        courier_obs.append([courier_pos_x, courier_pos_y, speed, clock])
   
        couriers_array = np.array(courier_obs).flatten()

        share_obs = []
        indices = tree.query(self.courier.position, k=10)[1]
        nearest_orders = [predicted_orders[i] for i in indices]
        for order in nearest_orders:
            pick_up_x = self.normalize(order.pick_up_point[0], self.lat_min, self.lat_max)
            pick_up_y = self.normalize(order.pick_up_point[1], self.lng_min, self.lng_max)
            drop_off_x = self.normalize(order.drop_off_point[0], self.lat_min, self.lat_max)
            drop_off_y = self.normalize(order.drop_off_point[1], self.lng_min, self.lng_max)
            ETA = self.normalize(order.ETA, self.time_min, self.time_max)

            # share_obs.append([pick_up_x, pick_up_y])
            share_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y, ETA])

        if len(share_obs) < 10:
            padding = np.full((10 - len(share_obs), 5), -1)
            # padding = np.full((10 - len(share_obs), 2), -1)
            share_obs = np.vstack([share_obs, padding])

        obs = np.concatenate((order_array, couriers_array, np.array(share_obs).flatten()))
        
        return obs
