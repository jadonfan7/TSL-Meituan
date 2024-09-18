from gym.spaces import Box
import numpy as np

class ObservationSpace:
    def __init__(self, gridmap, courier=None, share=False):
        self.orders = gridmap.orders
        self.num_orders = len(self.orders)
        self.share = share
        if courier is not None:
            self.courier = courier
        if self.share:
            self.couriers = gridmap.couriers
            self.num_couriers = len(self.couriers)
            
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def get_obs(self):

        order_obs = []
    
        # 设定坐标和时间的取值范围
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        # time_min, time_max = 0, 1440  # 假设时间范围是一天中的分钟数

        # 归一化订单信息
        for order in self.orders:
            pick_up_x = self.normalize(order.pick_up_point[0], x_min, x_max)
            pick_up_y = self.normalize(order.pick_up_point[1], y_min, y_max)
            drop_off_x = self.normalize(order.drop_off_point[0], x_min, x_max)
            drop_off_y = self.normalize(order.drop_off_point[1], y_min, y_max)
            # arrive_time = self.normalize(order.estimate_arrive_time, time_min, time_max)
            order_obs.append([pick_up_x, pick_up_y, drop_off_x, drop_off_y])

        orders_array = np.array(order_obs).flatten()
        
        courier_obs = []
        if self.share:
            # normalization
            for courier in self.couriers:
                courier_pos_x = self.normalize(courier.position[0], x_min, x_max)
                courier_pos_y = self.normalize(courier.position[1], y_min, y_max)
                courier_obs.append([courier_pos_x, courier_pos_y])
        else:
            courier_pos_x = self.normalize(self.courier.position[0], x_min, x_max)
            courier_pos_y = self.normalize(self.courier.position[1], y_min, y_max)
            courier_obs.append([courier_pos_x, courier_pos_y])
            
        couriers_array = np.array(courier_obs).flatten()

        # 合并订单和骑手数据
        combined_obs = np.concatenate((orders_array, couriers_array))

        # 返回订单和骑手信息的Box空间
        return combined_obs
    