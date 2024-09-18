import math
from utils.distance import manhattan
from agent.courier import Courier
from agent.order import Order

class GridMap:
    def __init__(self):
        self.world_length = 50
        size = (10, 10)
        # Define courier IDs, locations, and order IDs, pickup and dropoff points
        courierid = ['001', '002', '003', '004', '005']
        orderid = ['01', '02', '03', '04', '05']
        courier_location = [(0, 0), (1, 1), (2,7), (3,6), (5,9)]  
        order_pickup_point = [(0,0), (2, 3), (3,3), (4,0), (5,2)]  
        order_dropoff_point = [(1, 0), (0, 1), (3, 2), (2,8), (5, 4)]

        couriers = [Courier(i, j) for i,j in zip(courierid, courier_location)]
        initial_orders = [Order(i,j,k) for i,j,k in zip(orderid, order_pickup_point, order_dropoff_point)]  
        

        # Initialize GridMap and Environment
        self.size = size #(row, col)
        self.couriers = couriers
        self.orders = initial_orders
        
        self.num_couriers = len(couriers)
        self.num_orders = len(initial_orders)
        
        self._greedy_fcfs(initial_orders)
        


    def reset(self):
        # Define courier IDs, locations, and order IDs, pickup and dropoff points
        courierid = ['001', '002', '003', '004', '005']
        orderid = ['01', '02', '03', '04', '05']
        courier_location = [(0, 0), (1, 1), (2,7), (3,6), (5,9)]  
        order_pickup_point = [(0,0), (2, 3), (3,3), (4,0), (5,2)]  
        order_dropoff_point = [(1, 0), (0, 1), (3, 2), (2,8), (5, 4)]

        couriers = [Courier(i, j) for i,j in zip(courierid, courier_location)]
        initial_orders = [Order(i,j,k) for i,j,k in zip(orderid, order_pickup_point, order_dropoff_point)]  
        

        # Initialize GridMap and Environment
        self.couriers = couriers
        self.orders = initial_orders
        
        self.num_couriers = len(couriers)
        self.num_orders = len(initial_orders)
        
        self._greedy_fcfs(initial_orders)

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + ', size:' + str(self.size) + '\n'
        for c in self.couriers:
            message += repr(c) + '\n'
        for p in self.orders:
            message += repr(p) + '\n'
        return message

    def is_valid(self, p1):
        if p1[0]<0 or p1[1]<0 or p1[0]>self.size[0] or p1[1]>self.size[1]:
            return False
        return True
                
    def init_zero_map_cost(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                p = (row, col)
                p_up = (row-1, col)
                if self.is_valid(p_up): self.map_cost[(p, p_up)] = 0
                p_right = (row, col+1)
                if self.is_valid(p_right): self.map_cost[(p, p_right)] = 0
                p_down = (row+1, col)
                if self.is_valid(p_down): self.map_cost[(p, p_down)] = 0
                p_left = (row, col-1)
                if self.is_valid(p_left): self.map_cost[(p, p_left)] = 0

    def add_couriers(self, couriers):
        self.couriers.extend(couriers)

    def add_orders(self, orders):
        self.orders.extend(orders)

    # def plan_path(self, start_point, end_point):

    #     def check_optim(next_pos, min_dist, optim_next_pos):
    #         if self.is_valid(next_pos):
    #             dist = manhattan(next_pos, end_point)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 optim_next_pos = next_pos
    #         return min_dist, optim_next_pos
        
    #     path = []
    #     curr_pos = start_point
    #     while curr_pos != end_point:
    #         min_dist = math.inf
    #         optim_next_pos = None
    #         # up
    #         min_dist, optim_next_pos = check_optim((curr_pos[0]-1, curr_pos[1]), min_dist, optim_next_pos)
    #         # down
    #         min_dist, optim_next_pos = check_optim((curr_pos[0]+1, curr_pos[1]), min_dist, optim_next_pos)
    #         # left
    #         min_dist, optim_next_pos = check_optim((curr_pos[0], curr_pos[1]-1), min_dist, optim_next_pos)
    #         # right
    #         min_dist, optim_next_pos = check_optim((curr_pos[0], curr_pos[1]+1), min_dist, optim_next_pos)

    #         assert optim_next_pos is not None, 'no valid position for optim_next_pos'
    #         path.append(optim_next_pos)
    #         curr_pos = optim_next_pos

    #     return path

    def step(self):
        orderid = ['06', '07', '08', '09', '10']
        order_pickup_point = [(6,0), (6, 6), (7,3), (7,7), (8,1)]  
        order_dropoff_point = [(3,10), (7, 6), (4, 5), (8,10), (5,8)]

        new_orders = [Order(i, j, k) for i, j, k in zip(orderid, order_pickup_point, order_dropoff_point)]
        
        self.add_orders(new_orders)
        self.num_orders = len(new_orders)
        
        self._equitable_allocation(new_orders)

    def visualize(self):
        m = [["" for i in range(self.size[1])] for j in range(self.size[0])]
        for o in self.orders:
            if o.status == 'wait_pair' or o.status == 'wait_pick':
                # order
                m[o.pick_up_point[1]][o.pick_up_point[0]] += "p" + str(o.orderid)
        for c in self.couriers:
            m[c.position[1]][c.position[0]] += "c" + str(c.courierid) + str(c.waybill)
        
        for i in range(len(m)):
            for j in range(len(m[0])):
                if m[i][j]=="":
                    m[i][j]="     "

        m.reverse()

        for row in m:
            print(row)


    def _greedy_fcfs(self, orders):
        allocation = [0]*len(orders)

        for i, p in enumerate(orders):
            min_dist = math.inf
            assigned_courier = None
            for c in self.couriers:
                dist = manhattan(p.pick_up_point, c.position)
                if dist < min_dist:
                    min_dist = dist
                    assigned_courier = c
            if assigned_courier is not None:
                allocation[i] = assigned_courier.courierid
                # 更新快递员状态
                assigned_courier.pair_order(p)

                if assigned_courier.position == p.pick_up_point: # picking up
                    assigned_courier.pick_order(p)

                if assigned_courier.position == p.drop_off_point:  # dropping off
                    assigned_courier.drop_order(p)

        return allocation
    
    def _equitable_allocation(self, orders):
        allocation = [0] * len(orders)
        speed_upper_bound = 15

        for i, p in enumerate(orders):
            gap = math.inf
            assigned_courier = None
            best_insert_position = None

            nearby_couriers = self._get_nearby_couriers(p)
            for courier in nearby_couriers:
                min_speed, max_speed = self._cal_speed(p, courier)

                if (max_speed - min_speed < gap) and max_speed < speed_upper_bound:
                    gap = max_speed - min_speed
                    assigned_courier = courier
            
            if assigned_courier is not None:
                allocation[i] = assigned_courier.courierid
                # 插入订单到骑手的waybill
                assigned_courier.waybill.insert(best_insert_position, p)

                if assigned_courier.position == p.pick_up_point:  # picking up
                    assigned_courier.pick_order(p)
                if assigned_courier.position == p.drop_off_point:  # dropping off
                    assigned_courier.drop_order(p)

        return allocation

    def _get_nearby_couriers(self, order):
        nearby_couriers = []
        for courier in self.couriers:
            if manhattan(courier.position, order.pick_up_point) < 6:
                nearby_couriers.append(courier)
        return nearby_couriers

    def _cal_speed(self, order, courier):
        order_sequence = self._cal_sequence(order, courier)
        order_speed = {}
        visited_orders = set()  

        for i in range(len(order_sequence)):
            dist = 0
            order_type = order_sequence[i][1]
            order_id = order_sequence[i][2]
            
            if order_type == 'drop off' and order_id not in visited_orders:
                visited_orders.add(order_id)
                location = order_sequence[i][0]
                j = i - 1
                
                while j >= 0:
                    prev_order_id = order_sequence[j][2]
                    prev_location = order_sequence[j][0]
                    dist += manhattan(location, prev_location)
                    location = prev_location
                    
                    if prev_order_id == order_id and order_sequence[j][1] == 'pick up':
                        break
                    j -= 1


                matched_order = next((o for o in courier.waybill if o.orderid == order_id), None)
                
                if matched_order:
                    order_speed[order_id] = dist / matched_order.ETA  

        return min(order_speed).values(), max(order_speed).values()
                
    
    def _cal_sequence(self, order, courier):
        orders = courier.waybill.append(order)
        # ETA reveals the sequence of the appearance of orders on the platform
        orders = sorted(orders, key=lambda o: o.ETA)
        order_sequence = []
        order_sequence.append((orders[0].pick_up_point, 'pick up', orders[0].orderid))
        pointer1 = 0
        pointer2 = 1
        while pointer2 < len(orders):
            while pointer1 < len(order_sequence):
                last_location = order_sequence[-1][0]
                if (manhattan(last_location, orders[pointer1.drop_off_point]) <= manhattan(last_location, orders[pointer2].pick_up_point)) and pointer1 < pointer2:
                    order_sequence.append((orders[pointer1].drop_off_point, 'drop off', orders[pointer1].order_id))
                    pointer1 += 1
                else:
                    order_sequence.append((orders[pointer2].pick_up_point, 'pick up', orders[pointer2].order_id))
                    pointer2 += 1
                    break

        remain_order_drop = orders[pointer1:]
        if remain_order_drop != []:
            while remain_order_drop:
                next_point = min(remain_order_drop, key=lambda p: manhattan(order_sequence[-1][0], p.drop_off_point))
                order_sequence.append((next_point.drop_off_point, 'drop_off', next_point.order_id))
                remain_order_drop.remove(next_point)

        return order_sequence

    # def _equitable_allocation(self, orders):
    #     couriers = self.couriers
    #     allocation = [0] * len(orders)

    #     for i, p in enumerate(orders):
    #         min_travel_distcouriere = math.inf
    #         assigned_courier = None
    #         for c in couriers:
    #             pick_up_dist = manhattan(p.pick_up_point, c.position)
    #             drop_off_dist = manhattan(p.pick_up_point, p.drop_off_point)
    #             total_travel_distcouriere = c.travel_distcouriere + pick_up_dist + drop_off_dist
    #             # 有待考量！！！！！！！！

    #             # max(min(courier.travel_distcouriere))
    #             if total_travel_distcouriere < min_travel_distcouriere:
    #                 min_travel_distcouriere = total_travel_distcouriere
    #                 assigned_courier = c

    #         if assigned_courier is not None:
    #             allocation[i] = assigned_courier.courierid
    #             assigned_courier.pair_order(p)
    #             assigned_courier.travel_distcouriere += min_travel_distcouriere

    #             if assigned_courier.position == p.pick_up_point:  # picking up
    #                 assigned_courier.pick_order(p)
    #             if assigned_courier.position == p.drop_off_point:  # dropping off
    #                 assigned_courier.drop_order(p)
    #     return allocation

if __name__ == '__main__':
    m = GridMap((10,10), 1, 1, [Courier('1', (0,0))], [Order('001', (3,4), (5,5))])
    print(m)
    print('path from (0,0) to (5,5):')
    path = m.plan_path((0,0),(5,5))
    print(path)
    m.visualize()