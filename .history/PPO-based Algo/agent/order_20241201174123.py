class Order:
    def __init__(self, OrderID, is_prebooked, is_in_the_same_da_and_poi, order_create_time, pick_up_point, drop_off_point, meal_prepare_time, estimate_arrived_time):
        self.orderid = OrderID
        
        self.status = 'wait_pair' # wait_pair, wait_pick, picked_up, dropped

        self.is_prebooked = is_prebooked
        self.is_in_the_same_da_and_poi = is_in_the_same_da_and_poi        
        self.order_create_time = order_create_time
        self.meal_prepare_time = meal_prepare_time
        self.pick_up_point = pick_up_point
        self.drop_off_point = drop_off_point
        self.ETA = estimate_arrived_time
        
        self.pair_time = None
        
        self.is_late = 0
        self.ETA_usage = 0
        self.reject_count = 0
        self.price = 0

    def __repr__(self):
        message = 'cls: ' + type(self).__name__  + ', order_id: ' + str(self.orderid) + ', status: ' + self.status + ', pick_up_point: ' + str(self.pick_up_point) + ', drop_off_point: ' + str(self.drop_off_point) + ', pair_time: ' + str(self.pair_time)
        if self.status == 'dropped':
            if self.is_late:
                message += ', is_late: ' + str(self.is_late) 
            else:
                message += ', ETA_usage: ' + str(self.ETA_usage)

        return message