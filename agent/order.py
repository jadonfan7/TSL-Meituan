class Order:
    def __init__(self, OrderID,  pick_up_point, drop_off_point, prebook=0, prepare_time=5, estimate_arrive_time=50):
        self.orderid = OrderID
        self.prebook = prebook
        self.status = 'wait_pair' # wait_pair, wait_pick, picked_up, dropped
        self.prepare_time = prepare_time
        self.estimate_arrive_time = estimate_arrive_time
        self.pick_up_point = pick_up_point
        self.drop_off_point = drop_off_point
        # self.ETA = ETA

    def __repr__(self):
        return 'cls: ' + type(self).__name__  + ', order_id: ' + str(self.orderid) + ', status: ' + self.status + ', pick_up_point: ' + str(self.pick_up_point) + ', drop_off_point: ' + str(self.drop_off_point)

if __name__ == '__main__':
    p = Order("001",(0,0),(0,1))
    print(p)