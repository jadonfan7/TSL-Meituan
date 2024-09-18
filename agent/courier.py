from agent.order import Order
# from agent import Agent

class Courier:
    def __init__(self, CourierID, position):
        self.courierid = CourierID
        # self.status = 'idle' # idle, picking_up, dropping_off
        self.position = position
        self.waybill = []
        # self.path = []
        self.travel_distance = 0
        self.wait_to_pick = []

    def __repr__(self):
        message = 'cls: ' + type(self).__name__ + ', courier_id: ' + self.courierid + ', position: ' + str(self.position) + ', travel_distance: ' + str(self.travel_distance) + ', wait_to_pick: ' + str(self.wait_to_pick)
        if self.waybill != []:
            message += ', order: ' + str(self.waybill)
        else:
            message += ', order: None'
        return message

    def pair_order(self, order):
        order.status = 'wait_pick'
        # self.status = 'picking_up'
        self.wait_to_pick.append(order.orderid)

    def pick_order(self, order):
        self.waybill.append(order.orderid)
        if order.orderid in self.wait_to_pick:
            self.wait_to_pick.remove(order.orderid)
        order.status = 'picked_up'
        # Agent.reward += 10
        # self.status = 'dropping_off'

    def drop_order(self, order):
        self.waybill.remove(order.orderid)
        order.status = 'dropped'
        # Agent.reward += 10

        # if self.waybill==[]:
        #     self.status = 'idle'

    def reject_order(self, order):
        pass

    def move(self, direction):
        if direction == 'up':
            if self.position[1] + 1 < 10:
                self.position = (self.position[0], self.position[1] + 1)
                self.travel_distance += 1
        elif direction == 'down':
            if self.position[1] > 0:
                self.position = (self.position[0], self.position[1] - 1)
                self.travel_distance += 1
        elif direction == 'left':
            if self.position[0] > 0:
                self.position = (self.position[0] - 1, self.position[1])
                self.travel_distance += 1
        elif direction == 'right':
            if self.position[0] + 1 < 10:
                self.position = (self.position[0] + 1, self.position[1])
                self.travel_distance += 1
        

    # def assign_path(self, path1, path2):
    #     self.path = (path1 + path2)

    # def count_distance(self):
    #     assert self.status != 'idle'
    #     self.position = self.path.pop(0)
    #     self.travel_distance += 1

if __name__ == '__main__':
    c = Courier('1', (0,0))
    print(c)
    p = Order('001', (0,0), (1,0))
    print(p)
    c.pair_order(p)
    print(c)
    print(p)
    c.pick_order(p)
    print(c)
    print(p)
    c.drop_order(p)
    print(c)
    print(p)