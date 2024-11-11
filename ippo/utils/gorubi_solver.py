from gurobipy import *
# from utils.distance import haversine
from geopy.distance import geodesic


def gorubi_solver(couriers, orders, clock):
    num_couriers = len(couriers)
    num_orders = len(orders)

    capacity = couriers[0].capacity
    v = 3
    M = 10e6

    # 创建模型
    model = Model('dispatch orders')
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 10)

    # defined sets
    all_orders = {}
    destination = None

    for i in range(num_couriers):
        all_orders[i] = []
        all_orders[i].append(couriers[i].position)
        for order in couriers[i].wait_to_pick:
            all_orders[i].append(order.pick_up_point)

        for order in orders:
            all_orders[i].append(order.pick_up_point)

        for order in couriers[i].wait_to_pick:
            all_orders[i].append(order.drop_off_point)

        for order in orders:
            all_orders[i].append(order.drop_off_point)

        for order in couriers[i].waybill:
            all_orders[i].append(order.drop_off_point)

        all_orders[i].append(destination)

    PICK = {}
    DROP = {}
    for i in range(num_couriers):
        num_pick_up_point = len(couriers[i].wait_to_pick) + len(orders)
        num_drop_off_point = len(couriers[i].waybill) + len(couriers[i].wait_to_pick) + len(orders)
        PICK[i] = []
        DROP[i] = []
        for p in range(num_pick_up_point + 1):
            PICK[i].append(p)
        for d in range(num_drop_off_point + 1):
            DROP[i].append(d + num_pick_up_point + 1)
    
    J_old_pick = {}
    for i in range(num_couriers):
        num_wait_to_pick = len(couriers[i].wait_to_pick)
        J_old_pick[i] = []
        for num in range(num_wait_to_pick):
            J_old_pick[i].append(PICK[i][1 + num])

    J_old_drop = {}
    for i in range(num_couriers):
        num_wait_to_pick = len(couriers[i].wait_to_pick)
        num_waybill = len(couriers[i].waybill)
        J_old_drop[i] = []
        for num in range(num_wait_to_pick):
            J_old_drop[i].append(DROP[i][num])
        for num in range(num_waybill):
            J_old_drop[i].append(DROP[i][num_wait_to_pick + num_orders + num])

    J_new_pick = {}
    for i in range(num_couriers):
        num_wait_to_pick = len(couriers[i].wait_to_pick)
        J_new_pick[i] = []
        for num in range(num_orders):
            J_new_pick[i].append(PICK[i][num_wait_to_pick + num + 1])
        
    J_new_drop = {}
    for i in range(num_couriers):
        num_wait_to_pick = len(couriers[i].wait_to_pick)
        J_new_drop[i] = []
        for num in range(num_orders):
            J_new_drop[i].append(DROP[i][num_wait_to_pick + num])


    dist = {}
    for i in range(num_couriers):
        length = len(all_orders[i])
        for m in range(length):
            for n in range(length):
                if n == length - 1:
                    dist[i, m, n] = 0
                elif m == length - 1:
                    dist[i, m, n] = M
                else:
                    dist[i, m, n] = geodesic(all_orders[i][m], all_orders[i][n]).meters

    # 添加决策变量
    x = {}
    for i in range(num_couriers):
        for j in range(num_orders):
            x[i, j] = model.addVar(vtype = GRB.BINARY, name = f"x_{i}_{j}")

    y = {}
    for i in range(num_couriers):
        length = len(all_orders[i])
        for m in range(length):
            for n in range(length):
                y[i, m, n] = model.addVar(vtype = GRB.BINARY, name = f"y_{i}_{m}_{n}")
    t = {}
    for i in range(num_couriers):
        length = len(all_orders[i])
        for m in range(length):
            t[i, m] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"t_{i}_{m}")

    # 设置目标函数
    expr = quicksum((y[i, m, n] * dist[i, m, n]) for i in range(num_couriers) for n in range(len(all_orders[i])) for m in range(len(all_orders[i])))

    model.setObjective(expr, GRB.MINIMIZE)


    # 设置约束
    # (1)
    for i in range(num_couriers):
        if J_old_drop == {}:
            model.addConstr(quicksum(x[i, j] for j in range(num_orders)) <= capacity)
        else:
            model.addConstr(quicksum(x[i, j] for j in range(num_orders)) + len(J_old_drop[i]) <= capacity)
    # (2)
    for j in range(num_orders):
        model.addConstr(quicksum(x[i, j] for i in range(num_couriers)) == 1)
    # (3)
    for i in range(num_couriers):
        length = len(PICK[i] + DROP[i])
        for j in range(len(J_new_pick[i])):
            model.addConstr(x[i, j] == quicksum(y[i, m, J_new_pick[i][j]] for m in range(length) if m != J_new_pick[i][j]))
    # (4)
    for i in range(num_couriers):
        length = len(PICK[i] + DROP[i])
        for j in range(len(J_new_drop[i])):
            model.addConstr(x[i, j] == quicksum(y[i, m, J_new_drop[i][j]] for m in range(length) if m != J_new_drop[i][j]))
    # (5)
    for i in range(num_couriers):
        length = len(PICK[i] + DROP[i])
        for j in range(len(J_old_drop[i])):
            model.addConstr(quicksum(y[i, m, J_old_drop[i][j]] for m in range(length) if m != J_old_drop[i][j]) == 1)
    # (6)
    for i in range(num_couriers):
        length = len(PICK[i] + DROP[i])
        for m in range(length):
            if m == 0:
                model.addConstr(quicksum((y[i, m, n] - y[i, n, m]) for n in range(length)) == 1)
            elif m == length - 1:
                model.addConstr(quicksum((y[i, m, n] - y[i, n, m]) for n in range(length)) == -1)
            else:
                model.addConstr(quicksum((y[i, m, n] - y[i, n, m]) for n in range(length)) == 0)
    # (7)
    for i in range(num_couriers):
        length = len(all_orders[i])
        for m in range(length):
            for n in range(length):
                model.addConstr(t[i, n] >= t[i, m] + dist[i, m, n] / v - M * (1 - y[i, m, n]))
    # (8)
    for i in range(num_couriers):
        for j in range(num_orders):
            model.addConstr(t[i, J_new_pick[i][j]] <= t[i, J_new_drop[i][j]])

    for i in range(num_couriers):
        for j in range(len(J_old_pick[i])):
            model.addConstr(t[i, J_old_pick[i][j]] <= t[i, J_old_drop[i][j]])

    model.optimize()

    for i in range(num_couriers):
        for j in range(num_orders):
            if x[i, j].X > 1 - 1e-6:  # 如果x[i,j]接近于1
                # 将订单分配给对应的骑手
                assigned_courier = couriers[i]
                order = orders[j]
                assigned_courier.wait_to_pick.append(order)
                order.status = 'wait_pick'
                order.pair_time = clock
                  
                # 检查骑手是否在取货点
                if assigned_courier.position == order.pick_up_point:
                    assigned_courier.pick_order(order)
                
                # 检查骑手是否在送货点
                if assigned_courier.position == order.drop_off_point:
                    assigned_courier.drop_order(order)
    

    # print(f"Optimal solution found in {model.Runtime} seconds.")
    # print(f"objective value:{model.ObjVal}")

    # # 打印决策变量 x
    # print("x variables:")
    # for i in range(num_couriers):
    #     for j in range(num_orders):
    #         if x[i, j].X > 1e-6:  # 只打印值大于0的变量
    #             print(f"x[{i},{j}] = {x[i,j].X}")

    # # 打印决策变量 y
    # print("y variables:")
    # for i in range(num_couriers):
    #     length = len(all_orders[i])
    #     for m in range(length):
    #         for n in range(length):
    #             if y[i, m, n].X > 1e-6:  # 只打印值大于0的变量
    #                 print(f"y[{i},{m},{n}] = {y[i,m,n].X}")