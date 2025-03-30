import time
import numpy as np
import torch

from runner.base_runner import Runner
import matplotlib.pyplot as plt

from loguru import logger
import numpy as np
import pandas as pd

def _t2n(x):
    return x.detach().cpu().numpy()

class EnvRunner(Runner):
    def __init__(self):
        super(EnvRunner, self).__init__()

    def run(self, current_eval_time):
        
        self.eval_envs.reset(current_eval_time % 5)
        
        # eval info
        stats = {i: {
            # platform side
            "platform_cost": 0,
            "Hired_finish_num": [],
            "Hired_unfinish_num": [],
            "Hired_reject_num": [],
            "Hired_leisure_time": [],
            "Hired_running_time": [],
            "Hired_waiting_time": [],
            "Crowdsourced_finish_num": [],
            "Crowdsourced_unfinish_num": [],
            "Crowdsourced_reject_num": [],
            "Crowdsourced_leisure_time": [],
            "Crowdsourced_running_time": [],
            "Crowdsourced_waiting_time": [],

            # courier side
            "courier_num": 0,
            "Hired_num": 0,
            "Crowdsourced_num": 0,
            "Crowdsourced_on": 0,
            "Hired_distance_per_episode": [],
            "Crowdsourced_distance_per_episode": [],
            "Hired_actual_speed": [],
            "Hired_income": [],
            "Crowdsourced_actual_speed": [],
            "Crowdsourced_income": [],
            
            # order side
            "order_num": 0,
            "count_dropped_orders": 0,
            "count_unfinished_orders": 0,
            "unfinished_late_orders": 0,
            "count_reject_orders": 0,
            "max_reject_num": 0,
            "late_orders": 0,
            "ETA_usage": [],
            "order_price": [],
            "order_wait_pair": 0,
            "order_waiting_time": [],
        } for i in range(self.eval_envs.num_envs)}

        for eval_step in range(self.eval_episodes_length):
            
            # print("-"*25)
            print(f"THIS IS EVAL STEP {eval_step}")

            for i in range(self.eval_envs.num_envs):
                
                self.log_env(eval_step, i)
                
            self.eval_envs.step()
        
        # Evaluation over periods
        for i in range(self.eval_envs.num_envs):
            env = self.eval_envs.envs_map[i]
            stats[i]["platform_cost"] += env.platform_cost
            
            order_data = [{
                'orderid': order.orderid,
                'status': order.status,
                'distance_cluster': order.distance_cluster,
                'time_cluster': order.time_cluster,
                'order_create_time': order.order_create_time,
                'meal_prepare_time': order.meal_prepare_time,
                'pick_up_point': order.pick_up_point,
                'drop_off_point': order.drop_off_point,
                'ETA': order.ETA,
                'wait_time': order.wait_time,
                'price': order.price,
                'reject_count': order.reject_count,
                'pair_time': order.pair_time,
                'pair_courier': order.pair_courier.courier_id if order.pair_courier else None,
                'is_late': order.is_late,
                'ETA_usage': order.ETA_usage,
                'arrive_time': order.arrive_time,
                'type': order.pair_courier.courier_type if order.pair_courier else None
            } for order in env.orders]
            
            courier_data = [{
                'courier_id': courier.courier_id,
                'courier_type': courier.courier_type,
                'start_time': courier.start_time,
                'state': courier.state,
                
                'total_leisure_time': courier.total_leisure_time,
                'total_riding_time': courier.total_riding_time,
                'total_congestion_time': courier.total_congestion_time,
                'total_running_time': courier.total_running_time,
                
                'total_waiting_time': courier.total_waiting_time,
                
                'reject_order_num': courier.reject_order_num,
                'finish_order_num': courier.finish_order_num,
                
                'position': courier.position,
                'travel_distance': courier.travel_distance,
                
                'income': courier.income
            } for courier in env.couriers]


            df1 = pd.DataFrame(order_data)
            df1.to_csv(f'order_info{i}.csv', index=False)

            df2 = pd.DataFrame(courier_data)
            df2.to_csv(f'courier_info{i}.csv', index=False)


            for c in env.couriers:
                category = "Hired" if c.courier_type == 0 else "Crowdsourced"
                stats[i][f"{category}_num"] += 1

                if c.travel_distance > 0:
                    stats[i][f"{category}_distance_per_episode"].append(c.travel_distance)

                    stats[i][f"{category}_finish_num"].append(c.finish_order_num)
                    stats[i][f"{category}_unfinish_num"].append(len(c.waybill) + len(c.wait_to_pick))
                    stats[i][f"{category}_reject_num"].append(c.reject_order_num)
                    stats[i][f"{category}_leisure_time"].append(c.total_leisure_time)
                    stats[i][f"{category}_running_time"].append(c.total_running_time)
                    stats[i][f"{category}_waiting_time"].append(c.total_waiting_time)

                if c.actual_speed > 0:
                    stats[i][f"{category}_actual_speed"].append(c.actual_speed)

                if c.income > 0:
                    stats[i][f"{category}_income"].append(
                        c.income / (c.total_running_time + c.total_leisure_time) * 3600
                    )

                if category == "Crowdsourced" and c.state == "active":
                    stats[i]["Crowdsourced_on"] += 1

            stats[i]["courier_num"] = len(env.couriers)

            for o in env.orders:
                if o.status in {'wait_pair', 'wait_pick', 'picked_up'}:
                    stats[i]["count_unfinished_orders"] += 1
                    if o.ETA <= self.eval_envs.envs_map[0].clock:
                        stats[i]["unfinished_late_orders"] += 1
                        
                    if o.status == 'wait_pair':
                        stats[i]["order_wait_pair"] += 1
                    elif o.status == 'picked_up':
                        stats[i]["order_waiting_time"].append(o.wait_time)
                        stats[i]["order_price"].append(o.price)
                    else:
                        stats[i]["order_price"].append(o.price)
                        
                else:
                    stats[i]["count_dropped_orders"] += 1
                    if o.is_late == 1:
                        stats[i]["late_orders"] += 1
                    else:
                        stats[i]["ETA_usage"].append(o.ETA_usage)
                    stats[i]["order_waiting_time"].append(o.wait_time)
                    stats[i]["order_price"].append(o.price)
                
                if o.reject_count > 0:
                    stats[i]["count_reject_orders"] += 1
                    if stats[i]["max_reject_num"] <= o.reject_count:
                        stats[i]["max_reject_num"] = o.reject_count

            stats[i]["order_num"] = len(env.orders)
        
        message = ''
        for algo_num in range(self.eval_envs.num_envs):
            data = stats[algo_num]
            
            print(f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished), {data['order_wait_pair']} ({round(100 * data['order_wait_pair'] / data['order_num'], 2)}%) Orders waiting to be paired")

            # -----------------------
            # Distance
            hired_distance = np.mean(data["Hired_distance_per_episode"]) / 1000
            var_hired_distance = np.var(data["Hired_distance_per_episode"]) / 1000000
            crowdsourced_distance = np.mean(data["Crowdsourced_distance_per_episode"]) / 1000
            var_crowdsourced_distance = np.var(data["Crowdsourced_distance_per_episode"]) / 1000000
            total_distance = np.mean(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000
            var_total_distance = np.var(data["Hired_distance_per_episode"] + data["Crowdsourced_distance_per_episode"]) / 1000000
            total_courier_num = data['courier_num']

            print(f"In Algo{algo_num + 1}, Total couriers: {total_courier_num}")            
            print(f"\nIn Algo{algo_num + 1}, Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})")
            # -----------------------
            # Average Courier Finishing Number
            hired_finish_num = data["Hired_finish_num"]
            crowdsourced_finish_num = data["Crowdsourced_finish_num"]
            
            finish0 = np.mean(hired_finish_num)
            var0_finish = np.var(hired_finish_num)
            finish1 = np.mean(crowdsourced_finish_num)
            var1_finish = np.var(crowdsourced_finish_num)
            total_finish = np.mean(hired_finish_num + crowdsourced_finish_num)
            var_finish = np.var(hired_finish_num + crowdsourced_finish_num)

            print(f"Average Finished Orders per Courier for Algo{algo_num + 1}:")
            print(f"Hired finishes average {finish0} orders (Var: {var0_finish}), Crowdsourced finishes average {finish1} orders (Var: {var1_finish}), Total finish number per courier is {total_finish} orders (Var: {var_finish})")

                    
            # -----------------------
            # Average Courier unfinished Number
            hired_unfinish_num = data["Hired_unfinish_num"]
            crowdsourced_unfinish_num = data["Crowdsourced_unfinish_num"]
            
            unfinish0 = np.mean(hired_unfinish_num)
            var0_unfinish = np.var(hired_unfinish_num)
            unfinish1 = np.mean(crowdsourced_unfinish_num)
            var1_unfinish = np.var(crowdsourced_unfinish_num)
            total_unfinish = np.mean(hired_unfinish_num + crowdsourced_unfinish_num)
            var_unfinish = np.var(hired_unfinish_num + crowdsourced_unfinish_num)

            print(f"Average unfinished Orders per Courier for Algo{algo_num+1}:")
            print(f"Hired unfinishes average {unfinish0} orders (Var: {var0_unfinish}), Crowdsourced unfinishes average {unfinish1} orders (Var: {var1_unfinish}), Total unfinish number per courier is {total_unfinish} orders (Var: {var_unfinish})")


            # ---------------------
            # courier reject number
            Hired_reject_num = data['Hired_reject_num']
            Crowdsourced_reject_num = data['Crowdsourced_reject_num']
            avg_reject0 = np.mean(Hired_reject_num)
            var_reject0 = np.var(Hired_reject_num)
            avg_reject1 = np.mean(Crowdsourced_reject_num)
            var_reject1 = np.var(Crowdsourced_reject_num)
            avg_reject = np.mean(Hired_reject_num + Crowdsourced_reject_num)
            var_reject = np.var(Hired_reject_num + Crowdsourced_reject_num)
            print(
                f"The average rejection number for Algo{algo_num+1}: Hired - {avg_reject0} (Var: {var_reject0}), "
                f"Crowdsourced - {avg_reject1} (Var: {var_reject1}), "
                f"Total - {avg_reject} (Var: {var_reject})"
            )

            # -----------------------
            # Average Courier Leisure Time
            Hired_leisure_time = data['Hired_leisure_time']
            Crowdsourced_leisure_time = data['Crowdsourced_leisure_time']
            
            hired_leisure = np.mean(Hired_leisure_time) / 60
            hired_leisure_var = np.var(Hired_leisure_time) / 60**2
            Crowdsourced_leisure = np.mean(Crowdsourced_leisure_time) / 60
            Crowdsourced_leisure_var = np.var(Crowdsourced_leisure_time) / 60**2
            
            avg_leisure = np.mean(Hired_leisure_time + Crowdsourced_leisure_time) / 60
            avg_leisure_var = np.var(Hired_leisure_time + Crowdsourced_leisure_time) / 60**2

            print(f"Average leisure time per courier for Algo {algo_num+1}:")
            print(f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})")


            # -----------------------
            # Average Courier running Time
            Hired_running_time = data['Hired_running_time']
            Crowdsourced_running_time = data['Crowdsourced_running_time']
            
            hired_running = np.mean(Hired_running_time) / 60
            hired_running_var = np.var(Hired_running_time) / 60**2
            Crowdsourced_running = np.mean(Crowdsourced_running_time) / 60
            Crowdsourced_running_var = np.var(Crowdsourced_running_time) / 60**2
            
            avg_running = np.mean(Hired_running_time + Crowdsourced_running_time) / 60
            avg_running_var = np.var(Hired_running_time + Crowdsourced_running_time) / 60**2

            print(f"Average running time per courier for Algo {algo_num+1}:")
            print(f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})")
        
            # -----------------------
            # Average Courier waiting Time
            Hired_waiting_time = data['Hired_waiting_time']
            Crowdsourced_waiting_time = data['Crowdsourced_waiting_time']
            
            hired_waiting = np.mean(Hired_waiting_time) / 60
            hired_waiting_var = np.var(Hired_waiting_time) / 60**2
            Crowdsourced_waiting = np.mean(Crowdsourced_waiting_time) / 60
            Crowdsourced_waiting_var = np.var(Crowdsourced_waiting_time) / 60**2
            
            avg_waiting = np.mean(Hired_waiting_time + Crowdsourced_waiting_time) / 60
            avg_waiting_var = np.var(Hired_waiting_time + Crowdsourced_waiting_time) / 60**2

            print(f"Average waiting time per courier for Algo {algo_num+1}:")
            print(f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})")
            
            # -----------------------
            # Actual Speed
            Hired_actual_speed = data['Hired_actual_speed']
            Crowdsourced_actual_speed = data['Crowdsourced_actual_speed']

            hired_speed = np.mean(Hired_actual_speed)
            hired_speed_var = np.var(Hired_actual_speed)
            crowdsourced_speed = np.mean(Crowdsourced_actual_speed)
            crowdsourced_speed_var = np.var(Crowdsourced_actual_speed)
            avg_speed = np.mean(Hired_actual_speed + Crowdsourced_actual_speed)
            avg_speed_var = np.var(Hired_actual_speed + Crowdsourced_actual_speed)

            print(f"Average speed per courier for Algo{algo_num+1}:")
            print(f"Hired average speed is {hired_speed} m/s (Var: {hired_speed_var}), Crowdsourced average speed is {crowdsourced_speed} m/s (Var: {crowdsourced_speed_var}), Total average speed per courier is {avg_speed} m/s (Var: {avg_speed_var})")
            # -----------------------
            # Average Courier Income
            hired_income = data['Hired_income']
            crowdsourced_income = data['Crowdsourced_income']
            hired_income = np.mean(hired_income)
            crowdsourced_income = np.mean(crowdsourced_income)
            total_income = np.mean(hired_income + crowdsourced_income)
            hired_income_var = np.var(hired_income)
            crowdsourced_income_var = np.var(crowdsourced_income)
            total_income_var = np.var(hired_income + crowdsourced_income)

            print(f"Average Income per Courier for Algo{algo_num+1}:")
            print(f"Total: Hired's average income is {hired_income} dollars (Var: {hired_income_var}), Crowdsourced's average income is {crowdsourced_income} dollars (Var: {crowdsourced_income_var}), Total income per courier is {total_income} dollars (Var: {total_income_var})")
            
            # -----------------------
            # Platform cost
            platform_cost = data['platform_cost']
            print(f"The platform cost for Algo{algo_num+1} is {platform_cost} dollars.")
            
            # ---------------------
            # order reject rate
            reject_rate_per_episode = data['count_reject_orders'] / data['order_num'] # reject once or twice or more
            print(f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {data['max_reject_num']} times at most")
            
            # ---------------------
            # average waiting time for orders
            waiting_time_per_order = np.mean(data['order_waiting_time']) / 60
            var_waiting_time = np.var(data['order_waiting_time']) / 60**2
            print(f"The average waiting time for orders ({data['order_num'] - data['order_wait_pair']}) is {waiting_time_per_order} minutes (Var: {var_waiting_time})")

            # -----------------------
            # Average Order Price
            order_price = data['order_price']

            order_price_per_order = np.mean(order_price)
            order_price_var = np.var(order_price)

            print(f"Average Price per Order for Algo{algo_num+1}:")
            print(f"Total average is {order_price_per_order} dollars (Var: {order_price_var})")

            message += (
                f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished), {data['order_wait_pair']} ({round(100 * data['order_wait_pair'] / data['order_num'], 2)}%) Orders waiting to be paired\n"
                                
                f"Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})\n"
                
                f"Hired finishes average {finish0} orders (Var: {var0_finish}), Crowdsourced finishes average {finish1} orders (Var: {var1_finish}), Total finish number per courier is {total_finish} orders (Var: {var_finish})\n"
                
                f"Hired unfinishes average {unfinish0} orders (Var: {var0_unfinish}), Crowdsourced unfinishes average {unfinish1} orders (Var: {var1_unfinish}), Total unfinish number per courier is {total_unfinish} orders (Var: {var_unfinish})\n"
                
                f"Hired ones reject {avg_reject0} times (Var: {var_reject0}), Crowdsourced ones reject {avg_reject1} times (Var: {var_reject1}) and the total reject {avg_reject} times (Var: {var_reject}\n"
                
                f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})\n"
                
                f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})\n"
                
                f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})\n"
                
                f"Hired average speed is {hired_speed} m/s (Var: {hired_speed_var}), Crowdsourced average speed is {crowdsourced_speed} m/s (Var: {crowdsourced_speed_var}), Total average speed per courier is {avg_speed} m/s (Var: {avg_speed_var})\n"
                
                f"Total: Hired's average income is {hired_income} dollars (Var: {hired_income_var}), Crowdsourced's average income is {crowdsourced_income} dollars (Var: {crowdsourced_income_var}), Total income per courier is {total_income} dollars (Var: {total_income_var})\n"
                
                f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {data['max_reject_num']} times at most\n"
                
                f"The average waiting time for orders ({data['order_num'] - data['order_wait_pair']}) is {waiting_time_per_order} minutes (Var: {var_waiting_time})\n"
                        
                f"Total average price per order is {order_price_per_order} dollars (Var: {order_price_var})\n"
                                
                f"The platform total cost is {platform_cost} dollar\n"
                
            )

            if data['count_dropped_orders'] == 0:
                print(f"No order is dropped in Algo{algo_num+1}")
                
                message += f"No order is dropped in Algo{algo_num+1}\n"
            else:                
                late_rate = data['late_orders'] / data['count_dropped_orders']     
                ETA_usage_rate = np.mean(data['ETA_usage'])
                var_ETA = np.var(data['ETA_usage'])
                print(f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders")
                print(f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})")
                
                message += f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders\n" + f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})\n"
            
            if data['count_unfinished_orders'] == 0:
                print(f"No order is unfinished in Algo{algo_num+1}")
                message += f"No order is unfinished in Algo{algo_num+1}\n"
            else:
                unfinished = data['count_unfinished_orders'] / data['order_num']
                unfinished_late_rate = data['unfinished_late_orders'] / data['count_unfinished_orders']
                print(f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num']} orders ({unfinished}), with {unfinished_late_rate} being late")
                
                message += f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num']} orders ({unfinished}), with {unfinished_late_rate} being late\n"
  
            print("\n")
            
        logger.success(message)
        self.eval_envs.close()