import time
import numpy as np
import torch

from runner.base_runner import Runner
import matplotlib.pyplot as plt

from loguru import logger
import numpy as np

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
            "Hired_congestion_time": [],
            "Hired_waiting_time": [],
            "Crowdsourced_finish_num": [],
            "Crowdsourced_unfinish_num": [],
            "Crowdsourced_reject_num": [],
            "Crowdsourced_leisure_time": [],
            "Crowdsourced_running_time": [],
            "Crowdsourced_congestion_time": [],
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
            "overspeed_step": {"ratio0": [], "ratio1": []}, 
            
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
            "order_wait": 0,
            "order_waiting_time": [],
        } for i in range(self.eval_envs.num_envs)}

        for eval_step in range(self.eval_episodes_length):
            
            # print("-"*25)
            print(f"THIS IS EVAL STEP {eval_step}")

            for i in range(self.eval_envs.num_envs):
                
                self.log_env(eval_step, i)
                            
            self.eval_envs.step()
                        
            algo_stats = {i: {"num0": 0, "num1": 0, "count0": 0, "count1": 0} for i in range(self.eval_envs.num_envs)}

            for i in range(self.eval_envs.num_envs):
                for c in self.eval_envs.envs_map[i].active_couriers:
                    if c.state == 'active':
                        if c.courier_type == 0:
                            algo_stats[i]["num0"] += 1
                            if c.speed > 4:
                                algo_stats[i]["count0"] += 1
                        else:
                            algo_stats[i]["num1"] += 1
                            if c.speed > 4:
                                algo_stats[i]["count1"] += 1

            for i in range(self.eval_envs.num_envs):
                overspeed_ratio0 = algo_stats[i]["count0"] / algo_stats[i]["num0"] if algo_stats[i]["num0"] > 0 else 0
                overspeed_ratio1 = algo_stats[i]["count1"] / algo_stats[i]["num1"] if algo_stats[i]["num1"] > 0 else 0

                stats[i]["overspeed_step"]["ratio0"].append(overspeed_ratio0)
                stats[i]["overspeed_step"]["ratio1"].append(overspeed_ratio1) 
                                    
            self.eval_envs.eval_env_step()
            
        # Evaluation over periods
        for i in range(self.eval_envs.num_envs):
            env = self.eval_envs.envs_map[i]
            stats[i]["platform_cost"] += env.platform_cost

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
                stats[i][f"{category}_congestion_time"].append(c.total_congestion_time)
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
                if o.status == "dropped":
                    stats[i]["count_dropped_orders"] += 1
                    if o.is_late == 1:
                        stats[i]["late_orders"] += 1
                    else:
                        stats[i]["ETA_usage"].append(o.ETA_usage)
                else:
                    stats[i]["count_unfinished_orders"] += 1
                    if o.ETA <= self.envs.envs_map[0].clock:
                        stats[i]["unfinished_late_orders"] += 1
                        
                if o.reject_count > 0:
                    stats[i]["count_reject_orders"] += 1
                    if stats[i]["max_reject_num"] <= o.reject_count:
                        stats[i]["max_reject_num"] = o.reject_count

                if o.status == "wait_pair":
                    stats[i]["order_wait"] += 1
                else:
                    stats[i]["order_waiting_time"].append(o.wait_time)
                    stats[i]["order_price"].append(o.price)

            stats[i]["order_num"] = len(env.orders)
            
        for algo_num in range(self.eval_envs.num_envs):
            data = stats[algo_num]
            
            print(f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished, {data['order_wait']} ({round(100 * data['order_wait'] / data['order_num'], 2)}%) Orders waiting to be paired)")

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
            
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired', hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced', crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total', total_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Hired Var', var_hired_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Crowdsourced Var', var_crowdsourced_distance, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num + 1}/Eval Travel Distance/Total Var', var_total_distance, self.eval_num)
            
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

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Total', total_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Hired', finish0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Crowdsourced', finish1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Total Var', var_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Hired Var', var0_finish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Finish/Crowdsourced Var', var1_finish, self.eval_num)
                    
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

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Total', total_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Hired', unfinish0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Crowdsourced', unfinish1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Total Var', var_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Hired Var', var0_unfinish, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average unfinish/Crowdsourced Var', var1_unfinish, self.eval_num)

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
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Total', avg_reject, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Total_Var', var_reject, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Hired', avg_reject0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Hired_Var', var_reject0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Crowdsourced', avg_reject1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject Rate/Crowdsourced_Var', var_reject1, self.eval_num)

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
            leisure_courier_num = len(Hired_leisure_time + Crowdsourced_leisure_time)

            print(f"Average leisure time per courier for Algo {algo_num+1}:")
            print(f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Total', avg_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Hired', hired_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Crowdsourced', Crowdsourced_leisure, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Total Var', avg_leisure_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Hired Var', hired_leisure_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Leisure Time/Crowdsourced Var', Crowdsourced_leisure_var, self.eval_num)

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
            running_courier_num = len(Hired_running_time + Crowdsourced_running_time)

            print(f"Average running time per courier for Algo {algo_num+1}:")
            print(f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total', avg_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired', hired_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced', Crowdsourced_running, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Total Var', avg_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Hired Var', hired_running_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average running Time/Crowdsourced Var', Crowdsourced_running_var, self.eval_num)

            # -----------------------
            # Average Courier congestion Time
            Hired_congestion_time = data['Hired_congestion_time']
            Crowdsourced_congestion_time = data['Crowdsourced_congestion_time']
            
            hired_congestion = np.mean(Hired_congestion_time) / 60
            hired_congestion_var = np.var(Hired_congestion_time) / 60**2
            Crowdsourced_congestion = np.mean(Crowdsourced_congestion_time) / 60
            Crowdsourced_congestion_var = np.var(Crowdsourced_congestion_time) / 60**2
            
            avg_congestion = np.mean(Hired_congestion_time + Crowdsourced_congestion_time) / 60
            avg_congestion_var = np.var(Hired_congestion_time + Crowdsourced_congestion_time) / 60**2
            congestion_courier_num = len(Hired_congestion_time + Crowdsourced_congestion_time)

            print(f"Average congestion time per courier for Algo {algo_num+1}:")
            print(f"Hired congestion time is {hired_congestion} minutes (Var: {hired_congestion_var}), Crowdsourced congestion time is {Crowdsourced_congestion} minutes (Var: {Crowdsourced_congestion_var}), Total congestion time per courier is {avg_congestion} minutes (Var: {avg_congestion_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Total', avg_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Hired', hired_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Crowdsourced', Crowdsourced_congestion, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Total Var', avg_congestion_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Hired Var', hired_congestion_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average congestion Time/Crowdsourced Var', Crowdsourced_congestion_var, self.eval_num)
        
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
            waiting_courier_num = len(Hired_waiting_time + Crowdsourced_waiting_time)

            print(f"Average waiting time per courier for Algo {algo_num+1}:")
            print(f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Total', avg_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Hired', hired_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Crowdsourced', Crowdsourced_waiting, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Total Var', avg_waiting_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Hired Var', hired_waiting_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average waiting Time/Crowdsourced Var', Crowdsourced_waiting_var, self.eval_num)
            
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

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Total', avg_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Hired', hired_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Crowdsourced', crowdsourced_speed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Total Var', avg_speed_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Hired Var', hired_speed_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Speed/Crowdsourced Var', crowdsourced_speed_var, self.eval_num)
            
            # -----------------------
            # Overspeed
            Hired_overspeed = data['Hired_overspeed']
            Crowdsourced_overspeed = data['Crowdsourced_overspeed']
            hired_overspeed = np.mean(Hired_overspeed)
            crowdsourced_overspeed = np.mean(Crowdsourced_overspeed)
            total_overspeed = np.mean(Hired_overspeed + Crowdsourced_overspeed)

            print(f"Rate of Overspeed for Evaluation for Algo{algo_num+1}:")
            print(f"Hired overspeed rate is {hired_overspeed}, Crowdsourced overspeed rate is {crowdsourced_overspeed}, Total overspeed rate is {total_overspeed}")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Total', total_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Hired', hired_overspeed, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Overspeed Rate/Crowdsourced', crowdsourced_overspeed, self.eval_num)

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

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Total', total_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Hired', hired_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Crowdsourced', crowdsourced_income, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Total Var', total_income_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Hired Var', hired_income_var, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Income/Crowdsourced Var', crowdsourced_income_var, self.eval_num)
            
            # -----------------------
            # Platform cost
            platform_cost = data['platform_cost']
            print(f"The platform cost for Algo{algo_num+1} is {platform_cost} dollars.")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Platform Cost', platform_cost, self.eval_num)
            
            # ---------------------
            # order reject rate
            reject_rate_per_episode = data['count_reject_orders'] / data['order_num'] # reject once or twice or more
            print(f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {data['max_reject_num']} times at most")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Reject rate', reject_rate_per_episode, self.eval_num)
            
            # ---------------------
            # average waiting time for orders
            waiting_time_per_order = np.mean(data['order_waiting_time'])
            var_waiting_time = np.var(data['order_waiting_time'])
            print(f"The average waiting time for orders ({data['order_num'] - data['order_wait']}) is {waiting_time_per_order} dollar (Var: {var_waiting_time})")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Order Waiting Time/Total', waiting_time_per_order, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Order Waiting Time/Total_Var', var_waiting_time, self.eval_num)

            # -----------------------
            # Average Order Price
            order_price = data['order_price']

            order_price_per_order = np.mean(order_price)
            order_price_var = np.var(order_price)

            print(f"Average Price per Order for Algo{algo_num+1}:")
            print(f"Total average is {order_price_per_order} dollars (Var: {order_price_var})")

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Price/Total', order_price_per_order, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Average Price/Total Var', order_price_var, self.eval_num)
            
            message = (
                f"\nIn Algo{algo_num + 1} there are {data['Hired_num']} Hired, {data['Crowdsourced_num']} Crowdsourced with {data['Crowdsourced_on']} ({round(100 * data['Crowdsourced_on'] / data['Crowdsourced_num'], 2)}%) on, and {data['order_num']} Orders, ({data['count_dropped_orders']} dropped, {data['count_unfinished_orders']} unfinished, {data['order_wait']} ({round(100 * data['order_wait'] / data['order_num'], 2)}%) Orders waiting to be paired)\n"
                                
                f"Hired total distance: {hired_distance} km (Var: {var_hired_distance}), Crowdsourced total distance: {crowdsourced_distance} km (Var: {var_crowdsourced_distance}), Total distance: {total_distance} km (Var: {var_total_distance})\n"
                
                f"Hired finishes average {finish0} orders (Var: {var0_finish}), Crowdsourced finishes average {finish1} orders (Var: {var1_finish}), Total finish number per courier is {total_finish} orders (Var: {var_finish})\n"
                
                f"Hired unfinishes average {unfinish0} orders (Var: {var0_unfinish}), Crowdsourced unfinishes average {unfinish1} orders (Var: {var1_unfinish}), Total unfinish number per courier is {total_unfinish} orders (Var: {var_unfinish})\n"
                
                f"Hired ones reject {avg_reject0} times (Var: {var_reject0}), Crowdsourced ones reject {avg_reject1} times (Var: {var_reject1}) and the total reject {avg_reject} times (Var: {var_reject}\n"
                
                f"Hired leisure time is {hired_leisure} minutes (Var: {hired_leisure_var}), Crowdsourced leisure time is {Crowdsourced_leisure} minutes (Var: {Crowdsourced_leisure_var}), Total leisure time per courier is {avg_leisure} minutes (Var: {avg_leisure_var})\n"
                
                f"Hired running time is {hired_running} minutes (Var: {hired_running_var}), Crowdsourced running time is {Crowdsourced_running} minutes (Var: {Crowdsourced_running_var}), Total running time per courier is {avg_running} minutes (Var: {avg_running_var})\n"

                f"Hired congestion time is {hired_congestion} minutes (Var: {hired_congestion_var}), Crowdsourced congestion time is {Crowdsourced_congestion} minutes (Var: {Crowdsourced_congestion_var}), Total congestion time per courier is {avg_congestion} minutes (Var: {avg_congestion_var})\n"
                
                f"Hired waiting time is {hired_waiting} minutes (Var: {hired_waiting_var}), Crowdsourced waiting time is {Crowdsourced_waiting} minutes (Var: {Crowdsourced_waiting_var}), Total waiting time per courier is {avg_waiting} minutes (Var: {avg_waiting_var})\n"
                
                f"Hired average speed is {hired_speed} m/s (Var: {hired_speed_var}), Crowdsourced average speed is {crowdsourced_speed} m/s (Var: {crowdsourced_speed_var}), Total average speed per courier is {avg_speed} m/s (Var: {avg_speed_var})\n"

                f"Hired overspeed rate is {hired_overspeed}, Crowdsourced overspeed rate is {crowdsourced_overspeed}, Total overspeed rate is {total_overspeed}\n"     
                
                f"Total: Hired's average income is {hired_income} dollars (Var: {hired_income_var}), Crowdsourced's average income is {crowdsourced_income} dollars (Var: {crowdsourced_income_var}), Total income per courier is {total_income} dollars (Var: {total_income_var})\n"
                
                f"The rejection rate is {reject_rate_per_episode} and the order is rejected by {data['max_reject_num']} times at most\n"
                
                f"The average waiting time for orders ({data['order_num'] - data['order_wait']}) is {waiting_time_per_order} dollar (Var: {var_waiting_time})\n"
                           
                f"Total average is {order_price_per_order} dollars (Var: {order_price_var})\n"
                                
                f"The platform1 total cost is {platform_cost} dollar\n"
                
            )

        if data['count_dropped_orders'] == 0:
            print(f"No order is dropped in Algo{algo_num+1}")
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', -1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', -1, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var/Crowdsourced Var', 0, self.eval_num)
            
            message += "No order is dropped in Algo1\n"
        else:                
            late_rate = data['late_orders'] / data['count_dropped_orders']     
            ETA_usage_rate = np.mean(data['ETA_usage0'])
            var_ETA = np.var(data['ETA_usage0'])
            print(f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders")
            print(f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})")
            
            message += f"Rate of Late Orders is {late_rate} out of {data['count_dropped_orders']} orders\n" + f"Rate of ETA Usage is {ETA_usage_rate} (Var: {var_ETA})\n"

            self.writter.add_scalar(f'Algo{algo_num+1}/Eval Late Order Rate', late_rate, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate', ETA_usage_rate, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Eval ETA Usage Rate Var', var_ETA, self.eval_num)
        
        if data['count_unfinished_orders'] == 0:
            print(f"No order is unfinished in Algo{algo_num+1}")
            message += f"No order is unfinished in Algo{algo_num+1}\n"
            logger.success(message)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', 0, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', 0, self.eval_num)
        else:
            unfinished = data['count_unfinished_orders'] / (data['order_num'] - data['order_wait'])
            unfinished_late_rate = data['unfinished_late_orders'] / data['count_unfinished_orders']
            print(f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num'] - data['order_wait']} orders ({unfinished}), with {unfinished_late_rate} being late")
            
            message += f"Unfinished Orders in Algo{algo_num+1} is {data['count_unfinished_orders']} out of {data['order_num'] - data['order_wait']} orders ({unfinished}), with {unfinished_late_rate} being late\n"
            logger.success(message)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Orders Rate', unfinished, self.eval_num)
            self.writter.add_scalar(f'Algo{algo_num+1}/Unfinished Late Rate', unfinished_late_rate, self.eval_num)
        
        logger.success(message)
            
        print("\n")
        
        self.eval_envs.close()