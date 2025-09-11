import random


class MUDAlgorithm:
    def __init__(self, power_limit, tasks, edge_servers, power_function):
        self.power_limit = power_limit
        self.tasks = tasks
        self.edge_servers = edge_servers
        self.power_function = power_function
        self.Y = {}

    def run(self):
        X_tmp = {task.id: 0 for task in self.tasks}
        P_used = 0
        S_task = set(task.id for task in self.tasks)
        self.Y = {es.id: 0 for es in self.edge_servers}

        while S_task:
            best_valgo = -float('inf')
            best_task_id = None
            best_es_id = None
            best_palgo = 0
            for task in self.tasks:
                if task.id not in S_task:
                    continue
                for es in self.edge_servers:
                    if task.coverage[es.id] == 0:
                        continue
                    current_usage = sum(
                        t.usage for t in self.tasks if X_tmp[t.id] == es.id)
                    current_util = current_usage / es.capacity if es.capacity > 0 else 0
                    before_power = self.power_function(es, current_util)
                    after_util = (current_usage + task.usage) / \
                        es.capacity if es.capacity > 0 else 0
                    after_power = self.power_function(es, after_util)
                    P_algo = after_power - before_power
                    if P_algo <= 0:
                        continue
                    V_algo = task.rewards[es.id] / P_algo
                    if V_algo > best_valgo and (P_used + P_algo) <= self.power_limit:
                        best_valgo = V_algo
                        best_task_id = task.id
                        best_es_id = es.id
                        best_palgo = P_algo
            if best_task_id is not None:
                P_used += best_palgo
                X_tmp[best_task_id] = best_es_id
                S_task.remove(best_task_id)
            else:
                break
        for es in self.edge_servers:
            self.Y[es.id] = sum(
                t.usage for t in self.tasks if X_tmp[t.id] == es.id)
        return self.Y
