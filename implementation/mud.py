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
        # diagnostics per server
        diag = {es.id: {'coverage_skipped': 0, 'p_algo_nonpos': 0, 'budget_blocked': 0, 'candidates': 0, 'assigned': 0} for es in self.edge_servers}

        # maintain current usage per server to avoid repeated sums
        current_usage = {es.id: 0 for es in self.edge_servers}
        # fast lookup for task objects by id
        tasks_by_id = {t.id: t for t in self.tasks}

        while S_task:
            best_valgo = -float('inf')
            best_task_id = None
            best_es_id = None
            best_palgo = 0
            for task in self.tasks:
                if task.id not in S_task:
                    continue
                for es in self.edge_servers:
                    diag[es.id]['candidates'] += 1
                    if task.coverage.get(es.id, 0) == 0:
                        diag[es.id]['coverage_skipped'] += 1
                        continue
                    # use incremental current_usage
                    current_util = current_usage.get(es.id, 0) / es.capacity if es.capacity else 0
                    before_power = self.power_function(es, current_util)
                    after_util = (current_usage.get(es.id, 0) + task.usage) / \
                        es.capacity if es.capacity else 0
                    after_power = self.power_function(es, after_util)
                    P_algo = after_power - before_power
                    if P_algo <= 0:
                        diag[es.id]['p_algo_nonpos'] += 1
                        continue
                    if (P_used + P_algo) > self.power_limit:
                        diag[es.id]['budget_blocked'] += 1
                        continue

                    V_algo = task.rewards[es.id] / P_algo if P_algo != 0 else 0
                    if V_algo > best_valgo:
                        best_valgo = V_algo
                        best_task_id = task.id
                        best_es_id = es.id
                        best_palgo = P_algo

            if best_task_id is not None:
                P_used += best_palgo
                X_tmp[best_task_id] = best_es_id
                S_task.remove(best_task_id)
                diag[best_es_id]['assigned'] += 1
                # update incremental usage for the server
                # find the task object
                task_obj = tasks_by_id.get(best_task_id)
                if task_obj is not None:
                    current_usage[best_es_id] = current_usage.get(best_es_id, 0) + task_obj.usage
            else:
                break

        for es in self.edge_servers:
            self.Y[es.id] = sum(
                t.usage for t in self.tasks if X_tmp[t.id] == es.id)

        assigned_count = len(self.tasks) - len(S_task)
        failed_count = len(S_task)
        print(f"Allocated tasks: {assigned_count}, Failed: {failed_count}")
        return self.Y
