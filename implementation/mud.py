def mud_algorithm(tasks, es_dict, power_budget):
    unassigned_tasks = set(tasks)
    for es in es_dict.values():
        es.current_utilization = 0
    total_power = sum([es.power(0) for es in es_dict.values()])
    while unassigned_tasks and total_power <= power_budget:
        best = None
        best_efficiency = -float('inf')
        for task in unassigned_tasks:
            for es_name in task.candidates:
                es = es_dict[es_name]
                next_util = es.current_utilization + task.cpu
                if next_util > es.capacity:
                    continue
                p_before = es.power(es.current_utilization / es.capacity)
                p_after = es.power(next_util / es.capacity)
                delta_power = p_after - p_before
                if delta_power <= 0:
                    delta_power = 1e-3
                efficiency = task.reward / delta_power
                other_power = total_power - p_before
                if other_power + p_after > power_budget:
                    continue
                if efficiency > best_efficiency:
                    best = (task, es)
                    best_efficiency = efficiency
        if best is None:
            break
        task, es = best
        es.current_utilization += task.cpu
        total_power = sum([srv.power(srv.current_utilization / srv.capacity) for srv in es_dict.values()])
        unassigned_tasks.remove(task)
    for es in es_dict.values():
        es.max_utilization = es.current_utilization
        es.current_utilization = 0
