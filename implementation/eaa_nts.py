class EAA_NTS:
    def __init__(self, tasks, edge_servers, Y):
        self.tasks = tasks
        self.edge_servers = edge_servers
        self.Y = Y

    def run(self):
        allocation = {}
        residual_capacity = {es.id: self.Y.get(
            es.id, 0) for es in self.edge_servers}
        sorted_tasks = sorted(self.tasks, key=lambda t: max(
            t.rewards.values()) if t.rewards else 0, reverse=True)
        for task in sorted_tasks:
            allocated = False
            sorted_es = sorted(
                self.edge_servers, key=lambda es: task.rewards.get(es.id, 0), reverse=True)
            for es in sorted_es:
                if task.coverage[es.id] == 1 and residual_capacity[es.id] >= task.usage:
                    allocation[task.id] = es.id
                    residual_capacity[es.id] -= task.usage
                    allocated = True
                    break
            if not allocated:
                allocation[task.id] = 0
        return allocation
