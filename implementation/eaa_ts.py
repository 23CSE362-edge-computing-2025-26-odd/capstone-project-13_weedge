class EAA_TS:
    """
    Approximate time-sensitive allocation (greedy but allows splitting tasks across servers).

    For each task, try to allocate its usage starting from the server with highest
    reward-per-unit that has coverage and residual capacity. This is fast and
    produces split allocations (a list of (es_id, flow) per task) while respecting
    the per-server capacity Y.
    """

    def __init__(self, tasks, edge_servers, Y):
        self.tasks = tasks
        self.edge_servers = edge_servers
        self.Y = dict(Y)  # remaining capacity map

    def run(self):
        allocation = {task.id: [] for task in self.tasks}
        residual = {es.id: self.Y.get(es.id, 0) for es in self.edge_servers}

        # Sort tasks by max reward-per-unit descending (prioritize valuable tasks)
        sorted_tasks = sorted(self.tasks, key=lambda t: max(t.rewards.values()) / t.usage if t.usage > 0 else 0, reverse=True)

        for task in sorted_tasks:
            remaining = task.usage
            # build list of candidate servers sorted by reward-per-unit for this task
            candidates = [(es.id, (task.rewards.get(es.id, 0) / task.usage) if task.usage > 0 else 0) for es in self.edge_servers if task.coverage.get(es.id, 0) == 1]
            candidates.sort(key=lambda x: x[1], reverse=True)
            for es_id, rpu in candidates:
                if remaining <= 0:
                    break
                avail = residual.get(es_id, 0)
                if avail <= 0:
                    continue
                take = min(avail, remaining)
                allocation[task.id].append((es_id, take))
                residual[es_id] -= take
                remaining -= take
            # If not fully allocated, mark leftover as unassigned by leaving remaining > 0
            if remaining > 0:
                # we could record unallocated portion by appending (0, remaining) or keep as is
                pass

        return allocation
