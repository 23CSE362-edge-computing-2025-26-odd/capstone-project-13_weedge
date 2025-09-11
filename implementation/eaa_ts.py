import networkx as nx


class EAA_TS:
    def __init__(self, tasks, edge_servers, Y):
        self.tasks = tasks
        self.edge_servers = edge_servers
        self.Y = Y

    def power_function(self, es, utilization):
        coeffs = es.power_active_coeffs
        utils = sorted(coeffs.keys())
        if utilization <= utils[0]:
            alpha = coeffs[utils[0]]
        elif utilization >= utils[-1]:
            alpha = coeffs[utils[-1]]
        else:
            for i in range(len(utils)-1):
                if utils[i] <= utilization <= utils[i+1]:
                    u1, u2 = utils[i], utils[i+1]
                    a1, a2 = coeffs[u1], coeffs[u2]
                    alpha = a1 + (a2 - a1) * (utilization - u1) / (u2 - u1)
                    break
        return alpha * utilization + es.power_idle * (1 - utilization)

    def build_mcmf_graph(self):
        G = nx.DiGraph()
        total_task_usage = sum(task.usage for task in self.tasks)
        G.add_node('source', demand=-total_task_usage)
        G.add_node('sink', demand=total_task_usage)
        for task in self.tasks:
            task_node = f'task_{task.id}'
            G.add_node(task_node, demand=0)
            G.add_edge('source', task_node, capacity=task.usage, weight=0)
        for es in self.edge_servers:
            es_node = f'es_{es.id}'
            G.add_node(es_node, demand=0)
            G.add_edge(es_node, 'sink',
                       capacity=self.Y.get(es.id, 0), weight=0)
        for task in self.tasks:
            task_node = f'task_{task.id}'
            for es in self.edge_servers:
                if task.coverage[es.id] == 1:
                    es_node = f'es_{es.id}'
                    reward_per_unit = task.rewards[es.id] / \
                        task.usage if task.usage > 0 else 0
                    G.add_edge(task_node, es_node,
                               capacity=task.usage, weight=-reward_per_unit)
        return G

    def run(self):
        G = self.build_mcmf_graph()
        try:
            flow_dict = nx.min_cost_flow(G)
        except Exception as e:
            return {task.id: [] for task in self.tasks}
        allocation = {task.id: [] for task in self.tasks}
        for task in self.tasks:
            task_node = f'task_{task.id}'
            for es in self.edge_servers:
                es_node = f'es_{es.id}'
                flow = flow_dict.get(task_node, {}).get(es_node, 0)
                if flow > 0:
                    allocation[task.id].append((es.id, flow))
        return allocation
