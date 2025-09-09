import networkx as nx
import matplotlib.pyplot as plt


def build_mcmf(tasks, es_dict, splitting=True):
    G = nx.DiGraph()
    G.add_node('src')
    G.add_node('sink')
    for task in tasks:
        G.add_edge('src', f"T{task.id}", capacity=task.cpu, weight=0)
        for es_name in task.candidates:
            reward_per_unit = task.reward / task.cpu
            G.add_edge(f"T{task.id}", es_name,
                       capacity=task.cpu if splitting else task.cpu, weight=-reward_per_unit)
    for es in es_dict.values():
        G.add_edge(es.name, 'sink', capacity=int(
            round(es.max_utilization)), weight=0)
    return G


def solve_mcmf(G):
    flow_dict = nx.max_flow_min_cost(G, 'src', 'sink')
    allocations = {}
    for t_node in flow_dict['src']:
        if not t_node.startswith('T'):
            continue
        task_id = int(t_node[1:])
        allocations[task_id] = {}
        for es_name, units in flow_dict[t_node].items():
            if units > 0:
                allocations[task_id][es_name] = units
    return allocations


def plot_graph(G, title="Allocation Graph"):
    pos = nx.spring_layout(G)
    node_colors = []
    for node in G.nodes:
        if node == 'src':
            node_colors.append('green')
        elif node == 'sink':
            node_colors.append('red')
        elif node.startswith('T'):
            node_colors.append('gold')
        else:
            node_colors.append('skyblue')
    edge_labels = {(u, v): f"cap={d['capacity']},w={
        d['weight']}" for u, v, d in G.edges(data=True)}
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=1800, arrowsize=20, font_weight='bold')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='brown', font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.show()
