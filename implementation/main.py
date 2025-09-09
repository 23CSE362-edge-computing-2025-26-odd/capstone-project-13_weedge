from models import EdgeServer, Task
from mud import mud_algorithm
from allocation import build_mcmf, solve_mcmf, plot_graph
from simulation import simpy_run

def main():
    es_dict = {
        'ES1': EdgeServer('ES1', 20, 214, 81.6, 150),
        'ES2': EdgeServer('ES2', 15, 200, 82, 120)
    }
    tasks = [
        Task(1, 5, 30, ['ES1', 'ES2']),
        Task(2, 10, 50, ['ES1']),
        Task(3, 6, 36, ['ES2']),
        Task(4, 2, 10, ['ES1', 'ES2']),
        Task(5, 4, 24, ['ES1']),
    ]
    power_budget = 250

    # Step 1: MUD to constrain ES utilization
    mud_algorithm(tasks, es_dict, power_budget)
    for es in es_dict.values():
        print(f"{es}: Max Utilization = {es.max_utilization}")

    # Step 2: Allocation and visualization
    for splitting in [True, False]:
        label = "WITH Splitting" if splitting else "WITHOUT Splitting"
        print(f"\n--- {label} ---")
        G = build_mcmf(tasks, es_dict, splitting)
        plot_graph(G, title=f"MCMF Graph {label}")
        allocations = solve_mcmf(G)
        total_reward = 0
        for tid, blocks in allocations.items():
            reward_for_task = 0
            for es_name, units in blocks.items():
                task = next(t for t in tasks if t.id == tid)
                reward_for_task += task.reward * (units / task.cpu)
            total_reward += reward_for_task
            print(f"Task {tid}: {blocks}, Reward = {reward_for_task}")
        print(f"Total Reward: {total_reward}")

    # Step 3: SimPy Simulation of task execution
    print("\n--- SimPy Simulation (WITH Splitting) ---")
    G = build_mcmf(tasks, es_dict, True)
    allocations = solve_mcmf(G)
    timeline = simpy_run(tasks, es_dict, allocations)
    for t in timeline:
        print(f"{t[0]}: {t[1]}")

if __name__ == "__main__":
    main()
# main.py
from models import EdgeServer, Task
from mud import mud_algorithm
from allocation import build_mcmf, solve_mcmf, plot_graph
from simulation import simpy_run

def main():
    es_dict = {
        'ES1': EdgeServer('ES1', 20, 214, 81.6, 150),
        'ES2': EdgeServer('ES2', 15, 200, 82, 120)
    }
    tasks = [
        Task(1, 5, 30, ['ES1', 'ES2']),
        Task(2, 10, 50, ['ES1']),
        Task(3, 6, 36, ['ES2']),
        Task(4, 2, 10, ['ES1', 'ES2']),
        Task(5, 4, 24, ['ES1']),
    ]
    power_budget = 250

    # Step 1: MUD to constrain ES utilization
    mud_algorithm(tasks, es_dict, power_budget)
    for es in es_dict.values():
        print(f"{es}: Max Utilization = {es.max_utilization}")

    # Step 2: Allocation and visualization
    for splitting in [True, False]:
        label = "WITH Splitting" if splitting else "WITHOUT Splitting"
        print(f"\n--- {label} ---")
        G = build_mcmf(tasks, es_dict, splitting)
        plot_graph(G, title=f"MCMF Graph {label}")
        allocations = solve_mcmf(G)
        total_reward = 0
        for tid, blocks in allocations.items():
            reward_for_task = 0
            for es_name, units in blocks.items():
                task = next(t for t in tasks if t.id == tid)
                reward_for_task += task.reward * (units / task.cpu)
            total_reward += reward_for_task
            print(f"Task {tid}: {blocks}, Reward = {reward_for_task}")
        print(f"Total Reward: {total_reward}")

    # Step 3: SimPy Simulation of task execution
    print("\n--- SimPy Simulation (WITH Splitting) ---")
    G = build_mcmf(tasks, es_dict, True)
    allocations = solve_mcmf(G)
    timeline = simpy_run(tasks, es_dict, allocations)
    for t in timeline:
        print(f"{t[0]}: {t[1]}")

if __name__ == "__main__":
    main()
