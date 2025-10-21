import numpy as np

# Import your 3 phases
from fuzzy_phase import fuzzy_phase
from pso_phase import pso_allocate
from aco_phase import aco_allocate

# ----------------- Sample Task & Server Classes -----------------
class Task:
    def __init__(self, tid, cpu, reward, candidates):
        self.id = tid
        self.cpu = cpu
        self.reward = reward
        self.candidates = candidates
        self.rewards = {i: reward for i in candidates}  # uniform for demo

class EdgeServer:
    def __init__(self, sid, capacity, idle, coeffs):
        self.id = sid
        self.capacity = capacity
        self.power_idle = idle
        self.power_active_coeffs = coeffs
        self.current_utilization = 0
        self.max_utilization = 0

    def power(self, utilization):
        # simple quadratic power model
        coeffs = self.power_active_coeffs
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
        return alpha * utilization + self.power_idle * (1 - utilization)

# ----------------- Generate Demo Dataset -----------------
def generate_demo_data():
    U = np.array([8, 6, 5, 7], dtype=float)         # workloads
    R = np.array([[40, 60], [30, 50], [45, 55], [35, 65]], dtype=float)
    C = np.array([15, 20], dtype=float)             # capacities
    idle = np.array([50, 60], dtype=float)          # idle power
    slope = np.array([6.0, 7.0])                    # slope
    Plimit = 300.0

    # Edge servers
    es1 = EdgeServer(0, 15, 50, {0: 5, 1: 10})
    es2 = EdgeServer(1, 20, 60, {0: 6, 1: 12})
    es_dict = {0: es1, 1: es2}

    # Tasks
    tasks = [
        Task(0, 8, 40, [0, 1]),
        Task(1, 6, 30, [0, 1]),
        Task(2, 5, 45, [0, 1]),
        Task(3, 7, 35, [0, 1])
    ]
    return U, R, C, idle, slope, Plimit, tasks, es_dict

# ----------------- MAIN DRIVER -----------------
if __name__ == "__main__":
    U, R, C, idle, slope, Plimit, tasks, es_dict = generate_demo_data()

    print("\n=== Phase 1: Fuzzy Allocation ===")
    Y, alloc_matrix, fuzzy_reward, fuzzy_load, fuzzy_power = fuzzy_phase(tasks, es_dict, Plimit)
    print("Y capacity array (per server):", Y)
    print("Server utilizations:", fuzzy_load)
    print("Total Reward:", fuzzy_reward)
    print("Total Power:", fuzzy_power, "/", Plimit)

    print("\n=== Phase 2a: PSO Allocation (Splitting) ===")
    gbest, pso_reward, pso_load, pso_power = pso_allocate(U, R, C, idle, slope, Plimit)
    print("Server utilizations:", pso_load)
    print("Total Reward:", pso_reward)
    print("Total Power:", pso_power, "/", Plimit)

    print("\n=== Phase 2b: ACO Allocation (No Splitting) ===")
    best_X, aco_reward, aco_load, aco_power = aco_allocate(U, R, C, idle, slope, Plimit)
    print("Server utilizations:", aco_load)
    print("Total Reward:", aco_reward)
    print("Total Power:", aco_power, "/", Plimit)
