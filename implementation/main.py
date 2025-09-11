import random
import edge_sim_py as esp
from mud import MUDAlgorithm
from eaa_ts import EAA_TS
from eaa_nts import EAA_NTS


class RewardOrientedTaskOffloading:
    def __init__(self, power_limit=1000):
        self.power_limit = power_limit
        self.edge_servers = []
        self.tasks = []
        self.Y = {}

    def setup_environment(self, num_edge_servers=5, num_base_stations=10):
        topology = esp.Topology()
        for i in range(num_base_stations):
            base_station = esp.BaseStation()
            base_station.id = i + 1
            base_station.coordinates = (
                random.uniform(0, 100), random.uniform(0, 100))
            base_station.coverage_radius = random.uniform(15, 25)
            topology.add_node(base_station)

        for i in range(num_edge_servers):
            edge_server = esp.EdgeServer()
            edge_server.id = i + 1
            edge_server.capacity = random.randint(80, 120)
            edge_server.power_idle = random.uniform(40, 60)
            edge_server.power_active_coeffs = {
                0.1: random.uniform(180, 200),
                0.2: random.uniform(210, 230),
                0.5: random.uniform(250, 280),
                0.8: random.uniform(300, 350),
                1.0: random.uniform(380, 420)
            }
            bs_id = random.randint(1, num_base_stations)
            edge_server.base_station = bs_id
            topology.add_node(edge_server)
            self.edge_servers.append(edge_server)
        return topology

    def power_function(self, edge_server, utilization):
        coeffs = edge_server.power_active_coeffs
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
        return alpha * utilization + edge_server.power_idle * (1 - utilization)

    def generate_tasks(self, num_tasks=20):
        self.tasks = []
        for i in range(num_tasks):
            task = esp.Application()
            task.id = i + 1
            task.usage = random.randint(5, 25)
            task.rewards = {}
            for es in self.edge_servers:
                base_reward = random.randint(100, 500)
                task.rewards[es.id] = base_reward
            task.coverage = {es.id: 1 for es in self.edge_servers}
            self.tasks.append(task)

    def run_simulation(self):
        mud_algo = MUDAlgorithm(
            self.power_limit, self.tasks, self.edge_servers, self.power_function)
        self.Y = mud_algo.run()
        print(f"Maximum allowable utilizations (Y): {self.Y}")

        eaa_ts_algo = EAA_TS(self.tasks, self.edge_servers, self.Y)
        allocation_ts = eaa_ts_algo.run()
        print(f"EAA-TS allocation: {allocation_ts}")

        eaa_nts_algo = EAA_NTS(self.tasks, self.edge_servers, self.Y)
        allocation_nts = eaa_nts_algo.run()
        print(f"EAA-NTS allocation: {allocation_nts}")


if __name__ == '__main__':
    simulator = RewardOrientedTaskOffloading(power_limit=800)
    print("Setting up environment and generating tasks...")
    simulator.setup_environment()
    simulator.generate_tasks()
    print("Running simulation...")
    simulator.run_simulation()
