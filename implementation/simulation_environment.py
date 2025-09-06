import simpy
import random


num_servers = 4
num_taks = 15
periods = 10


class EdgeServer:
    def __init__(self, env, id, capacity):
        self.env = env
        self.id = id
        self.capacity = capacity
        self.allocated = 0

    def utlization(self):
        return self.allocated / self.capacity

    def power(self):
        # Power calc
        return 1


class Task:
    def __init__(self, id, usage, reward):
        self.id = id
        self.usage = usage
        self.reward = reward


def create_server(env, num_servers):
    server_par = {
        {'capacity': 100},
        {'capacity': 120},
        {'capacity': 110},
        {'capacity': 95},
    }
    servers = []
    for i in range(num_servers):
        p = server_par[i % len(server_par)]
        servers.append(EdgeServer(env, i, p['capacity']))
    return servers


def create_tasks(num_tasks, num_sevrers):
    # grain_size usage
    min_usage, max_usage = 1, 10
    min_reward, maxx_reward = 50, 200
    tasks = []
    for i in range(num_tasks):
        usage = random.randint(min_usage, max_usage)
        reward = random.randint(min_reward, maxx_reward)
        tasks.append(Task(i, usage, reward))
    return tasks


def main():
    random.seed(42)
    env = simpy.Environment()
    servers = create_server(env, num_servers)
    tasks = create_tasks(num_taks, num_servers)
    print("Edge Servers:")
    for s in servers:
        print(f"  Server {s.id}: Capacity={
              s.capacity}, Alpha={s.alpha}, Beta={s.beta}")

    print("Tasks:")
    for t in tasks:
        print(f"  Task {t.id}: Usage={t.usage}, Reward={
              t.reward}, Coverage={t.coverage}")

    env.run(until=periods)


if __name__ == "__main__":
    main()
