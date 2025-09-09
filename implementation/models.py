def power_func(utilization, alpha=214, beta=81.6):
    return alpha * utilization + beta * (1 - utilization)

class EdgeServer:
    def __init__(self, name, capacity, alpha, beta, power_cap):
        self.name = name
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.power_cap = power_cap
        self.max_utilization = 0
        self.current_utilization = 0

    def power(self, utilization=None):
        if utilization is None:
            utilization = self.current_utilization / self.capacity
        return power_func(utilization, self.alpha, self.beta)

    def __repr__(self):
        return f"ES({self.name}, max_util={self.max_utilization}, power_cap={self.power_cap})"

class Task:
    def __init__(self, id, cpu, reward, candidates):
        self.id = id
        self.cpu = cpu
        self.reward = reward
        self.candidates = candidates

    def __repr__(self):
        return f"Task({self.id}, cpu={self.cpu}, reward={self.reward})"
