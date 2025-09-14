import numpy as np

# ----------------- Fitness -----------------
def fitness(X, U, R, C, idle, slope, Plimit, penalty_w=1e6):
    reward = np.sum(R * (U.reshape(-1,1) * X))
    server_load = (U.reshape(-1,1) * X).sum(axis=0)

    frac_util = server_load / C
    power = idle + slope * (frac_util**2) * 100
    total_power = np.sum(power)

    tasks_per_server = (X > 1e-6).sum(axis=0)
    latency_penalty = 10.0 * np.sum(tasks_per_server)

    penalty = penalty_w * np.sum(np.maximum(server_load - C, 0))
    penalty += penalty_w * max(total_power - Plimit, 0)
    penalty += penalty_w * np.sum(np.maximum(X.sum(axis=1) - 1, 0))

    return -(reward - latency_penalty) + penalty

# ----------------- ACO Allocation -----------------
def aco_allocate(U, R, C, idle, slope, Plimit, 
                 n_ants=30, iters=100, alpha=1.0, beta=2.0, 
                 rho=0.2, Q=10, seed=42):
    rng = np.random.default_rng(seed)
    n_tasks, n_servers = R.shape

    pheromone = np.ones((n_tasks, n_servers))
    heuristic = R / (np.maximum(C[np.newaxis, :], U[:, np.newaxis]))

    best_X = None
    best_fit = np.inf

    for epoch in range(iters):
        solutions, scores = [], []
        for ant in range(n_ants):
            X = np.zeros((n_tasks, n_servers))
            server_remaining = C.copy()

            for i in range(n_tasks):
                prob = (pheromone[i] ** alpha) * (heuristic[i] ** beta)
                mask = (server_remaining - U[i]) >= 0
                if not np.any(mask): mask[:] = True
                prob = prob * mask
                if prob.sum() == 0: prob = mask.astype(float)
                prob = prob / prob.sum()

                s = rng.choice(n_servers, p=prob)
                X[i, s] = 1.0
                server_remaining[s] -= U[i]

            fit = fitness(X, U, R, C, idle, slope, Plimit)
            solutions.append(X); scores.append(fit)
            if fit < best_fit:
                best_fit = fit; best_X = X.copy()

        pheromone *= (1 - rho)
        elite_idx = np.argsort(scores)[:max(1, n_ants // 5)]
        for idx in elite_idx:
            sc, X_elite = scores[idx], solutions[idx]
            for i in range(n_tasks):
                j = np.argmax(X_elite[i])
                pheromone[i, j] += Q / (1.0 + max(0, sc))

    best_reward = np.sum(R * (U.reshape(-1,1) * best_X))
    server_load = (U.reshape(-1,1) * best_X).sum(axis=0)
    frac_util = server_load / C
    power = idle + slope * (frac_util**2) * 100
    total_power = power.sum()

    return best_X, best_reward, server_load, total_power
