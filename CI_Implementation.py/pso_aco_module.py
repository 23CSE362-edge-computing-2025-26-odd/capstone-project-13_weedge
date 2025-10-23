import numpy as np

def project_row_to_simplex_leq1(row):
    r = np.clip(row, 0, 1)
    s = r.sum()
    if s <= 1: return r
    u = np.sort(r)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(r - theta, 0)

def project_matrix(X): return np.array([project_row_to_simplex_leq1(row) for row in X])

def fitness(X, U, R, C, idle, slope, Plimit, penalty_w=1e6):
    reward = np.sum(R * (U.reshape(-1,1) * X))
    server_load = (U.reshape(-1,1) * X).sum(axis=0)
    frac_util = server_load / C
    power = idle + slope * (frac_util**2) * 100
    total_power = np.sum(power)
    penalty = penalty_w * np.sum(np.maximum(server_load - C, 0))
    return -(reward) + penalty

# --- PSO (task splitting) ---
def pso_allocate(U, R, C, idle, slope, Plimit, swarm_size=20, iters=60):
    rng = np.random.default_rng(42)
    n_tasks, n_servers = R.shape
    X = rng.random((swarm_size, n_tasks, n_servers))
    for k in range(swarm_size): X[k] = project_matrix(X[k])
    V = rng.normal(0, 0.1, size=(swarm_size, n_tasks, n_servers))
    pbest = X.copy()
    pbest_val = np.array([fitness(X[k], U, R, C, idle, slope, Plimit) for k in range(swarm_size)])
    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = pbest_val[g_idx]
    for _ in range(iters):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = 0.72*V + 1.4*r1*(pbest - X) + 1.4*r2*(gbest - X)
        X = np.clip(X + V, 0, 1)
        for k in range(swarm_size): X[k] = project_matrix(X[k])
        vals = np.array([fitness(X[k], U, R, C, idle, slope, Plimit) for k in range(swarm_size)])
        improved = vals < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = vals[improved]
        if pbest_val.min() < gbest_val:
            g_idx = np.argmin(pbest_val)
            gbest = pbest[g_idx].copy()
            gbest_val = pbest_val[g_idx]
    return gbest

# --- ACO (non-splitting, fully stable) ---
def aco_allocate(U, R, C, idle, slope, Plimit, n_ants=20, iters=60):
    rng = np.random.default_rng(42)
    n_tasks, n_servers = R.shape
    pheromone = np.ones((n_tasks, n_servers))
    heuristic = np.clip(R / np.maximum(C[np.newaxis, :], U[:, np.newaxis]), 1e-6, 1e6)
    best_X, best_fit = None, np.inf
    for _ in range(iters):
        solutions, scores = [], []
        for _ in range(n_ants):
            X = np.zeros((n_tasks, n_servers))
            server_remaining = C.copy()
            for i in range(n_tasks):
                prob = (pheromone[i]**1) * (heuristic[i]**2)
                mask = (server_remaining - U[i]) >= 0
                prob *= mask
                prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
                prob = np.clip(prob, 0.0, None)
                total = prob.sum()
                if total <= 1e-12:
                    prob = np.ones(n_servers) / n_servers
                else:
                    prob /= total
                prob = np.clip(prob, 0.0, 1.0)
                prob /= prob.sum() + 1e-12
                s = rng.choice(n_servers, p=prob)
                X[i, s] = 1.0
                server_remaining[s] -= U[i]
            fit = fitness(X, U, R, C, idle, slope, Plimit)
            solutions.append(X)
            scores.append(fit)
            if fit < best_fit:
                best_fit, best_X = fit, X.copy()
        pheromone *= 0.8
        elite = np.argsort(scores)[:max(1, n_ants // 5)]
        for e in elite:
            for i in range(n_tasks):
                j = np.argmax(solutions[e][i])
                pheromone[i, j] += 10 / (1 + scores[e])
    return best_X
