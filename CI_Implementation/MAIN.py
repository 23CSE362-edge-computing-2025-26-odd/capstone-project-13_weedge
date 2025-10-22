#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from parse_data import create_edgesimpy_dataset
from fuzzy_module import fuzzy_priority
from pso_aco_module import pso_allocate, aco_allocate
from edge_sim_py import Simulator, ComponentManager, EdgeServer, Service, Application

# -------------------------------------------------------------
# Step 1 – Parse CSV datasets and generate EdgeSimPy JSON format
# -------------------------------------------------------------
print("Parsing datasets and creating EdgeSimPy-compatible dataset...")
dataset = create_edgesimpy_dataset(
    "task_dataset.csv",
    "SPECpower_ssj2008_Results.csv",
    "edgesimpy_dataset.json"
)
print("Dataset created: edgesimpy_dataset.json")

task_df = pd.read_csv("task_dataset.csv")

# -------------------------------------------------------------
# Step 2 – Load dataset and initialize EdgeSimPy environment
# -------------------------------------------------------------
with open("edgesimpy_dataset.json") as f:
    data = json.load(f)

cm = ComponentManager()
sim = Simulator(cm)

def get_attr(d, key, default=None):
    """Safely fetch an attribute from the JSON dictionary."""
    if isinstance(d, dict) and "attributes" in d:
        return d["attributes"].get(key, default)
    return d.get(key, default) if isinstance(d, dict) else default

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
N_SERVERS = 29
N_TASKS = 1000
POWER_LIMIT_FACTOR = 1.2

servers_data = data["EdgeServer"][:N_SERVERS]
applications_data = data["Application"][:N_TASKS]
services_data = data["Service"][:N_TASKS]

# -------------------------------------------------------------
# Step 3 – Create EdgeSimPy components
# -------------------------------------------------------------
edge_servers, applications, services = [], [], []

for s in servers_data:
    srv = EdgeServer()
    srv.id = get_attr(s, "id")
    srv.cpu = get_attr(s, "cpu", 1000)
    srv.model_name = get_attr(s, "model_name", f"Server_{srv.id}")
    srv.idle_watts = get_attr(s, "idle_watts", get_attr(s, "idlepower", 100))
    srv.max_watts = get_attr(s, "max_watts", get_attr(s, "maxpowerconsumption", srv.idle_watts + 100))
    edge_servers.append(srv)

for a in applications_data:
    app = Application()
    app.id = get_attr(a, "id")
    app.label = get_attr(a, "label", f"App_{app.id}")
    applications.append(app)

for svc in services_data:
    ser = Service()
    ser.id = get_attr(svc, "id")
    ser.cpu_demand = get_attr(svc, "cpu_demand", 10)
    services.append(ser)

sim.edge_servers, sim.applications, sim.services = edge_servers, applications, services

# -------------------------------------------------------------
# Step 4 – Compute fuzzy-based Y_hat values
# -------------------------------------------------------------
print("Computing fuzzy-based Y_hat values...")

num_servers = len(edge_servers)
server_util = np.zeros(num_servers)

for app in applications_data:
    reward = (
        get_attr(app, "temperature", 30) * 0.4 +
        get_attr(app, "pressure", 60) * 0.2 +
        get_attr(app, "vibration", 40) * 0.4
    )
    cpu_demand = get_attr(app, "cpu_demand", 10)
    power_est = min(cpu_demand * 5, 100)
    fprio = fuzzy_priority(reward, power_est, cpu_demand)
    sid = (get_attr(app, "id", 1) - 1) % num_servers
    server_util[sid] += fprio * cpu_demand

C = np.array([get_attr(s, "cpu", 1000) for s in servers_data], dtype=float)
idle = np.array([get_attr(s, "idle_watts", 100.0) for s in servers_data], dtype=float)
max_watts = np.array([get_attr(s, "max_watts", idle[i] + 100.0) for i, s in enumerate(servers_data)], dtype=float)

# Compute per-server fuzzy scores
fuzzy_scores = np.zeros(num_servers)
for i in range(num_servers):
    reward_component = np.clip((server_util[i] / (np.max(server_util) + 1e-9)) * 100, 0, 100)
    power_component = np.clip(((max_watts[i] - idle[i]) / np.max(max_watts - idle)) * 100, 0, 100)
    util_component = np.clip((C[i] / np.max(C)) * 100, 0, 100)
    fuzzy_scores[i] = fuzzy_priority(reward_component, power_component, util_component)

# Normalize fuzzy scores
fuzzy_scores = 0.4 + (fuzzy_scores - np.min(fuzzy_scores)) / (np.ptp(fuzzy_scores) + 1e-9) * 0.6

# Compute final Y_hat
y_hat = C * fuzzy_scores * ((max_watts - idle) / np.mean(max_watts - idle))

pd.DataFrame({
    "Server_ID": np.arange(1, num_servers + 1),
    "Y_hat": y_hat,
    "Fuzzy_Score": fuzzy_scores
}).to_csv("y_hat.csv", index=False)

print("Fuzzy heterogeneous Y_hat values computed.")

# -------------------------------------------------------------
# Step 5 – Run PSO and ACO allocations
# -------------------------------------------------------------
U = np.array([get_attr(a, "cpu_demand", 10) for a in applications_data], dtype=float)

R = []
for a in applications_data:
    reward = (
        get_attr(a, "temperature", 30) * 0.4 +
        get_attr(a, "pressure", 60) * 0.2 +
        get_attr(a, "vibration", 40) * 0.4
    )
    power_est = min(get_attr(a, "cpu_demand", 10) * 5, 100)
    R.append(fuzzy_priority(reward, power_est, get_attr(a, "cpu_demand", 10)))
R = np.tile(np.array(R).reshape(-1, 1), (1, len(C)))

print("Running PSO allocation...")
pso_X = pso_allocate(U, R, C, idle, np.ones(num_servers), np.sum(max_watts) * POWER_LIMIT_FACTOR)
pd.DataFrame(pso_X).to_csv("pso_X.csv", index=False)
print("PSO allocation completed.")

print("Running ACO allocation...")
aco_X = aco_allocate(U, R, C, idle, np.ones(num_servers), np.sum(max_watts) * POWER_LIMIT_FACTOR)
pd.DataFrame(aco_X).to_csv("aco_X.csv", index=False)
print("ACO allocation completed.")

# -------------------------------------------------------------
# Step 6 – Display allocation summary for first few tasks
# -------------------------------------------------------------
print("Setting up simulation environment...")

y_hat_dict = {i + 1: round(float(y_hat[i]), 2) for i in range(num_servers)}
print("Maximum allowable utilizations (Y):", y_hat_dict)

pso_alloc = {i + 1: list(np.where(pso_X[i] > 0.1)[0] + 1) for i in range(min(20, len(U)))}
aco_alloc = {i + 1: int(np.argmax(aco_X[i]) + 1) for i in range(min(20, len(U)))}

print("PSO allocation:", pso_alloc)
print("ACO allocation:", aco_alloc)

# -------------------------------------------------------------
# Step 7 – Compute task reward metrics
# -------------------------------------------------------------
task_rewards = []
for i in range(U.shape[0]):
    task_id = i + 1
    assigned_server = np.argmax(aco_X[i]) + 1
    latency = float(task_df.iloc[i]["Network_Latency"]) if "Network_Latency" in task_df.columns else np.random.uniform(10, 18)
    reward = float(R[i, assigned_server - 1]) * U[i] / max(latency, 1e-6)
    task_rewards.append((task_id, assigned_server, latency, reward))

task_rewards_df = pd.DataFrame(task_rewards, columns=["Task_ID", "Server_ID", "Latency", "Reward"])
task_rewards_df.to_csv("task_rewards.csv", index=False)

top15 = task_rewards_df.sort_values(by="Reward", ascending=False).head(15)
top15.to_csv("top_tasks.csv", index=False)
print("\nTop 15 tasks by reward:")
print(top15.to_string(index=False))

# -------------------------------------------------------------
# Step 8 – Final performance summary
# -------------------------------------------------------------
pso_reward = np.sum(R * (U.reshape(-1, 1) * pso_X))
pso_load = (U.reshape(-1, 1) * pso_X).sum(axis=0)
pso_power = np.sum(idle + ((pso_load / C) ** 2) * (max_watts - idle))

aco_reward = np.sum(R * (U.reshape(-1, 1) * aco_X))
aco_load = (U.reshape(-1, 1) * aco_X).sum(axis=0)
aco_power = np.sum(idle + ((aco_load / C) ** 2) * (max_watts - idle))

print("\nSummary:")
print(f"PSO total reward = {pso_reward:.2f}, total power = {pso_power:.2f}")
print(f"ACO total reward = {aco_reward:.2f}, total power = {aco_power:.2f}")
print("\nOutput files generated: y_hat.csv, pso_X.csv, aco_X.csv, task_rewards.csv, top_tasks.csv")
