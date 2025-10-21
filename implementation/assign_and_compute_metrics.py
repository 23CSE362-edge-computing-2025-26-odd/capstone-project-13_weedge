import json
import csv
import os
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).parent
DATASET_JSON = DATA_DIR / "dataset.json"
TASK_CSV = DATA_DIR / "task_dataset.csv"
SPEC_CSV = DATA_DIR / "SPECpower_ssj2008_Results.csv"


def load_dataset():
    with open(DATASET_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_specpower():
    spec = []
    if not SPEC_CSV.exists():
        return spec
    with open(SPEC_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 5:
                continue
            cpu_desc = row[0].strip()
            try:
                max_watts = float(row[3])
            except ValueError:
                max_watts = None
            spec.append({'cpu_description': cpu_desc, 'max_watts': max_watts})
    return spec


def main():
    ds = load_dataset()

    # Build server map from dataset.json
    servers = {}
    for s in ds.get('EdgeServer', []):
        a = s.get('attributes', {})
        sid = a.get('id')
        servers[sid] = {
            'id': sid,
            'model_name': a.get('model_name'),
            'cpu_description': a.get('cpu_description'),
            'max_watts': a.get('max_watts') or a.get('maxpowerconsumption') or (a.get('power_model_parameters') or {}).get('max_power_consumption'),
            'idle_watts': a.get('idle_watts') or (a.get('power_model_parameters') or {}).get('idlepower'),
            'efficiency_ssj_per_watt': a.get('efficiency_ssj_per_watt'),
            'assigned_tasks': [],
        }

    spec = parse_specpower()

    # Use Application entries from dataset.json as tasks
    tasks = []
    for app in ds.get('Application', []):
        a = app.get('attributes', {})
        rel = app.get('relationships', {})
        task = {
            'id': a.get('id'),
            'label': a.get('label'),
            'sensor_id': a.get('sensor_id'),
            'timestamp': a.get('timestamp'),
            'network_latency': float(a.get('network_latency', 0)),
            'edge_processing_time': float(a.get('edge_processing_time', 0)),
            'predicted_failure': float(a.get('predicted_failure', 0)),
            'cpu_demand': float(a.get('cpu_demand', 0)),
            'memory_demand': float(a.get('memory_demand', 0)),
            'priority': a.get('priority', 0),
            'complexity_score': float(a.get('complexity_score', 0)),
            'assigned_server': None,
        }

        # If the application has a server relationship in dataset.json, use it
        # There may be a 'server' relationship elsewhere; attempt to find a server mapping
        # Search through other objects referencing this application for assigned server
        # Fallback: assign using round-robin later
        tasks.append((task, rel))

    # Some dataset placements may be stored separately (e.g., in Service or other items)
    # We'll try to look up explicit assignments within dataset.json by scanning all objects for relationships pointing to EdgeServer.
    app_to_server = {}
    # There's a 'Application' array; sometimes assignments are represented in other arrays (e.g., containers). We'll scan whole dataset for objects that reference an application and a server.
    for key, items in ds.items():
        if not isinstance(items, list):
            continue
        for item in items:
            rel = item.get('relationships', {})
            if not rel:
                continue
            app_rel = None
            if 'application' in rel:
                app_rel = rel.get('application')
            elif 'applications' in rel and isinstance(rel.get('applications'), list) and rel.get('applications'):
                app_rel = rel.get('applications')[0]
            if app_rel and isinstance(app_rel, dict):
                app_id = app_rel.get('id')
                server_rel = rel.get('server') or rel.get('servers') or rel.get('edge_server')
                if server_rel and isinstance(server_rel, dict) and server_rel.get('class') == 'EdgeServer':
                    app_to_server[app_id] = server_rel.get('id')

    # Assign tasks
    server_ids = list(servers.keys())
    rr = 0
    task_outputs = []
    for t, rel in tasks:
        app_id = t['id']
        assigned = None
        # prefer explicit relationship if present in relationships of the Application object
        if rel and 'server' in rel and isinstance(rel['server'], dict):
            assigned = rel['server'].get('id')
        if not assigned:
            assigned = app_to_server.get(app_id)
        if not assigned:
            if server_ids:
                assigned = server_ids[rr % len(server_ids)]
                rr += 1
        t['assigned_server'] = assigned
        if assigned in servers:
            servers[assigned]['assigned_tasks'].append(t['id'])

        # Compute latency and reward
        latency = t['network_latency'] + t['edge_processing_time']

        # Reward formula (assumptions): higher for low latency, high server efficiency, penalize predicted failures
        srv = servers.get(assigned, {})
        eff = srv.get('efficiency_ssj_per_watt') or 0
        max_watts = srv.get('max_watts') or 1
        # coefficients chosen heuristically; documentable
        alpha = 100.0
        beta = 0.01
        gamma = 1.0
        reward = alpha * (1.0 / (1.0 + latency)) + beta * (eff / max_watts) - gamma * t['predicted_failure']

        task_outputs.append({
            'task_id': t['id'],
            'label': t['label'],
            'sensor_id': t['sensor_id'],
            'assigned_server': assigned,
            'latency': latency,
            'reward': reward,
            'predicted_failure': t['predicted_failure'],
            'cpu_demand': t['cpu_demand'],
            'memory_demand': t['memory_demand'],
        })

    # Build server summary
    server_outputs = []
    for sid, s in servers.items():
        num_tasks = len(s['assigned_tasks'])
        # aggregate latency and reward for tasks assigned to this server
        assigned_tasks = [t for t in task_outputs if t['assigned_server'] == sid]
        avg_latency = sum(t['latency'] for t in assigned_tasks) / (len(assigned_tasks) or 1)
        total_reward = sum(t['reward'] for t in assigned_tasks)
        server_outputs.append({
            'server_id': sid,
            'model_name': s.get('model_name'),
            'cpu_description': s.get('cpu_description'),
            'max_watts': s.get('max_watts'),
            'idle_watts': s.get('idle_watts'),
            'efficiency_ssj_per_watt': s.get('efficiency_ssj_per_watt'),
            'num_assigned_tasks': num_tasks,
            'avg_latency': avg_latency,
            'total_reward': total_reward,
        })

    # Write outputs
    out_dir = DATA_DIR
    with open(out_dir / 'task_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(task_outputs[0].keys()) if task_outputs else [])
        if task_outputs:
            writer.writeheader()
            writer.writerows(task_outputs)

    with open(out_dir / 'server_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(server_outputs[0].keys()) if server_outputs else [])
        if server_outputs:
            writer.writeheader()
            writer.writerows(server_outputs)

    with open(out_dir / 'task_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(task_outputs, f, indent=2)
    with open(out_dir / 'server_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(server_outputs, f, indent=2)

    print(f'Wrote {len(task_outputs)} task metrics and {len(server_outputs)} server summaries to {out_dir}')

    # Print metrics to console. By default print a concise summary; to print full lists set env var PRINT_METRICS_FULL=1
    full = os.getenv('PRINT_METRICS_FULL', '0') == '1'

    if full:
        print('\n--- Task metrics (full) ---')
        for t in task_outputs:
            print(json.dumps(t, ensure_ascii=False))
        print('\n--- Server metrics (full) ---')
        for s in server_outputs:
            print(json.dumps(s, ensure_ascii=False))
    else:
        print('\n--- Task metrics (top 20 by reward) ---')
        top_tasks = sorted(task_outputs, key=lambda x: x.get('reward', 0), reverse=True)[:20]
        for t in top_tasks:
            print(f"task_id={t['task_id']}, assigned_server={t['assigned_server']}, latency={t['latency']:.3f}, reward={t['reward']:.3f}")

        # Server summaries are written to files (server_metrics.csv / server_metrics.json).
        # Removed verbose per-server console printing to keep MAIN.py output concise.


if __name__ == '__main__':
    main()
