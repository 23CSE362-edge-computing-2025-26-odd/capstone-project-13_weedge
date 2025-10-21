
import json
import random
import os
import edge_sim_py as esp
from mud import MUDAlgorithm
from eaa_ts import EAA_TS
from eaa_nts import EAA_NTS
from coverage_utils import infer_coverage_from_dataset
# Import the metrics script so running main.py also computes assignments and metrics
import assign_and_compute_metrics as metrics_module


class RewardOrientedTaskOffloading:
    def __init__(self, powerlimit=1000):
        self.powerlimit = powerlimit
        self.edgeservers = []
        self.tasks = []
        self.Y = None

    def setupenvironment(self):
        with open('dataset.json', 'r') as f:
            data = json.load(f)

        self.edgeservers = []
        for es_data in data.get("EdgeServer", []):
            edgeserver = esp.EdgeServer()
            edgeserver.id = es_data['attributes']['id']
            edgeserver.capacity = es_data['attributes'].get('cpu', 100)
            edgeserver.poweridle = es_data['attributes'].get('idlewatts', 80)
            edgeserver.poweractivecoeffs = {
                1.0: es_data['attributes'].get('maxwatts', 200)
            }
            edgeserver.basestation = es_data['attributes'].get(
                'basestation', None)
            self.edgeservers.append(edgeserver)

    def powerfunction(self, edgeserver, utilization):
        coeffs = edgeserver.poweractivecoeffs
        utils = sorted(coeffs.keys())
        if utilization <= utils[0]:
            alpha = coeffs[utils[0]]
        elif utilization >= utils[-1]:
            alpha = coeffs[utils[-1]]
        else:
            for i in range(len(utils) - 1):
                if utils[i] <= utilization <= utils[i + 1]:
                    u1, u2 = utils[i], utils[i + 1]
                    a1, a2 = coeffs[u1], coeffs[u2]
                    alpha = a1 + (a2 - a1) * (utilization - u1) / (u2 - u1)
                    break
        return alpha * utilization + edgeserver.poweridle * (1 - utilization)

    def gettasks(self):
        with open('dataset.json', 'r') as f:
            data = json.load(f)

        self.tasks = []
        # Add all User nodes as tasks
        for user in data.get("User", []):
            task = esp.Application()
            task.id = user["attributes"]["id"]
            task.usage = user["attributes"].get("usage", random.randint(5, 25))
            task.rewards = {}
            for es in self.edgeservers:
                task.rewards[es.id] = random.randint(100, 500)
            task.coverage = {}
            self.tasks.append(task)

        # Add all Application nodes as tasks
        for app in data.get("Application", []):
            task = esp.Application()
            task.id = app["attributes"]["id"]
            task.usage = app["attributes"].get("usage", random.randint(5, 25))
            task.rewards = {}
            for es in self.edgeservers:
                task.rewards[es.id] = random.randint(100, 500)
            task.coverage = {}
            # Record associated user (if present) so we can look up coverage by user coordinates
            # Some Application entries are generated from sensors/users and the coverage mapping
            # is keyed by User id (in `coverage_maps`). Save user id on the task for later.
            user_id = None
            rel = app.get('relationships', {})
            if rel:
                users = rel.get('users') or rel.get('user')
                if isinstance(users, list) and users:
                    # relationships.users is a list of {'class':'User','id': X}
                    try:
                        user_id = users[0].get('id')
                    except Exception:
                        user_id = None
            task.user_id = user_id
            self.tasks.append(task)

    def runsimulation(self):
        mudalgo = MUDAlgorithm(self.powerlimit, self.tasks,
                               self.edgeservers, self.powerfunction)
        self.Y = mudalgo.run()
        # If any server Y is zero, compute a conservative fallback:
        # fallback = sum of usages of tasks that can cover that server.
        for es in self.edgeservers:
            if self.Y.get(es.id, 0) == 0:
                covered_usage = sum(t.usage for t in self.tasks if t.coverage.get(es.id, 0) == 1)
                if covered_usage > 0:
                    self.Y[es.id] = covered_usage
                else:
                    # As a last resort, set Y to server capacity (allows at least some allocation)
                    self.Y[es.id] = es.capacity

        print(f"Maximum allowable utilization Y (after fallback fill): {self.Y}")
        eaatsalgo = EAA_TS(self.tasks, self.edgeservers, self.Y)
        allocation_ts = eaatsalgo.run()
        print(f"EAA-TS allocation: {allocation_ts}")
        eaantsalgo = EAA_NTS(self.tasks, self.edgeservers, self.Y)
        allocation_nts = eaantsalgo.run()
        print(f"EAA-NTS allocation: {allocation_nts}")


if __name__ == "__main__":
    # Read edge servers first to compute total max power
    with open('dataset.json', 'r') as f:
        data = json.load(f)
    edge_server_list = data.get("EdgeServer", [])
    max_power_sum = sum(es['attributes'].get('maxwatts', 200)
                        for es in edge_server_list)

    # Allow overriding the fraction of total max power used via env var POWER_LIMIT_FACTOR
    # Default to 1.2 (120% of sum max watts) to reduce budget-blocked rejections during testing.
    try:
        factor = float(os.environ.get('POWER_LIMIT_FACTOR', '1.2'))
    except Exception:
        factor = 1.2
    simulator = RewardOrientedTaskOffloading(
        powerlimit=int(max_power_sum * factor))
    print(f"Using POWER_LIMIT_FACTOR={factor}, powerlimit={simulator.powerlimit} (sum max watts={max_power_sum})")

    simulator.setupenvironment()
    print(f"Loaded edge servers: {[es.id for es in simulator.edgeservers]}")

    print("Loading tasks from dataset (User + Application)...")
    simulator.gettasks()
    print(f"Loaded tasks: {[task.id for task in simulator.tasks]} (count: {len(simulator.tasks)})")

    print("Inferring coverage for tasks...")
    try:
        cov_radius = float(os.environ.get('COVERAGE_RADIUS', '60'))
    except Exception:
        cov_radius = 60.0
    print(f"Using COVERAGE_RADIUS={cov_radius}")
    coverage_maps = infer_coverage_from_dataset('./dataset.json', coverage_radius=cov_radius)

    edge_ids = [es.id for es in simulator.edgeservers]
    for task in simulator.tasks:
        # Prefer to look up coverage by the user id associated to the Application task
        user_key = getattr(task, 'user_id', None)
        if user_key is not None and user_key in coverage_maps:
            coverage = coverage_maps.get(user_key, {})
        else:
            coverage = coverage_maps.get(task.id, {})
        for eid in edge_ids:
            if eid not in coverage:
                coverage[eid] = 0
        task.coverage = coverage

    print("Running simulation...")
    simulator.runsimulation()

    # After the simulation and allocations, also run the assignment+metrics routine
    # This will create `task_metrics.csv`, `server_metrics.csv`, `task_metrics.json`, and `server_metrics.json` in the same folder.
    print("Computing task/server metrics and writing outputs...")
    try:
        metrics_module.main()
    except Exception as e:
        print(f"Failed to compute metrics: {e}")

    # Optionally generate the reward distribution plot after metrics are written.
    # Set PLOT_AFTER_RUN=1 to enable. Optional env vars:
    # - PLOT_STEP (int) default 20000
    # - PLOT_OPEN=1 to open the saved PNG with default viewer
    try:
        plot_after = os.environ.get('PLOT_AFTER_RUN', '0') == '1'
        if plot_after:
            try:
                import plot_rewards
                metrics_file = os.path.join(os.path.dirname(__file__), 'server_metrics.json')
                step = int(os.environ.get('PLOT_STEP', '20000'))
                out = os.path.join(os.path.dirname(__file__), 'reward_distribution.png')
                # call plot_rewards main functions programmatically
                rewards = [plot_rewards.load_server_rewards(metrics_file)]
                max_val = max((max(r) for r in rewards if r), default=step * 5)
                bins = list(range(0, int((max_val)) + step, step))
                labels = [os.path.splitext(os.path.basename(metrics_file))[0]]
                fig, ax = plot_rewards.plot_grouped_bar([plot_rewards.group_counts(r, bins) for r in rewards], bins, labels, out)
                if os.environ.get('PLOT_OPEN', '0') == '1':
                    try:
                        import os as _os
                        _os.startfile(out)
                    except Exception as e:
                        print(f"Failed to open plot: {e}")
            except Exception as e:
                print(f"Failed to generate reward plot: {e}")
    except Exception:
        pass
