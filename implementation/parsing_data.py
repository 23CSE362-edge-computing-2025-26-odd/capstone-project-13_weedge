#!/usr/bin/env python3
import pandas as pd
import json
import random


def create_edgesimpy_dataset(output_path='edgesimpy_dataset.json'):
    # Load datasets
    iiot_df = pd.read_csv('./task_dataset.csv')

    specpower_df = pd.read_csv('./SPECpower_ssj2008_Results.csv')
    edgesimpy_dataset = {
        "NetworkSwitch": [],
        "NetworkLink": [],
        "BaseStation": [],
        "User": [],
        "Service": [],
        "Application": [],
        "EdgeServer": [],
        "CircularDurationAndIntervalAccessPattern": [],
        "Topology": []
    }

    # Add topology
    edgesimpy_dataset["Topology"].append({
        "attributes": {"id": 1},
        "relationships": {}
    })

    # Get unique sensors and total servers
    unique_sensors = sorted(iiot_df['Sensor_ID'].unique())
    num_sensors = len(unique_sensors)
    num_servers = len(specpower_df)

    # Create spatial layout - servers arranged in grid
    grid_size = int(num_servers**0.5) + 1

    for i in range(num_servers):
        server_spec = specpower_df.iloc[i]

        cpu_description = server_spec['CPU Description']
        memory_gb = int(server_spec['Total Memory (GB)'])
        cpu_ops = int(server_spec['ssj_ops @ 100%'])
        idle_power = float(server_spec['Avg Watts @ Active Idle'])
        max_power = float(server_spec['Avg Watts @ 100%'])
        efficiency = float(server_spec['Result (Overall ssj_ops/watt)'])

        cpu_units = int(cpu_ops / 1_000_000)

        # Grid layout for coordinates
        grid_x = (i % grid_size) * 20
        grid_y = (i // grid_size) * 20
        coords = [grid_x, grid_y]

        # Assign to base station (cycle through available sensors)
        base_station_id = (i % num_sensors) + 1

        server = {
            "attributes": {
                "id": i + 1,
                "available": True,
                "model_name": f"Server_{i+1}",
                "cpu": cpu_units,
                "memory": memory_gb * 1024,
                "disk": 50000,
                "coordinates": coords,
                "power_model_parameters": {
                    "static_power_percentage": idle_power / max_power,
                    "idlepower": idle_power,
                    "max_power_consumption": max_power
                },
                "maxpowerconsumption": max_power,
                "max_concurrent_layers": 5,
                "cpu_description": cpu_description,
                "idle_watts": idle_power,
                "max_watts": max_power,
                "efficiency_ssj_per_watt": efficiency
            },
            "relationships": {
                "power_model": "LinearServerPowerModel",
                "base_station": {"class": "BaseStation", "id": base_station_id}
            }
        }
        edgesimpy_dataset["EdgeServer"].append(server)

    for idx, row in iiot_df.iterrows():
        app_id = idx + 1

        # Calculate CPU demand from sensor data
        complexity = (
            (row['Temperature'] / 100) * 0.3 +
            (row['Pressure'] / 120) * 0.2 +
            (row['Vibration'] / 50) * 0.3 +
            (row['Edge_Processing_Time'] / 15) * 0.2
        )
        cpu_demand = max(1, int(complexity * 5))
        memory_demand = int(row['Edge_Processing_Time'] * 200)

        # Priority based on failure and maintenance
        priority = 1 if row['Predicted_Failure'] == 1 else 2
        if row['Maintenance_Status'] == 'Warning':
            priority = 1
        elif row['Maintenance_Status'] == 'Failure':
            priority = 0

        application = {
            "attributes": {
                "id": app_id,
                "label": f"Task_{row['Sensor_ID']}_{app_id}",
                "sensor_id": row['Sensor_ID'],
                "timestamp": str(row['Timestamp']),
                "network_latency": float(row['Network_Latency']),
                "edge_processing_time": float(row['Edge_Processing_Time']),
                "predicted_failure": int(row['Predicted_Failure']),
                "maintenance_status": row['Maintenance_Status'],
                "temperature": float(row['Temperature']),
                "pressure": float(row['Pressure']),
                "vibration": float(row['Vibration']),
                "fuzzy_pid_output": float(row['Fuzzy_PID_Output']),
                "cpu_demand": cpu_demand,
                "memory_demand": memory_demand,
                "priority": priority,
                "complexity_score": float(complexity)
            },
            "relationships": {
                "services": [{"class": "Service", "id": app_id}],
                "users": [{"class": "User", "id": ((app_id % num_sensors) + 1)}]
            }
        }
        edgesimpy_dataset["Application"].append(application)

    for app in edgesimpy_dataset["Application"]:
        app_id = app["attributes"]["id"]
        sensor_id = app["attributes"]["sensor_id"]

        server_id = ((app_id - 1) % num_servers) + 1

        service = {
            "attributes": {
                "id": app_id,
                "label": app["attributes"]["label"],
                "state": 0,
                "_available": True,
                "cpu_demand": app["attributes"]["cpu_demand"],
                "memory_demand": app["attributes"]["memory_demand"],
                "image_digest": f"sha256:{'0' * 63}{(app_id % 10):01d}"
            },
            "relationships": {
                "application": {"class": "Application", "id": app_id},
                "server": {"class": "EdgeServer", "id": server_id}
            }
        }
        edgesimpy_dataset["Service"].append(service)

    for i, sensor_id in enumerate(unique_sensors):
        coords = [i * 30, -20]

        sensor_tasks = iiot_df[iiot_df['Sensor_ID'] == sensor_id]
        avg_latency = sensor_tasks['Network_Latency'].mean()
        wireless_delay = int(avg_latency * 0.3)

        connected_servers = [
            {"class": "EdgeServer", "id": j + 1}
            for j in range(num_servers)
            if (j % num_sensors) == i
        ]

        base_station = {
            "attributes": {
                "id": i + 1,
                "coordinates": coords,
                "wireless_delay": wireless_delay
            },
            "relationships": {
                "users": [],
                "edge_servers": connected_servers,
                "network_switch": {"class": "NetworkSwitch", "id": i + 1}
            }
        }
        edgesimpy_dataset["BaseStation"].append(base_station)

    for i in range(num_sensors):
        coords = edgesimpy_dataset["BaseStation"][i]["attributes"]["coordinates"]

        connected_servers = edgesimpy_dataset["BaseStation"][i]["relationships"]["edge_servers"]

        switch = {
            "attributes": {
                "id": i + 1,
                "coordinates": coords,
                "active": True,
                "power_model_parameters": {
                    "chassis_power": 60,
                    "ports_power_consumption": {
                        "125": 1,
                        "12.5": 0.3
                    }
                }
            },
            "relationships": {
                "power_model": "ConteratoNetworkPowerModel",
                "edge_servers": connected_servers,
                "links": [],
                "base_station": {"class": "BaseStation", "id": i + 1}
            }
        }
        edgesimpy_dataset["NetworkSwitch"].append(switch)

    link_id = 1
    for i in range(len(edgesimpy_dataset["NetworkSwitch"])):
        for j in range(i + 1, len(edgesimpy_dataset["NetworkSwitch"])):
            switch_i = edgesimpy_dataset["NetworkSwitch"][i]
            switch_j = edgesimpy_dataset["NetworkSwitch"][j]

            x1, y1 = switch_i["attributes"]["coordinates"]
            x2, y2 = switch_j["attributes"]["coordinates"]
            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            delay = int(max(5, distance * 0.5))

            link = {
                "attributes": {
                    "id": link_id,
                    "delay": delay,
                    "bandwidth": 12.5,
                    "bandwidth_demand": 0,
                    "active": True
                },
                "relationships": {
                    "topology": {"class": "Topology", "id": 1},
                    "active_flows": [],
                    "applications": [],
                    "nodes": [
                        {"class": "NetworkSwitch", "id": i + 1},
                        {"class": "NetworkSwitch", "id": j + 1}
                    ]
                }
            }
            edgesimpy_dataset["NetworkLink"].append(link)
            link_id += 1

    for i, sensor_id in enumerate(unique_sensors):
        user_id = i + 1
        bs_id = i + 1

        sensor_apps = [app for app in edgesimpy_dataset["Application"]
                       if app["attributes"]["sensor_id"] == sensor_id]

        delays = {str(app["attributes"]["id"]): int(app["attributes"]["network_latency"])
                  for app in sensor_apps}
        delay_slas = {str(app["attributes"]["id"]): int(app["attributes"]["network_latency"] * 1.5)
                      for app in sensor_apps}

        coords = edgesimpy_dataset["BaseStation"][bs_id -
                                                  1]["attributes"]["coordinates"]

        user = {
            "attributes": {
                "id": user_id,
                "coordinates": coords,
                "coordinates_trace": [coords] * 100,
                "delays": delays,
                "delay_slas": delay_slas,
                "communication_paths": {},
                "making_requests": {str(app["attributes"]["id"]): {"1": True}
                                    for app in sensor_apps}
            },
            "relationships": {
                "access_patterns": {str(app["attributes"]["id"]): {
                    "class": "CircularDurationAndIntervalAccessPattern",
                    "id": app["attributes"]["id"]
                } for app in sensor_apps},
                "mobility_model": "pathway",
                "applications": [{"class": "Application", "id": app["attributes"]["id"]}
                                 for app in sensor_apps],
                "base_station": {"class": "BaseStation", "id": bs_id}
            }
        }
        edgesimpy_dataset["User"].append(user)

        edgesimpy_dataset["BaseStation"][bs_id - 1]["relationships"]["users"].append({
            "class": "User", "id": user_id
        })

        # Create access patterns
        for app in sensor_apps:
            access_pattern = {
                "attributes": {
                    "id": app["attributes"]["id"],
                    "duration_values": [float('inf')],
                    "interval_values": [0],
                    "history": [{
                        "start": 1,
                        "end": float('inf'),
                        "duration": float('inf'),
                        "waiting_time": 0,
                        "access_time": 0,
                        "interval": 0,
                        "next_access": float('inf')
                    }]
                },
                "relationships": {
                    "user": {"class": "User", "id": user_id},
                    "app": {"class": "Application", "id": app["attributes"]["id"]}
                }
            }
            edgesimpy_dataset["CircularDurationAndIntervalAccessPattern"].append(
                access_pattern)

    class EdgeSimPyEncoder(json.JSONEncoder):
        def encode(self, obj):
            if isinstance(obj, float):
                if obj == float('inf'):
                    return 'Infinity'
                elif obj == float('-inf'):
                    return '-Infinity'
            return super().encode(obj)

        def iterencode(self, obj, _one_shot=False):
            for chunk in super().iterencode(obj, _one_shot):
                chunk = chunk.replace('"Infinity"', 'Infinity')
                chunk = chunk.replace('"-Infinity"', '-Infinity')
                yield chunk

    with open(output_path, 'w') as f:
        json.dump(edgesimpy_dataset, f, indent=4, cls=EdgeSimPyEncoder)

    return edgesimpy_dataset


if __name__ == '__main__':
    dataset = create_edgesimpy_dataset(
        output_path='dataset.json'
    )
