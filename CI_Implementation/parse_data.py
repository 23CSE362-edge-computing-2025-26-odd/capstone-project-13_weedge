import pandas as pd
import json

def create_edgesimpy_dataset(task_csv, spec_csv, output_json="edgesimpy_dataset.json"):
    iiot_df = pd.read_csv(task_csv)
    spec_df = pd.read_csv(spec_csv)

    dataset = {
        "EdgeServer": [],
        "Application": [],
        "Service": []
    }

    # --- Edge Servers from SPECpower ---
    for i, row in spec_df.iterrows():
        try:
            cpu_ops = int(str(row.get("ssj_ops @ 100%", 1000000)).replace(",", "").split()[0])
        except:
            cpu_ops = 1000000
        idle_power = float(row.get("Avg Watts @ Active Idle", 50))
        max_power = float(row.get("Avg Watts @ 100%", 100))
        dataset["EdgeServer"].append({
            "id": i + 1,
            "cpu_capacity": max(1, cpu_ops // 1_000_000),
            "idle_power": idle_power,
            "max_power": max_power
        })

    # --- Applications (Tasks) ---
    for i, row in iiot_df.iterrows():
        temp = float(row["Temperature"])
        pres = float(row["Pressure"])
        vib = float(row.get("Vibration", 0))
        ept = float(row["Edge_Processing_Time"])
        complexity = ((temp / 100) * 0.3) + ((pres / 120) * 0.2) + ((vib / 50) * 0.3) + ((ept / 15) * 0.2)
        cpu_demand = max(1, int(complexity * 5))
        priority = 1 if int(row["Predicted_Failure"]) == 1 else 2

        dataset["Application"].append({
            "id": i + 1,
            "sensor": row["Sensor_ID"],
            "cpu_demand": cpu_demand,
            "priority": priority,
            "temperature": temp,
            "pressure": pres,
            "vibration": vib
        })

        dataset["Service"].append({
            "id": i + 1,
            "app_id": i + 1,
            "server_id": (i % len(spec_df)) + 1,
            "cpu_demand": cpu_demand
        })

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Created EdgeSimPy dataset: {output_json}")
    return dataset
