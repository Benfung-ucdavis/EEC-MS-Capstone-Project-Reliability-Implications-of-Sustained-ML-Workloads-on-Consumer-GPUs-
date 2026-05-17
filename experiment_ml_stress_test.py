import os
import csv
import time
import subprocess
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# USER SETTINGS
# =========================
# For validation test:
# IDLE_MINUTES = 1
# TRAIN_MINUTES = 5
#
# For official experiment:
# IDLE_MINUTES = 5
# TRAIN_MINUTES = 60

IDLE_MINUTES = 5          # change to 5 for official run
TRAIN_MINUTES = 60         # change to 60 for official run
LOG_INTERVAL_SEC = 2

BATCH_SIZE = 2048
INPUT_DIM = 4096
HIDDEN_DIM = 4096
OUTPUT_DIM = 1024
LEARNING_RATE = 1e-3

CSV_FILENAME = "gpu_ml_stress_log.csv"

# =========================
# CHECK GPU
# =========================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available. Make sure PyTorch with CUDA is installed.")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
print(f"Using GPU: {gpu_name}")

# =========================
# MODEL
# =========================
model = nn.Sequential(
    nn.Linear(INPUT_DIM, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

def generate_batch():
    x = torch.randn(BATCH_SIZE, INPUT_DIM, device=device)
    y = torch.randn(BATCH_SIZE, OUTPUT_DIM, device=device)
    return x, y

def query_gpu_metrics():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,fan.speed",
        "--format=csv,noheader,nounits"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]

    while len(parts) < 10:
        parts.append("N/A")

    return {
        "gpu_timestamp": parts[0],
        "gpu_name": parts[1],
        "utilization_gpu_percent": parts[2],
        "temperature_gpu_c": parts[3],
        "power_draw_w": parts[4],
        "clock_sm_mhz": parts[5],
        "clock_mem_mhz": parts[6],
        "memory_used_mb": parts[7],
        "memory_total_mb": parts[8],
        "fan_speed_percent": parts[9],
    }

fieldnames = [
    "phase",
    "wall_time",
    "elapsed_sec",
    "gpu_timestamp",
    "gpu_name",
    "utilization_gpu_percent",
    "temperature_gpu_c",
    "power_draw_w",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "memory_used_mb",
    "memory_total_mb",
    "fan_speed_percent",
    "loss"
]

with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    print(f"Starting idle baseline for {IDLE_MINUTES} minute(s)...")
    idle_start = time.time()
    last_log_time = 0

    while True:
        now = time.time()
        elapsed = now - idle_start

        if elapsed >= IDLE_MINUTES * 60:
            break

        if now - last_log_time >= LOG_INTERVAL_SEC:
            metrics = query_gpu_metrics()
            row = {
                "phase": "idle",
                "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_sec": round(elapsed, 2),
                **metrics,
                "loss": ""
            }
            writer.writerow(row)
            f.flush()

            print(
                f"[IDLE] t={elapsed:.1f}s | "
                f"Temp={metrics['temperature_gpu_c']}C | "
                f"Power={metrics['power_draw_w']}W | "
                f"Util={metrics['utilization_gpu_percent']}%"
            )

            last_log_time = now

        time.sleep(0.2)

    print(f"\nStarting sustained ML training for {TRAIN_MINUTES} minute(s)...")
    train_start = time.time()
    last_log_time = 0
    step = 0

    while True:
        now = time.time()
        elapsed = now - train_start

        if elapsed >= TRAIN_MINUTES * 60:
            break

        x, y = generate_batch()

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        step += 1

        if now - last_log_time >= LOG_INTERVAL_SEC:
            torch.cuda.synchronize()
            metrics = query_gpu_metrics()

            row = {
                "phase": "train",
                "wall_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_sec": round(elapsed, 2),
                **metrics,
                "loss": float(loss.item())
            }

            writer.writerow(row)
            f.flush()

            print(
                f"[TRAIN] t={elapsed:.1f}s | Step={step} | "
                f"Loss={loss.item():.4f} | "
                f"Temp={metrics['temperature_gpu_c']}C | "
                f"Power={metrics['power_draw_w']}W | "
                f"Util={metrics['utilization_gpu_percent']}% | "
                f"SM Clock={metrics['clock_sm_mhz']} MHz"
            )

            last_log_time = now

print(f"\nExperiment complete. Log saved to: {os.path.abspath(CSV_FILENAME)}")