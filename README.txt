# GPU ML Workload Reliability Study

This repository contains code, telemetry logs, figures, and supporting materials for a capstone study on reliability-relevant stress behavior of an NVIDIA GeForce RTX 3080 under sustained machine-learning workloads.


This project runs synthetic PyTorch machine-learning workloads on an NVIDIA GPU while logging telemetry with `nvidia-smi`. The goal is to characterize how sustained ML training affects GPU temperature, power draw, utilization, clock behavior, and memory usage.

The experiments include:

1. **Experiment 1: Sustained Control Workload**
   - Runs a large synthetic ML workload for one hour.
   - Used to observe startup transient behavior and steady-state GPU stress.

2. **Experiment 2: Batch-Size Scaling**
   - Keeps model dimensions fixed.
   - Changes batch size to evaluate how batch size affects power, temperature, and utilization.

3. **Experiment 3: Model-Complexity Scaling**
   - Changes model dimensions.
   - Evaluates whether model complexity affects GPU stress more strongly than batch size.

The main finding is that GPU utilization alone is not sufficient for characterizing workload intensity. Power draw and model structure provide more useful insight into reliability-relevant stress behavior.

---

## System Used in This Study

The original experiments were performed on:

- GPU: NVIDIA GeForce RTX 3080
- CPU: Intel Core i9-12900KF
- RAM: 64 GB
- OS: Windows 11 Home
- GPU Driver: NVIDIA 591.74
- Framework: PyTorch with CUDA acceleration
- Telemetry Tool: NVIDIA System Management Interface (`nvidia-smi`)
- Logging Interval: 2 seconds
- Ambient Room Temperature: approximately 70°F / 21°C

Other NVIDIA GPUs may be used, but results will vary depending on GPU model, cooling, driver version, case airflow, power limits, and ambient temperature.

---

## Installation and Setup

1. Install NVIDIA GPU Driver

Install the latest NVIDIA driver for your GPU from NVIDIA.

After installation, open Command Prompt and run:

nvidia-smi


2. Run the experiment script 

Make sure all background apps and processes are closed. Let the computer sit in idle for around 5 minutes.

Run the code in command prompt: python experiment_ml_stress_test.py

The idle and training procedure should automatically start, and you should not use the computer for the next 65 minutes.

Once completed, the output log "gpu_ml_stress_log.csv" should automatically be created.

Remember to rename the output log if doing multiple tests before starting the experiment script again.














