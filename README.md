# ⚡ Eco-Twin Smart Microgrid (RL Agent)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable_Baselines3-EE4C2C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Execution_Server-009688.svg)
![Status](https://img.shields.io/badge/Status-In_Progress-orange.svg)

An end-to-end Reinforcement Learning architecture designed to autonomously manage a physical microgrid. This project bridges a custom physics simulation environment with a production-ready API for IoT hardware execution.

## 📖 Project Overview
Optimizing battery storage and solar generation requires predicting the future. This project uses a **Proximal Policy Optimization (PPO)** agent to learn the optimal charge/discharge strategy for a campus microgrid, minimizing reliance on the public grid and maximizing solar efficiency.

The architecture is split into two distinct phases to ensure scalability:
1. **The Simulation Engine:** A mathematically rigorous `Gymnasium` environment simulating hardware limits, energy loss, and historical weather patterns.
2. **The Execution API:** A lightweight `FastAPI` server that acts as a bridge, ingesting real-time IoT hardware states (ESP32) and fetching live internet weather data to generate instantaneous hardware commands.

## 🧠 System Architecture

- **State Vector (8 Inputs):** `[Battery_SoC, Solar_Now, Solar_+1h, Solar_+2h, Solar_+3h, Campus_Demand, Time_Sin, Time_Cos]`
- **Action Space:** Continuous `[-1.0, 1.0]` (Full Discharge to Full Charge).
- **Reward Function:** Heavily penalizes public grid energy purchases and strictly enforces battery Depth of Discharge (DoD) health limits.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
