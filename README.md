# UPP Combi Route Optimization via UCB Multi-Armed Bandit

## Overview

This repository implements a demand prediction and multi-armed bandit optimization system designed to minimize student waiting times for university transport routes serving Universidad Politécnica de Pachuca (UPP).

The system evaluates trade-offs between vehicle capacity utilization and passenger waiting duration, contrasting conventional fixed-capacity dispatch policies against dynamic exploration-exploitation policies using the Upper Confidence Bound (UCB1) algorithm.

## Problem Statement

Transport routes operating from departure points such as Puente de Tuzos experience low passenger arrival rates (~1 student per 7 minutes during off-peak hours). Strict adherence to full-vehicle dispatch policies (18 passengers) results in excessive cumulative waiting times for early-arriving passengers.

This project simulates 21 days of operational demand, fits simple linear regression models to predict filling times and travel durations, and optimizes departure decision policies using reinforcement learning primitives.

## Project Structure

```text
upp-route-rl/
├── src/
│   ├── config.py                 # System parameters and route configurations
│   ├── data_generator.py         # Stochastic Poisson demand generation engine
│   ├── regression_model.py       # Ordinary Least Squares regression pipeline
│   └── ucb_bandit.py             # UCB multi-armed bandit optimization engine
├── data/                         # Synthetic operational datasets
├── outputs/                      # Generated analytical plots and figures
├── Documentation/                # Academic report sources (LaTeX)
├── 01_generate_dataset.py        # Dataset simulation entry point
├── 02_train_regression.py        # Regression model training entry point
├── 03_run_ucb.py                 # UCB simulation entry point
├── requirements.txt              # Project dependencies
└── README.md                     # Technical documentation
```

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/upp-route-rl.git
   cd upp-route-rl
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Workflow

1. **Dataset Generation:**
   ```bash
   python 01_generate_dataset.py
   ```

2. **Linear Regression Training:**
   ```bash
   python 02_train_regression.py
   ```

3. **UCB Policy Optimization:**
   ```bash
   python 03_run_ucb.py
   ```

## License and Academic Credits

Developed as part of the Artificial Intelligence curriculum at Universidad Politécnica de Pachuca (UPP).
