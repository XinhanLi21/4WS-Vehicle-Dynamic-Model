
# 🚗 4WS Reinforcement Learning (TD3 + Simulink Vehicle Model)

This project is a complete pipeline for vehicle model control using Reinforcement Learning (TD3).  
The workflow includes:

- Simulink vehicle model → C code → DLL
- Python + Stable-Baselines3 TD3 training → model saving
- Inference and plotting trajectories, steering angles (δ_f, δ_r) and sideslip angle (β).

## 📦 Project Structure

```
4ws_rl/
├── path5_mpc4ws.mat                  # Path data
├── vehiclemodel_public_0326_win64.dll# Vehicle model DLL
├── wrapper.py                        # DLL interface
├── rl_td3_quick.py                   # Training script
├── plot.py                           # Plotting script (visualization)
├── checkpoints/                      # Saved models
├── environment.yml                   # Conda environment file
└── README.md                         # Project introduction (this file)
```

## 🚀 Quick Start

### 1️⃣ Install Conda

If you don't have Anaconda or Miniconda installed, install Miniconda first:

https://docs.conda.io/en/latest/miniconda.html

### 2️⃣ Create environment

```bash
conda env create -f environment.yml
conda activate 4ws_rl
```

✅ This will automatically install the required:

- Python 3.12
- PyTorch 2.5.1 + CUDA 12.4
- Stable-Baselines3 2.6.0
- Gymnasium 1.1.1
- Numpy 2.2.4
- Matplotlib, Scipy, tqdm, rich, cloudpickle, etc.

### 3️⃣ Train TD3 agent

```bash
python rl_td3_quick.py
```

The training progress and model checkpoints will be saved into `checkpoints/`.

### 4️⃣ Visualize trained model

After training or using existing checkpoints:

```bash
python plot.py
```

This will generate:

```
steering_curves.png    # Front/rear steering angle + beta curve
traj_final.png         # X-Y trajectory
```


