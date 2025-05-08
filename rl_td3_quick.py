# rl_td3_fast_spawn.py  —— 多进程 (spawn) + checkpoint + 轨迹
# =============================================================
import multiprocessing as mp
mp.set_start_method("spawn", force=True)        # Windows 多进程安全

import os, math, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.io import loadmat
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.utils import set_random_seed

try:
    from mpc_4ws import MPC4WS
except ModuleNotFoundError:
    from rl_mpc import MPC4WS
from wrapper import VehicleModelPublic


# ---------------- 单环境封装 ----------------------------------
class FourWSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, path, dll, vx=80/3.6, Ts=0.05, dt=0.01, beta_lim=1):
        super().__init__()
        self.path, self.vx = path, vx
        self.Ts, self.dt   = Ts, dt
        self.ratio         = int(Ts / dt)      # 5
        self.beta_lim, self.act_lim = beta_lim, 0.3

        self.action_space      = Box(-self.act_lim, self.act_lim, (1,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (3,), np.float32)

        self.ctrl = MPC4WS(path, Ts=Ts, vx=vx)
        self.veh  = VehicleModelPublic(str(pathlib.Path(dll)))
        self.traj = []

    # ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ctrl.reset(); self.veh.terminate()
        self.veh.initial(0.0, self.vx, 0.0)
        self.t = self.beta = self.delta_f = 0.0
        self.traj = []
        return self._obs(), {}

    # ----------------------------------------------------------
    def step(self, action):
        self.beta = float(np.clip(self.beta + action[0], -self.beta_lim, self.beta_lim))
        # 车辆积分
        for _ in range(self.ratio):
            ov = self.veh.step(self.delta_f, self.beta * self.delta_f)
            self.traj.append((ov[0], ov[1]))
            self.t += self.dt
        # MPC
        y = self.ctrl.step(self.t, self._assemble_u(ov, self.beta * self.delta_f))
        self.delta_f = float(y[0])
        lat_err, yaw_err = float(y[2]), float(y[3])

        reward = -20.0 * lat_err**2
        done   = self.t >= 12.0
        info   = {"traj": np.asarray(self.traj, np.float32)} if done else {}

        return np.array([lat_err, yaw_err, self.beta], np.float32), reward, done, False, info

    # ----------------------------------------------------------
    def _obs(self):
        ov  = self.veh._get_current_observation()
        lat, yaw = self.ctrl.step(self.t, self._assemble_u(ov, 0.0))[2:4]
        return np.array([lat, yaw, self.beta], np.float32)

    def _assemble_u(self, ov, delta_r):
        X, Y, yaw, Vx, Vy, r = ov
        return np.array([Vy*3.6, Vx*3.6,
                         math.degrees(yaw), math.degrees(r),
                         Y, X, math.degrees(delta_r), 0.0], np.float32)

    def close(self): self.veh.terminate()


# ---------------- 轨迹 PNG / CSV 回调 -------------------------
class TrajCallback(BaseCallback):
    def __init__(self, ref_path, out_dir="trajectories_fast"):
        super().__init__()
        self.ref = ref_path
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.i = 0

    def _on_step(self):  # 必须实现
        return True

    def _on_rollout_end(self):
        for info in self.locals["infos"]:
            if "traj" in info:
                t = info["traj"]
                np.savetxt(f"{self.out}/ep{self.i:04d}.csv", t, delimiter=",")
                fig, ax = plt.subplots(figsize=(4, 5))
                ax.plot(self.ref[:, 0], self.ref[:, 1], "k--", label="reference")
                ax.plot(t[:, 0], t[:, 1], "r-", label="actual")
                ax.set_ylim(-1, 6)
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.set_xlabel("X"); ax.set_ylabel("Y")
                fig.tight_layout()
                fig.savefig(f"{self.out}/ep{self.i:04d}.png", dpi=150)
                plt.close(fig)
                self.i += 1
        return True


# ---------------- VecEnv 工厂 ---------------------------------
def make_env(rank, path, dll, seed=0):
    def _init():
        env = FourWSEnv(path, dll)
        env.reset(seed=seed + rank)
        return env
    return _init


# =========================== main =============================
if __name__ == "__main__":
    DLL  = r"D:\OneDriveTemp\OneDrive\Desktop\5.4\vehiclemodel_public_0326_win64.dll"
    PATH = np.asarray(loadmat("path5_mpc4ws.mat")["path5"], float)

    n_envs = 8                     # 可按 CPU 调整
    set_random_seed(42)

    env = SubprocVecEnv(
        [make_env(i, PATH, DLL, 42) for i in range(n_envs)],
        start_method="spawn"        # ★ 指定 spawn
    )
    env = VecMonitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| envs:", n_envs)

    td3 = TD3(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=dict(net_arch=[400, 300]),   # 大网络
        buffer_size=300_000,
        batch_size=2048,
        learning_rate=2e-3,
        gradient_steps=8,
        action_noise=NormalActionNoise(np.zeros(1), 0.1*np.ones(1)),
        gamma=0.98,
        tau=0.005,
        train_freq=(1, "step"),
        verbose=1
    )

    # ---------- 组合回调：轨迹 + checkpoint --------------------
    ckpt = CheckpointCallback(
        save_freq=8_000,            # 每 8000 环境步保存
        save_path="checkpoints",
        name_prefix="td3_step"
    )
    cb_list = CallbackList([TrajCallback(PATH), ckpt])

    td3.learn(total_timesteps=8_000_000,
              callback=cb_list,
              progress_bar=True)

    td3.save("td3_mpc_big_final_spawn.zip")
    print("✔ training finished — final model saved")
