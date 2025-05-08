# eval_td3_checkpoint.py
import os, math, pathlib, numpy as np, matplotlib.pyplot as plt

import torch
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.io import loadmat

# ------------- 引入训练时定义的 FourWSEnv -----------------
from rl_td3_quick import FourWSEnv, MPC4WS, VehicleModelPublic


def evaluate(checkpoint, dll, path_mat, out_dir="eval_results", vx_kmh=80/3.6, Ts=0.05, dt=0.01):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 加载参考路径 ----------------------------------------------------
    ref_path = np.asarray(loadmat(path_mat)["path5"], dtype=float)

    # ---- 创建单环境 ------------------------------------------------------
    env = DummyVecEnv([lambda: FourWSEnv(ref_path, dll, vx=vx_kmh, Ts=Ts, dt=dt)])

    # ---- 恢复模型 --------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: TD3 = TD3.load(checkpoint, env=env, device=device, print_system_info=True)

    # ---- 运行回放 ----------------------------------------------------
    obs= env.reset()
    done = False

    xs, ys = [], []
    delta_f_hist, delta_r_hist = [], []
    lat_err_hist = []

    fenv: FourWSEnv = env.envs[0]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        X, Y, _, _, _, _ = fenv.veh._get_current_observation()
        xs.append(X); ys.append(Y)

        delta_f_hist.append(fenv.delta_f)
        delta_r_hist.append(fenv.beta * fenv.delta_f)
        lat_err_hist.append(float(obs[0][0]))

    xs, ys = np.asarray(xs), np.asarray(ys)
    delta_f_hist, delta_r_hist = np.asarray(delta_f_hist), np.asarray(delta_r_hist)
    lat_err_hist = np.asarray(lat_err_hist)
    time_axis = np.arange(len(delta_f_hist)) * Ts

    mean_lat_err = np.mean(np.abs(lat_err_hist))
    print(f"平均横向误差 = {mean_lat_err:.4f} m")

    # 保存 CSV
    np.savetxt(os.path.join(out_dir, "xy_traj.csv"),
               np.column_stack([xs, ys]),
               delimiter=",", header="X,Y", comments='')
    np.savetxt(os.path.join(out_dir, "steering_angles.csv"),
               np.column_stack([time_axis, delta_f_hist, delta_r_hist]),
               delimiter=",", header="t,delta_f(rad),delta_r(rad)", comments='')

    # 绘图 XY
    plt.figure(figsize=(4, 5))
    plt.plot(ref_path[:, 0], ref_path[:, 1], "k--", label="reference")
    plt.plot(xs, ys, "r-", label="actual")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title(f"XY Trajectory – |lat_err|_avg={mean_lat_err:.3f} m")
    plt.ylim(-1, 6); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "xy_traj.png"), dpi=150)
    plt.close()

    # 绘图 前后轮
    plt.figure(figsize=(6, 3))
    plt.plot(time_axis, np.degrees(delta_f_hist), label="front wheel δ_f")
    plt.plot(time_axis, np.degrees(delta_r_hist), label="rear wheel δ_r")
    plt.xlabel("time (s)"); plt.ylabel("steer angle (deg)")
    plt.title("Steering Angles vs. Time")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "steer_angles.png"), dpi=150)
    plt.close()

    print(f"✓ 评估完成，结果保存在 “{out_dir}” 文件夹内")


# =========================== 直接运行 =============================

if __name__ == "__main__":
    # >>>>>>>>> 这里配置你的路径 <<<<<<<<<<
    CHECKPOINT = "checkpoints/td3_step_5888000_steps.zip"
    DLL        = r"D:\chrome-download\5\5.4\vehiclemodel_public_0326_win64.dll"
    PATH_MAT   = "path5_mpc4ws.mat"
    OUT_DIR    = "my_eval"

    evaluate(CHECKPOINT, DLL, PATH_MAT, OUT_DIR)




