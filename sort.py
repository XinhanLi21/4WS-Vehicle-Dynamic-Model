import os, math, pathlib, numpy as np, matplotlib.pyplot as plt
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.io import loadmat
from rl_td3_quick import FourWSEnv, MPC4WS, VehicleModelPublic


def evaluate_single(checkpoint, dll, path_mat, vx_kmh=80/3.6, Ts=0.05, dt=0.01):
    """
    评估单个checkpoint，返回平均横向误差
    """
    ref_path = np.asarray(loadmat(path_mat)["path5"], dtype=float)

    env = DummyVecEnv([lambda: FourWSEnv(ref_path, dll, vx=vx_kmh, Ts=Ts, dt=dt)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model: TD3 = TD3.load(checkpoint, env=env, device=device, print_system_info=False)

    obs = env.reset()
    done = False

    lat_err_hist = []
    fenv: FourWSEnv = env.envs[0]

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        lat_err_hist.append(float(obs[0][0]))

    lat_err_hist = np.asarray(lat_err_hist)
    mean_lat_err = np.mean(np.abs(lat_err_hist))

    return mean_lat_err


def evaluate_all(checkpoints_dir, dll, path_mat):
    """
    批量评估所有checkpoint，返回误差排序和最优模型
    """
    # 查找所有 zip checkpoint
    checkpoints = sorted([os.path.join(checkpoints_dir, f)
                          for f in os.listdir(checkpoints_dir)
                          if f.endswith(".zip")])

    results = []

    for ckpt in checkpoints:
        print(f"\n📌 正在评估 {ckpt} ...")
        mean_lat_err = evaluate_single(ckpt, dll, path_mat)
        print(f"➡️ 平均横向误差: {mean_lat_err:.4f} m")
        results.append((ckpt, mean_lat_err))

    # 排序
    results.sort(key=lambda x: x[1])
    print("\n====== 📊 全部模型横向误差排序 ======")
    for ckpt, err in results:
        print(f"{os.path.basename(ckpt)} -> 平均横向误差: {err:.4f} m")

    best_ckpt, best_err = results[0]
    print("\n✅ 最佳模型:")
    print(f"{os.path.basename(best_ckpt)} -> 平均横向误差: {best_err:.4f} m")

    return best_ckpt, best_err


if __name__ == "__main__":
    # >>>>>>>> 配置路径 <<<<<<<<
    CHECKPOINT_DIR = "checkpoints"
    DLL            = r"D:\chrome-download\5\5.4\vehiclemodel_public_0326_win64.dll"
    PATH_MAT       = "path5_mpc4ws.mat"

    evaluate_all(CHECKPOINT_DIR, DLL, PATH_MAT)
