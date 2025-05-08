4WS Reinforcement Learning Project
基于四轮转向（4WS）车辆模型，结合MPC控制器与强化学习TD3算法，实现自动路径跟踪与优化。
本项目从 Simulink 动力学建模 → 生成DLL → Python训练TD3 → 结果可视化，提供了完整流程与代码。

<!-- 可选：你可以自行截图轨迹图和转角图存到 images/overview.png，这样github页面就会显示！ -->

📖 目录
项目简介

项目结构

安装与环境

训练TD3智能体

回放与可视化

注意事项

致谢与引用

项目简介
本项目利用Simulink生成的车辆动力学DLL文件，结合MPC控制器与TD3算法，训练智能体调节车辆后轮转角（β角），从而优化路径跟踪效果。
训练完成后可输出前后轮转角、β角、以及车辆实际轨迹与目标轨迹对比图。

主要技术栈：

Simulink + Simulink Coder (生成DLL)

Python + Stable-Baselines3 (TD3)

Matplotlib + Scipy + Gymnasium (可视化与仿真环境)

项目结构
pgsql
复制
编辑
simulink/                      # Simulink模型（原始文件）
vehiclemodel_public_0326_win64.dll  # DLL文件（Simulink生成的动力学模型）
path5_mpc4ws.mat               # 目标路径文件
rl_td3_quick.py                # TD3训练脚本
plot.py                        # 结果回放与可视化脚本
sort.py                        # 辅助工具
wrapper.py                     # DLL调用封装
checkpoints/                   # 训练好的模型（自动生成）
安装与环境
Python依赖
推荐使用 Python 3.8+ / Anaconda / Miniconda

bash
复制
编辑
pip install numpy scipy matplotlib gymnasium stable-baselines3
推荐使用 PyTorch GPU 版本，可自动加速训练。

额外说明
Simulink生成的DLL文件目前仅支持Windows系统。

训练TD3智能体
bash
复制
编辑
python rl_td3_quick.py
训练过程中会自动保存模型至 checkpoints/ 目录

每隔一定步数保存一次checkpoint，方便中断恢复

训练数据示例（训练完成后）
python
复制
编辑
checkpoints/
├── td3_step_8000_steps.zip
├── td3_step_16000_steps.zip
├── ...
└── td3_mpc_big_final_spawn.zip
回放与可视化
bash
复制
编辑
python plot.py
回放后会自动生成以下图片：

✅ steering_curves.png (前后轮转角 + β曲线)

✅ traj_final.png (X-Y轨迹与目标路径对比)

图片保存目录为当前路径下。
