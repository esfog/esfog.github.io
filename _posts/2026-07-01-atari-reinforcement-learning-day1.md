---
layout: post
title: "Atari 强化学习笔记 (Day1) - 环境搭建与 GPU 验证"
tags: 强化学习 深度学习 Atari
---

# Day1 2026.7.1

**开发环境：** Python 3.13.9 / GPU Nvidia 5060 Laptop

---

## 一、环境准备和验证（由豆包AI制定，我进行了根据实际验证情况的补充修正）

### 步骤 1：下载并安装基础软件（顺序不能乱）

**安装 NVIDIA 最新显卡驱动**（我更新了驱动，CMD 返回 `NVIDIA-SMI 610.62  KMD Version: 610.62  CUDA UMD Version: 13.3`）

- 网址：https://www.nvidia.cn/Download/index.aspx
- 选型：GeForce RTX 5060 Laptop GPU，对应你的 Windows 系统
- 安装完成后重启电脑，打开 CMD 输入 `nvidia-smi`，确认 CUDA Version ≥ 12.6

**安装 Python 3.11 稳定版**（我本地已经有了 3.13 了，没换）

- 官网：https://www.python.org/downloads/release/python-311/
- 安装关键：勾选 "Add Python to PATH"

**安装 VS Code**

- 官网：https://code.visualstudio.com/
- 安装后在扩展市场安装：Python、Pylance 两个插件

### 步骤 2：创建专用 Python 虚拟环境（CMD/PowerShell 执行）

```bash
# 创建独立环境，存放 RL 游戏 AI 所有依赖（注意执行前切换到准备好的目录）
python -m venv rl_atari

# 激活环境（激活后前缀会出现 (rl_atari)）（不要用 PowerShell，默认权限不够）
rl_atari\Scripts\activate

# 更新 pip
python -m pip install --upgrade pip
```

### 步骤 3：一键安装全套依赖（复制整段运行）

```bash
# 1. GPU 版 PyTorch CUDA 12.6（适配 5060）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  （实际用了发现不行，5060 实际跑不了 12.6）
# 如果已经安装了没法直接覆盖，需要先卸载：
# pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 2. Gymnasium + Atari 模拟器底层 ALE
pip install "gymnasium[atari]"

# 3. Stable Baselines3（封装 DQN/PPO，快速出效果）
pip install stable-baselines3[extra]

# 4. 图像处理、绘图工具
pip install numpy opencv-python matplotlib
```

### 步骤 4：验证 GPU 是否可用（进入 Python 交互窗口）

```python
python
import torch
print(torch.cuda.is_available())          # 输出 True → 显卡加速正常；False 需要切换笔记本独显模式、重装驱动
print(torch.cuda.get_device_capability())  # 输出 (12, 0) 代表识别你的 RTX5060 算力
```

### 步骤 5：新建测试代码，验证 Atari 模拟器可视化（test_pong.py）

> **注意：** 在实际用 VSCode 编写运行 Python 之前，需要让 VSCode 识别你的 Python 虚拟环境。在 VSCode 里 `Ctrl+Shift+P` 输入 `Python: Select Interpreter`，然后将之前准备好的 `rl_atari\Scripts\python.exe` 关联上。可以通过右下角是否显示 `rl_atari` 来确认是否生效。

```python
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# human 模式弹出游戏窗口
env = gym.make("ALE/Pong-v5", render_mode="human")

obs, info = env.reset()

# 随机操作 1000 帧测试画面
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

**验收标准：** 弹出乒乓球游戏窗口，挡板自动随机移动，无报错。

### 步骤 6：测试 SB3 封装 DQN 最简训练 Demo（核心工具验证）（sb3_demo.py）

```python
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py
gym.register_envs(ale_py)

# 1. 创建原生游戏环境
env = gym.make("ALE/Pong-v5", render_mode="human")
# 2. 套 Atari 标准包装器：灰度、裁剪、缩放 84×84、4 帧堆叠，大幅度减小观测尺寸
env = AtariWrapper(env)
# 3. 调低回放池容量，避免爆内存，RTX5060 Laptop 直接用这个参数
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=80000,
    batch_size=32,
    learning_rate=1e-4
)
# 少量步数测试连通性
model.learn(total_timesteps=50000)
model.save("pong_test_model")
env.close()
```

运行后控制台打印训练日志，窗口同步显示游戏画面，代表 SB3 完全可用。

**我本机训练日志（得等一会）：**

```
| rollout/            |          |
|    ep_len_mean      | 213      |
|    ep_rew_mean      | -21      |
|    exploration_rate | 0.838    |
| time/               |          |
|    episodes         | 4        |
|    fps              | 3        |
|    time_elapsed     | 235      |
|    total_timesteps  | 851      |
| train/              |          |
|    learning_rate    | 0.0001   |
|    loss             | 0.0395   |
|    n_updates        | 187      |
```

### 步骤 7：可选附加测试（手动操控游戏）（manual_test.py）

```python
from gymnasium.utils.play import play
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

# 关键：render_mode 改成 rgb_array
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
play(env, zoom=2)
```

键盘直接操控打砖块，直观理解游戏动作空间。

---

## 二、今晚完成这套操作，你能收获什么（由豆包AI制定）

1. **完整可复用、GPU 加速的强化学习专用开发环境**，后续两周所有 RL 代码都在此环境运行
2. **确认 RTX5060 显卡 CUDA 正常工作**，显存、算力满足 Atari 视觉 DQN 训练需求
3. **跑通两条核心管线：**
   - Gym 原生模拟器：画面、动作、奖励、游戏结束整套交互闭环
   - SB3 封装 DQN：一行 CNN 策略即可基于像素画面训练 AI
4. **验证三件核心工具全部无兼容 bug：** Python 3.11、Gymnasium (ALE)、Stable Baselines3
5. **建立直观视觉感受：** 亲眼看到 AI 读取游戏画面并输出操作，建立「画面输入 → 网络 → 动作输出」全局认知
6. **掌握虚拟环境创建、库安装、代码运行等基础 Python 操作**，弥补你 Python 零基础短板

---

## 三、为什么必须今晚做完，对后续学习的支撑作用（对应你两周 Gym 学习计划）（由豆包AI制定）

**1. 消除环境干扰，后续只专注算法逻辑**

如果今晚不搭好，明天开始学 MDP、DQN 理论时，会频繁被安装报错、CUDA 失效、库版本冲突打断思路；你 Python 零基础，环境问题会大量占用理论学习时间。

**2. SB3 是第一周核心学习载体**

第一周前 7 天的核心实操全部依赖 SB3：快速训练 Pong / 打砖块、调参、观察奖励收敛。今晚验证 SB3 可用，明天可以直接上手训练、观察 AI 变强，不用额外花半天修复工具。

**3. 模拟器可视化是理解理论的直观载体**

明天学习 MDP、状态 S、动作 A、奖励 R 时，你可以随时弹出 Pong 窗口对照：

- 画面像素 = 状态 S
- 左右移动按键 = 动作 A
- 得分增减 = 奖励 R

没有今晚搭建好的可视化窗口，抽象公式很难结合游戏场景理解。

**4. 为第二周手写原生 DQN 做底层验证基准**

第二周你会抛弃 SB3、自己手写 CNN、回放池、损失函数。今晚确认 SB3 能正常收敛，后续手写代码训练不收敛时，可以快速定位是自己代码逻辑 bug，而非环境 / 显卡 / 库底层问题，大幅减少调试成本。

**5. 提前熟悉图像张量逻辑，贴合你的客户端渲染功底**

今晚运行代码会接触图像数组、灰度画面、分辨率维度概念，明天学习 CNN 卷积时，你可以用渲染纹理、像素缓冲区的知识快速类比，降低深度学习理解门槛。

---

## 四、核心总结

今晚工作本质：**扫清所有工程环境障碍，把后续两周的学习重心完全锁定在「强化学习 + 视觉 DQN 原理」上。** 不提前完成环境搭建，后续每一天理论 + 代码练习都会被底层安装、兼容问题分流大量精力，打乱每日 1～2 小时的学习节奏。

---

## 五、其他学习过程中的备注

**1. 关于 PyTorch CUDA 版本兼容性**

你的 RTX5060 Laptop 是 sm_120（算力 12.0）；你装的 PyTorch cu126 构建包最高只支持到 sm_90（RTX40 系、Ada 架构），不支持新的 Blackwell 12 代 GPU；因此 GPU 无法加载 CUDA 算子内核，直接抛出 `no kernel image is available`，无法使用显卡加速训练。

**2. 关于 AtariWrapper 的重要性**

你现在的环境原始观测是 (3, 210, 160) 彩色大图，哪怕 buffer_size 降到 5 万，数组依然巨大，只是崩溃晚一点，测试大概率还是内存报错。

SB3 训练 Atari **必须**用官方封装器做灰度、裁剪、缩放到 84×84、4 帧堆叠，把观测压缩成 (4, 84, 84)，内存占用直接减少几十倍。

你写的 `buffer_size=50000`、`batch_size=32` 参数本身没问题，32G 内存完全扛得住；**不加 AtariWrapper 是致命漏洞：**
- 内存爆炸报错
- 和 DeepMind 原版 DQN、那本第三版英文书的标准输入流程不一致，后续手写 DQN 会逻辑脱节
- CNN 网络输入维度不匹配，训练效果极差

**3.**（待补充）
