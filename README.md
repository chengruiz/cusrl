# CusRL: Customizable Reinforcement Learning

CusRL is a flexible and modular reinforcement-learning framework designed for customization.
By breaking down complex algorithms into minimal components, it allows users to easily modify
or integrate components instead of rebuilding the entire algorithm from scratch, making it
particularly well-suited for advancing robotics learning.

> **Note:** This project is under **active development**, which means the interface is unstable
and breaking changes are likely to occur frequently.

## Setup

CusRL requires Python 3.10 or later. It can be installed via PyPI with:

```bash
# Choose one of the following:
# 1. Minimal installation
pip install cusrl
# 2. Install with export and logging utilities
pip install cusrl[all]
```

or by cloning this repository and installing it with:

```bash
git clone https://github.com/chengruiz/cusrl.git
# Choose one of the following:
# 1. Minimal installation
pip install -e . --config-settings editable_mode=strict
# 2. Install with optional dependencies
pip install -e .[all] --config-settings editable_mode=strict
# 3. Install dependencies for development
pip install -e .[dev] --config-settings editable_mode=strict
pre-commit install
```

## Quick Start

List all available experiments:

```bash
python -m cusrl list-experiments
```

Train a PPO agent and evaluate it:

```bash
python -m cusrl train -env MountainCar-v0 -alg ppo --logger tensorboard --seed 42
python -m cusrl play --checkpoint logs/MountainCar-v0:ppo
```

Or if you have [IssacLab](https://github.com/isaac-sim/IsaacLab) installed:

```bash
python -m cusrl train -env Isaac-Velocity-Rough-Anymal-C-v0 -alg ppo \
    --logger tensorboard --environment-args="--headless"
python -m cusrl play --checkpoint logs/Isaac-Velocity-Rough-Anymal-C-v0:ppo
```

Try distributed training:

```bash
torchrun --nproc-per-node=2 -m cusrl train -env Isaac-Velocity-Rough-Anymal-C-v0 \
    -alg ppo --logger tensorboard --environment-args="--headless"
```

## Highlights

CusRL provides a modular and extensible framework for RL with the following key features:

- **Modular Design**: Components are highly decoupled, allowing for free combination and customization
- **Diverse Network Architectures**: Support for MLP, CNN, RNNs, Transformer and custom architectures
- **Modern Training Techniques**: Built-in support for distributed and mixed-precision training

## Implemented Algorithms

- [Adversarial Motion Prior (AMP)](https://dl.acm.org/doi/10.1145/3450626.3459670)
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
  with [distinct lambda values](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e95475f5fb8edb9075bf9e25670d4013-Abstract-Conference.html)
- [Preserving Outputs Precisely, while Adaptively Rescaling Targets (Pop-Art)](https://proceedings.neurips.cc/paper/2016/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) with recurrent policy support
- [Random Network Distillation (RND)](https://arxiv.org/abs/1810.12894)
- Symmetry Augmentations:
  [Symmetric Architecture](https://dl.acm.org/doi/abs/10.1145/3359566.3360070),
  [Symmetric Data Augmentation](https://ieeexplore.ieee.org/abstract/document/10611493),
  [Symmetry Loss](https://dl.acm.org/doi/abs/10.1145/3197517.3201397)

## Cite

If you find this framework useful for your research, please consider citing our work on legged locomotion:

- [Efficient Learning of A Unified Policy For Whole-body Manipulation and Locomotion Skills](https://www.arxiv.org/abs/2507.04229), IROS 2025
- [Learning Accurate and Robust Velocity Tracking for Quadrupedal Robots](https://onlinelibrary.wiley.com/doi/10.1002/rob.70028), JFR 2025
- [Learning Safe Locomotion for Quadrupedal Robots by Derived-Action Optimization](https://ieeexplore.ieee.org/abstract/document/10802725), IROS 2024

## Acknowledgement

CusRL is based on or inspired by these projects:

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): Reliable implementations of reinforcement learning algorithms
- [RSL RL](https://github.com/leggedrobotics/rsl_rl): Fast and simple implementation of RL algorithms
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): GPU-accelerated simulation environments for robot research
- [robot_lab](https://github.com/fan-ziqi/robot_lab): RL extension library for robots, based on IsaacLab
