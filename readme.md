<div align="center">

# Rethinking Latent Representations in Behavior Cloning: An Information Bottleneck Approach for Robot Manipulation


[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.11+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[**Project Page**](https://haoyizhu.github.io/spa/) | [**Paper**](https://haoyizhu.github.io/spa/static/images/paper.pdf) | [**arXiv**](https://arxiv.org/abs/2410.08208) | [**HuggingFace Model**](https://huggingface.co/HaoyiZhu/SPA) | [**Real-World Codebase**](https://github.com/HaoyiZhu/RealRobot) | [**Twitter/X**](https://x.com/HaoyiZhu/status/1844675411760013471)

[Haoyi Zhu](https://www.haoyizhu.site/), [Honghui Yang](https://hhyangcs.github.io/), [Yating Wang](https://scholar.google.com/citations?hl=zh-CN&user=5SuBWh0AAAAJ),  [Jiange Yang](https://yangjiangeyjg.github.io/), [Liming Wang](https://wanglimin.github.io/), [Tong He](http://tonghe90.github.io/)
</div>




## :hammer: Installation
<details>
<summary><b>Basics</b></summary>

```bash
# clone project
git clone https://github.com/BaiShuanghao/BC-IB.git
cd BC-IB

# crerate conda environment
conda create -n bcib python=3.8
conda activate bcib

# install PyTorch, please refer to https://pytorch.org/get-started/previous-versions/ for other CUDA versions
# e.g. cuda 11.8:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# install mujoco (both mujoco200 and mujoco210 are supported)
# e.g. mujoco200:
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -P ~/.mujoco
unzip ~/.mujoco/mujoco200_linux.zip
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
wget https://www.roboti.us/file/mjkey.txt -P ~/.mujoco

cd third_party/envs
cd mujoco-py
pip install -e .
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

# 环境变量及依赖项
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl
source ~/.bashrc

sudo apt update && sudo apt upgrade -y
sudo apt install libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libglfw3 libglfw3-dev libglew-dev libosmesa6 libosmesa6-dev libgles2-mesa
conda install -c conda-forge mesalib

# install LIBERO
cd third_party/benchmarks
cd ../../benchmarks
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd libero
pip install -r requirements.txt
cd ../LIBERO
pip install -e.
cd ../../..

# run 
python notebooks/test_env/test_egl.py
python notebooks/test_env/test_fabric.py
python notebooks/test_env/test_mujoco.py
python notebooks/test_env/test_robosuite.py

# install CortexBench
cd third_party/envs
pip install -e ./mj_envs
pip install -e ./mjrl
cd ../benchmarks
pip install -e ./dmc2gym
pip install -e ./metaworld
pip install -e ./trifinger_simulation
cd eai-vc
pip install -e ./vc_models
pip install -e ./cortexbench/mujoco_vc 
pip install -e ./cortexbench/trifinger_vc
cd ../../..

# install Baselines
cd third_party/methods
pip install -e ./MPI
pip install -e ./r3m
pip install -e ./voltron-robotics
cd ../..

```
</details>


conda install pinocchio -c conda-forge

conda env create -f environment.yml
conda env update --file environment.yml --prune