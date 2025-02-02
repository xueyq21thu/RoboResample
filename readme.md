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

# install basic packages
pip3 install -r requirements.txt
```
</details>

pip install gym==0.23.1
conda install pinocchio -c conda-forge