<div align="center">

##  Rethinking Latent Representations in Behavior Cloning: An Information Bottleneck Approach for Robot Manipulation


[![python](https://img.shields.io/badge/-Python_3.8-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[**Project Page**](https://haoyizhu.github.io/spa/) | [**Paper**](https://haoyizhu.github.io/spa/static/images/paper.pdf) | [**arXiv**](https://arxiv.org/abs/2410.08208)

[Shuanghao Bai](https://baishuanghao.github.io/), [Wanqi Zhou](https://ellezwq.github.io/), [Pengxiang Ding](https://dingpx.github.io/),  Wei Zhao, [Donglin Wang](https://milab.westlake.edu.cn/), [Badong Chen](https://gr.xjtu.edu.cn/web/chenbd/home).
</div>




## 🛠️ Installation

Refer to [install.md](documents/install.md) for installation instructions.

Check [error_catch.md](documents/error_catch.md) for troubleshooting steps based on the errors I personally encountered during installation.


## 💾 Data Preparation

Refer to [download_dataset.md](documents/download_dataset.md) for instructions on downloading datasets.


## 💻 Baseine Model Preparation

Refer to [download_model.md](documents/download_model.md) for instructions on downloading pre-trained models of basleines.


## 📈 Usage

Refer to [usage.md](documents/usage.md) for instructions on training and evaluation.

## 📨 Contact

If you have any questions, please create an issue on this repository or contact us at baishuanghao@stu.xjtu.edu.cn.


## 📝 Citation

If you find our work useful, please consider citing:
```
```


## 🙏 Acknowledgements

For LIBERO experiments, our code is primarily built upon the [LIBERO source code](https://github.com/Lifelong-Robot-Learning/LIBERO). The baseline implementations are mainly adapted from the LIBERO source code, while the environment is set up using [robosuite](https://github.com/ARISE-Initiative/robosuite).

For CortexBench experiments, our code is primarily based on the [eai-vc source code](https://github.com/facebookresearch/eai-vc). The baselines reference [ResNet](https://github.com/KaimingHe/deep-residual-networks), [ViT](https://github.com/google-research/vision_transformer), [R3M](https://github.com/facebookresearch/r3m), [MVP](https://github.com/ir413/mvp), [VC-1](https://github.com/facebookresearch/eai-vc), [Voltron](https://github.com/siddk/voltron-robotics), and [MPI](https://github.com/OpenDriveLab/MPI), while the environment setup is based on [CortexBench](https://github.com/facebookresearch/eai-vc), [dmc2gym](https://github.com/denisyarats/dmc2gym), [MetaWorld](https://github.com/Farama-Foundation/Metaworld), [trifinger_simulation](https://github.com/open-dynamic-robot-initiative/trifinger_simulation), [gym](https://github.com/openai/gym), and [MuJoCo](https://github.com/google-deepmind/mujoco).

