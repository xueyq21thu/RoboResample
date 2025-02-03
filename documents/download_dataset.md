## Dataset Downloading

### 1. LIBERO 

For LIBERO, you can refer to the [LIBERO source code](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Datasets).

Alternatively, you can download the datasets locally using the following links:

- [LIBERO-Goal](https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip)
- [LIBERO-Object](https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip)
- [LIBERO-Spatial](https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip)
- [LIBERO-100](https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip)

The structure is:

```
datasets
│
├── libero_10           
│
├── libero_90     
│
├── libero_goal   
│
├── libero_object
│
└── libero_spatial
```

### 2. CortexBench

For a detailed introduction to CortexBench, refer to [this link](https://github.com/facebookresearch/eai-vc/tree/main/cortexbench).
In our experiments, we only use four suites from CortexBench: Adroit, DMControl, MetaWorld, and Trifinger.

For dataset download links, see [this link](https://github.com/facebookresearch/eai-vc/blob/main/cortexbench/DATASETS.md).


The structure is:

```
cortexbench
│
├── adroit-expert-v1.0      
│    ├── pen-v0.pickle
│    └── relocate-v0.pickle
│
├── dmc-expert-v1.0     
│    ├── dmc_cheetah_run-v1.pickle
│    ├── ...
│    └── ...
│
├── metaworld-expert-v1.0   
│    ├── assembly-v2-goal-observable.pickle
│    ├── ...
│    └── ...
│
└── trifinger-demos
     ├── move
     │    ├── demo-0000
     │    ├── ...
     │    └── ...
     │
     └── reach
          ├── demo-0000
          ├── ...
          └── ...
```
