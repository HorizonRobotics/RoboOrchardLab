# Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=IYLvsCQAAAAJ&hl" target="_blank">Shuo Wang</a>,
    <text>Yucheng Wang</text>,
    <text>Guoxin Lian</text>,
    <a href="https://yongcaiwang.github.io/" target="_blank">Yongcai Wang</a>,
    <text>Maiyue Chen</text>,
    <text>Kaihui Wang</text>,
    <text>Bo Zhang</text>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>,
    <a>Yutian Zhou</a>,
    <a>Wanting Li</a>,
    <text>Deying Li</text>
    <a href="https://zhaoxinf.github.io/" target="_blank">Zhaoxin Fan</a>,
</div>


<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/progress-think" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2511.17097" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/Progress-Think" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction
Progress-Think enables more accurate Vision-Language Navigation by modeling semantic progress from visual observations to guide policies coherently over long multi-step instructions.

![](https://horizonrobotics.github.io/robot_lab/progress-think/stats/x1.png)



## Installation

### 1. Set up Progress-Think

```
conda create -n progress_think python=3.10
conda activate progress_think
pip install ".[progress_think]"

cd projects/progress_think
pip install -r requirements.txt

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```
 
### 2. Set up the Habitat environment


Progress-Think relies on **Habitat-Sim 0.1.7** for simulation and dataset generation.  
Please follow the official build-from-source guide:  
https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md

Then install the habitat-lab 0.1.7 dependency.
``` 
# Install habitat-lab
cd projects/progress_think
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all
```

### 3. Set up VLN-CE Extensions
```
cd projects/progress_think
git clone https://github.com/markinruc/VLN_CE.git
```

## Inference Data Preparation

Please download the Matterport3D scene data and R2R-CE/RxR-CE datasets following [VLN-CE](https://github.com/jacobkrantz/VLN-CE). You can refer to the following file structure or modify the config in ./VLN-CE

```graphql
data/datasets
├─ RxR_VLNCE_v0
|   ├─ train
|   |    ├─ train_guide.json.gz
|   |    ├─ train_guide_gt.json.gz
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ val_unseen_guide_gt.json.gz
|   ├─ ...
├─ R2R_VLNCE_v1-3_preprocessed
|   ├─ train
|   |    ├─ train.json.gz
|   |    ├─ train_gt.json.gz
|   ├─ val_unseen
|   |    ├─ val_unseen.json.gz
|   |    ├─ val_unseen_gt.json.gz
data/scene_dataset
├─ mp3d
|   ├─ ...
|   |    ├─ ....glb
|   |    ├─ ...
|   ├─ ...
```


## Inference

Please modify the model-path and result-path in run_infer.sh.
```
cd projects/progress_think
./run_infer.sh
```
Results will be saved in the specified result-path. Run the following command to obtain the final metrics:
```
python analyze_results.py --path result-path
```


## Citation

```bibtex
@article{wang2025progress,
  title={Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation},
  author={Wang, Shuo and Wang, Yucheng and Lian, Guoxin and Wang, Yongcai and Chen, Maiyue and Wang, Kaihui and Zhang, Bo and Su, Zhizhong and Zhou, Yutian and Li, Wanting and others},
  journal={arXiv preprint arXiv:2511.17097},
  year={2025}
}
```

## Acknowledgments
Our code is based in part on [VILA](https://github.com/NVlabs/VILA), [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE), and [VLN-CE](https://github.com/jacobkrantz/VLN-CE). Thanks for their great works.

