# MapDream: Task-Driven Map Learning for Vision-Language Navigation

<div align="center" class="authors">
    <text>Guoxin Lian</text>,
    <a href="https://scholar.google.com/citations?user=IYLvsCQAAAAJ&hl" target="_blank">Shuo Wang</a>,
    <text>Yucheng Wang</text>,
    <a href="https://yongcaiwang.github.io/" target="_blank">Yongcai Wang</a>,
    <text>Maiyue Chen</text>,
    <text>Kaihui Wang</text>,
    <text>Bo Zhang</text>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>,
    <text>Deying Li</text>,
    <a href="https://zhaoxinf.github.io/" target="_blank">Zhaoxin Fan</a>
</div>


<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/mapdream" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2602.00222" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/MapDream" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction
MapDream is a task-driven generative mapping framework for Vision-Language Navigation that learns compact BEV representations via autoregressive map synthesis and jointly optimizes mapping and decision-making.

![](https://horizonrobotics.github.io/robot_lab/mapdream/stats/x2.png)



## Installation

### 1. Set up MapDream

```
conda create -n mapdream python=3.10
conda activate mapdream
pip install ".[mapdream]"

cd projects/mapdream
pip install -r requirements.txt

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```
 
### 2. Set up the Habitat environment


MapDream relies on **Habitat-Sim 0.1.7** for simulation and dataset generation.  
Please follow the official build-from-source guide:  
https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/BUILD_FROM_SOURCE.md

Then install the habitat-lab 0.1.7 dependency.
``` 
# Install habitat-lab
cd projects/mapdream
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all
```

### 3. Set up VLN-CE Extensions
```
cd projects/mapdream
git clone https://github.com/markinruc/VLN_CE.git
```
The RGB sensor resolution must be set to **448 × 448**.

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
cd projects/mapdream
./run_infer.sh
```
Results will be saved in the specified result-path. Run the following command to obtain the final metrics:
```
python analyze_results.py --path result-path
```


## Citation

```bibtex
@article{lian2026mapdream,
  title={MapDream: Task-Driven Map Learning for Vision-Language Navigation},
  author={Lian, Guoxin and Wang, Shuo and Wang, Yucheng and Wang, Yongcai and Chen, Maiyue and Wang, Kaihui and Zhang, Bo and Su, Zhizhong and Li, Deying and Fan, Zhaoxin},
  journal={arXiv preprint arXiv:2602.00222},
  year={2026}
}
```

## Acknowledgments
Our code is based in part on [VILA](https://github.com/NVlabs/VILA), [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE), and [VLN-CE](https://github.com/jacobkrantz/VLN-CE). Thanks for their great works.

