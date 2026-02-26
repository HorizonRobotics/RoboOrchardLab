# FineGrasp: Towards Robust Grasping for Delicate Objects

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=Ke5SamYAAAAJ&hl=en" target="_blank">Yun Du*</a>,
    <a href="https://scholar.google.com/citations?hl=en&user=aB02FDEAAAAJ" target="_blank">Mengao Zhao*</a>,
    <a href="https://wzmsltw.github.io/" target="_blank">Tianwei Lin</a>,
    <a href="" target="_blank">Yiwei Jin</a>,
    <a href="" target="_blank">Chaodong Huang</a>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>
</div>
<div align="center" style="line-height: 3;">
  </a>
    <a href="https://horizonrobotics.github.io/robot_lab/finegrasp/index.html" target="_blank" style="margin: 2px;">
    <img alt="Project" src="https://img.shields.io/badge/Project-Website-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
    <a href="https://github.com/HorizonRobotics/robo_orchard_lab/tree/master/projects/finegrasp_graspnet1b" target="_blank" style="margin: 2px;">
    <img alt="FineGrasp Code" src="https://img.shields.io/badge/FineGrasp_Code-Github-bule" style="display: inline-block; vertical-align: middle;"/>
  </a>
  </a>
    <a href="https://github.com/HorizonRobotics/robo_orchard_grasp_label" target="_blank" style="margin: 2px;">
    <img alt="Grasp Label Code" src="https://img.shields.io/badge/Grasp Label Code-Github-bule" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2507.05978" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
    <a href="https://huggingface.co/HorizonRobotics/FineGrasp" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
</div>



# Introduction

Training and evaluation scripts for FineGrasp on the GraspNet-1B dataset.

# Usage

## Prepare Conda Env

```bash
conda create -n finegrasp python=3.10.12
conda activate finegrasp

## Install requirements

## Install torch, spconv that matches your cuda version, for example:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install spconv-cu118==2.3.8

## Option 1: Install robo_orchard_lab and finegrasp from internet
pip3 install robo_orchard_lab[finegrasp]

## Option 2: Install robo_orchard_lab and finegrasp from local
git clone https://github.com/HorizonRobotics/robo_orchard_lab.git
cd robo_orchard_lab && make version
pip3 install .[finegrasp]
```


## Install MinkowskiEngine and graspNetAPI
```bash
git clone https://github.com/graspnet/graspnetAPI.git && cd graspnetAPI
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip3 install .
pip3 install transforms3d==0.4.2 numpy==1.26.4

git clone https://github.com/NVIDIA/MinkowskiEngine.git && cd MinkowskiEngine
conda install openblas-devel -c anaconda
python3 setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## Install pointnet2 and knn op

Compile and install pointnet2 and knn operators from [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp/tree/main)

```
git clone https://github.com/mahaoxiang822/Scale-Balanced-Grasp

cd Scale-Balanced-Grasp/pointnet2 && python3 setup.py install && cd ../..

cd Scale-Balanced-Grasp/knn && python3 setup.py install

```
## Prepare Dataset

For dataset basic preparation, please follow the instructions provided in [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp).


## check Envrionment
```bash
cd projects/finegrasp_graspnet1b
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data \
    --kwargs '{"train_split":"mini", "test_split":"mini", "test_mode":["test_scale_mini"], "step_log_freq":10, "max_epoch":1, "lr":0.0}'

```
## Single GPU Training

```bash
cd projects/finegrasp_graspnet1b
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data
```

## Multi GPU Training

```bash
cd projects/finegrasp_graspnet1b
accelerate launch \
    --multi-gpu \
    --num-processes 4 \
    --gpu-ids 0,1,2,3 \
    train.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace \
    --data_root ./data
```


## Eval
```bash
cd projects/finegrasp_graspnet1b
python3 eval.py \
    --config configs/config_finegrasp_minkunet.py \
    --workspace_root ./workspace/eval_results
    --data_root ./data
```

## Quick inference using pipeline and pretrain models

```bash
cd projects/finegrasp_graspnet1b
python3 infer.py
```

## Results
Grasp performance in GraspNet-1Billion dataset.

| Method    | Camera    | Seen (AP) | Similar (AP) | Novel (AP) | Average (AP) |
| --------- | --------- | --------- | ------------ | ---------- | ---------- |
| FineGrasp | Realsense | 71.67     | 62.83        | 27.40      | 53.97 |
| FineGrasp + CD |  Realsense | 73.71 | 64.56 | 28.14 | 55.47 |

## :handshake: Acknowledgement

We thank the following repositories for their valuable contributions:

- [GraspNetAPI](https://github.com/graspnet/graspnetAPI)
- [Scale-Balanced-Grasp](https://github.com/mahaoxiang822/Scale-Balanced-Grasp)
- [EconomicGrasp](https://github.com/iSEE-Laboratory/EconomicGrasp)
- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

