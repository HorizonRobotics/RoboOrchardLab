# Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation

<div align="center" class="authors">
    <a href="https://scholar.google.com/citations?user=IYLvsCQAAAAJ&hl" target="_blank">Shuo Wang</a>,
    <a href="https://yongcaiwang.github.io/" target="_blank">Yongcai Wang</a>,
    <a>Wanting Li</a>,
    <a href="https://scholar.google.com/citations?user=TkwComsAAAAJ&hl=en" target="_blank">Xudong Cai</a>, <br>
    <text>Yucheng Wang</text>,
    <text>Maiyue Chen</text>,
    <text>Kaihui Wang</text>,
    <a href="https://scholar.google.com/citations?user=HQfc8TEAAAAJ&hl=en" target="_blank">Zhizhong Su</a>,
    <text>Deying Li</text>,
    <a href="https://zhaoxinf.github.io/" target="_blank">Zhaoxin Fan</a>
</div>


<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/aux-think" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/Homepage-green" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2505.11886" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-Arxiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HorizonRobotics/Aux-Think" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/Model-HuggingFace-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/datasets/HorizonRobotics/Aux-Think" target="_blank" style="margin: 2px;">
    <img alt="Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction
Aux-Think internalizes Chain-of-Thought (CoT) only during training, enabling efficient Vision-Language Navigation without explicit reasoning at inference, and achieving strong performance with minimal data.

![](https://horizonrobotics.github.io/robot_lab/aux-think/stats/pipeline.png)



## Installation

### 1. Set up Aux-Think

```
conda create -n aux_think python=3.10
conda activate aux_think
pip install ".[aux_think]"

cd projects/aux_think
pip install -r requirements.txt

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VLM
git clone https://github.com/markinruc/VILA.git
cd VILA
pip install .
```
 
### 2. Set up the Habitat environment
Since Habitat-sim 0.1.7 supports up to python3.8, we need an additional conda environment to support the simulator.

```
conda create -n habitat python=3.8
conda activate habitat
```

We recommend to download the habitat-sim package from the [conda website](https://anaconda.org/aihabitat/habitat-sim/0.1.7/download/linux-64/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2) and then install it offline. Or you can install directly by:
```
# Install habitat-sim
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
```
Then install the habitat-lab dependency.
``` 
# Install habitat-lab
git clone --branch v0.1.7 https://github.com/facebookresearch/habitat-lab.git

cd habitat-lab
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all

pip install msgpack_numpy jsonlines lmdb webdataset==0.1.103 dtw fastdtw termcolor imageio
```

### 3. Set up VLN-CE Extensions
```
cd projects/aux_think
git clone https://github.com/markinruc/VLN_CE.git
```

## Inference Data Preparation

Please download the Matterport3D scene data and R2R/RxR datasets following [VLN-CE](https://github.com/jacobkrantz/VLN-CE). You can refer to the following file structure or modify the config in ./VLN-CE

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


```
cd projects/aux_think
./run_infer.sh RESULTS_PATH 0 1 2 3 4 5 6 7
```
Results will be saved in the specified SAVE_PATH. Run the following command to obtain the final metrics:
```
python analyze_results.py --path RESULTS_PATH
```

To stop the inference, run:
```
./kill_infer.sh
```


## Citation

```bibtex
@article{wang2025think,
  title={Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation},
  author={Wang, Shuo and Wang, Yongcai and Li, Wanting and Cai, Xudong and Wang, Yucheng and Chen, Maiyue and Wang, Kaihui and Su, Zhizhong and Li, Deying and Fan, Zhaoxin},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgments
Our code is based in part on [VILA](https://github.com/NVlabs/VILA), [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE), and [VLN-CE](https://github.com/jacobkrantz/VLN-CE). Thanks for their great works.

