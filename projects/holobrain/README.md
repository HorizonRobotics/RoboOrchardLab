<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_logo.png?raw=true" alt="HoloBrain Logo" width="400" style="vertical-align: middle; margin-right: 15px;">
  <h1 style="display: inline-block; margin: 10; font-size: 2em">A foundation model for general embodied manipulation</h1>
</div>

<div align="center" class="authors">
Xuewu Lin, Yun Du, Hongyu Xie, Yiwei Jin, Jiawei Li, Shijie Wu, Qingze Wang, Mengao Zhao, Ziang Li, Chaodong Huang, Mengdi Li, Hongzhe Bi, Lichao Huang, Zhizhong Su, Tianwei Lin
</div>

<div align="center" style="line-height: 3;">
  <a href="https://horizonrobotics.github.io/robot_lab/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://img.shields.io/badge/🏠HoloBrain-HomePage-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2602.12062" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📄Paper-arXiv-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/HorizonRobotics/RoboOrchardLab/tree/master/projects/holobrain/" target="_blank" style="margin: 2px;">
    <img alt="Code" src="https://img.shields.io/badge/💻Code-Github-black" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/collections/HorizonRobotics/holobrain" target="_blank" style="margin: 2px;">
    <img alt="Model" src="https://img.shields.io/badge/⚙️HoloBrain Model-HuggingFace-orange" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## :book: Framework
<div align="center">
  <img src="https://github.com/HorizonRobotics/robot_lab/blob/master/holobrain/assets/holobrain_framework.png?raw=true" width="90%" alt="HoloBrain" />
  <p style="font-size:1em; color:#555;">By incorporating explicit embodiment modeling (e.g., camera parameters and kinematic descriptions), our model effectively unifies training across heterogeneous robots. Together with a full-stack VLA infrastructure (RoboOrchard) and an effective test-driven data strategy, HoloBrain-0 delivers superior performance on both real world and simulation manipulation benchmarks.</p>
</div>

## :file_folder: Quick Start
###  1. Installation
```bash
cd /path/to/robo_orchard_lab
make version
pip install ".[holobrain_0]"
```

###  2. Prepare Data
#### Preparing [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) Training Data.
Follow the instructions in the RoboTwin code repository to download the required assets and generate data.
Then, use the following command to package the data into LMDB format for training.
```bash
# require data format from the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587
python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer.py \
    --input_path path/to/robotwin_data \
    --output_path "projects/holobrain/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```

### 3. Run Training
```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

# train with single-gpu
python3 train.py --config ${CONFIG}

# train with multi-gpu multi-machine
# example: 2 machines × 8 gpus
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    train.py \
    --workspace ./workspace \
    --config ${CONFIG}
```

### 4. Run Evaluation

#### Close loop evaluation on RoboTwin2.0 Env
Refer to [robotwin_eval](projects/holobrain/holobrain_robotwin_eval/README.md).

### 5. Export Model and Processors
```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py # or configs/config_holobrain_gd_common.py

python3 export.py --config ${CONFIG} --workspace ./model_export_path
```

### 6. Model Inference
The exported model and processor can be used very conveniently. You can insert the code below into any location to perform model inference.
```python
from robo_orchard_lab.models.holobrain.processor import (
  HoloBrainProcessor,
  MultiArmManipulationInput,
  MultiArmManipulationOutput,
)
from robo_orchard_lab.models.mixin import ModelMixin

processor = HoloBrainProcessor.load("./model_export_path", "robotwin2_0_processor.json")
model = ModelMixin.load_model("./model_export_path/model", load_impl="native")

input_data: MultiArmManipulationInput
input_data = processor.pre_process(input_data)
model_outs = model(input_data)
output_data: MultiArmManipulationOutput = processor.post_process(input_data, model_outs)  
```

## :page_facing_up: Citation
```
@misc{lin2026holobrain0technicalreport,
      title={HoloBrain-0 Technical Report}, 
      author={Xuewu Lin and Tianwei Lin and Yun Du and Hongyu Xie and Yiwei Jin and Jiawei Li and Shijie Wu and Qingze Wang and Mengdi Li and Mengao Zhao and Ziang Li and Chaodong Huang and Hongzhe Bi and Lichao Huang and Zhizhong Su},
      year={2026},
      eprint={2602.12062},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.12062}, 
}
```
