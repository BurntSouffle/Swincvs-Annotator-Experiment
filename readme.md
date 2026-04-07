# SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy

**Authors**:  
Franciszek Nowak, Evangelos B. Mazomenos, Brian Davidson, Matthew J. Clarkson  

--- 

## Overview

Welcome. This repository provides code necessary for reproduction of the SwinCVS publication. The work proposes a SwinV2+LSTM based architecture called SwinCVS, to classify three Critical View of Safety (CVS) criteria from an open access Endoscapes2023 dataset.  

## Implemented models

- **SwinV2 Backbone**: Pure SwinV2 backbone. Can be run on random weights or initialised using provided ImageNet weights.
- **SwinCSV (E2E, with multiclassifier)**: SwinCVS with end-to-end training and multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (E2E, without multiclassifier)**: SwinCVS with end-to-end training, but without multiclassifier. Backbone weights initialised on ImageNet.
- **SwinCSV (Frozen, without multiclassifier)**: SwinCVS where image encoding backbone is frozen. Suggested backbone weights pretrained on Endoscapes.

## Installation

- Clone this repository
- Confirm you have cuda enabled. In console type nvidia-smi. Our driver API details are:
NVIDIA-SMI 550.120 | Driver Version: 550.120 | CUDA Version: 12.4
- Install runtime API cuda 12.1 - Remember to add to path!
- Install dependencies:<br>
conda create --name swincvs python=3.9.19<br>
conda activate swincvs<br>
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia<br>
pip install -r requirements.txt<br>

## Usage

Script is run by executing SwinCVS.py from the root of the repository. Specific model training parameters are set within config/SwinCVS_config.yaml. Settings that specify model selection are:
- MODEL.LSTM: False - just SwinV2 backbone training, True - SwinCVS = SwinV2 with LSTM
- MODEL.E2E: False - backbone weights frozen, True - End-to-end training
- MODEL.MULTICLASSIFIER: False - does not add an additional classifier after backbone, True - adds a classifier after backbone, before LSTM 
- MODEL.INFERENCE: False - allows for training, True - skips all training, performs only testing on provided weights
- BACKBONE.PRETRAINED: 'str' - which backbone weights to load, imagenet or endoscapes<br>

The script automatically downloads the dataset and the weights. If you already have dataset downloaded please specify directory that contains 'endoscapes' folder with all the dataset data.<br>

After changing the config, execute the script by running the SwinCVS.py and specifying which config file to use (default below):<br>
python3 SwinCVS.py --config_path config/SwinCVS_config.yaml

## Citation

If you use this work in your research, please cite our paper:
Nowak, F., Mazomenos, E., Davidson, B., Clarkson, M., SwinCVS: A Unified Approach to Classifying Critical View of Safety Structures in Laparoscopic Cholecystectomy. Int J CARS (2025). https://doi.org/10.1007/s11548-025-03354-9
