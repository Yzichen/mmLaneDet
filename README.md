# MMLaneDet

<div align="center">
  <img src="figs/examp.png"/>
</div><br/>


## Introduction

MMLaneDet is an open source lane detection toolbox based on Pytorch. It contains many 
excellent lane detection models and our DALNet (The code will be made available after acceptance of the paper).

## Supported datasets:
- [x] [Tusimple](configs/_base_/datasets/tusimple.py)
- [x] [CULane](configs/_base_/datasets/culane.py)
- [x] [LLAMAS](configs/_base_/datasets/llamas.py)

## Supported detectors:
- [x] [SCNN](configs/scnn)
- [x] [RESA](configs/resa)
- [x] [UFLD](configs/ufld)
- [x] [LaneATT](configs/laneatt)
- [x] [CondLane](configs/condlane)
- [x] [GANet](configs/ganet)
- [x] [BezierLaneNet](configs/BezierLaneNet)
- [x] [CLRNet](configs/clrnet)

## Preparation
### Environments Preparation
Python == 3.8 \
CUDA == 11.1 \
pytorch == 1.9.1 \
mmcv-full == 1.5.1 \
mmdet == 2.25.0 

```Shell
python setup.py develop
```

### Data  Preparation
#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3).  Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```
For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/dataset_converts/generate_seg_tusimple.py --root $TUSIMPLEROOT
```

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

#### LLAMAS
Download [LLAMAS](https://unsupervised-llamas.com/llamas/download).  Then extract them to `$LLAMASROOT`. Create link to `data` directory.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $LLAMASROOT data/llamas
```

## Train & inference
```bash
# train
bash tools/dist_train.sh /path_to_your_config 8
# inference
bash tools/dist_test.sh /path_to_your_config /path_to_your_pth 8 --eval mAP
```

## Results
### Results on Tusimple
|     Model     | Setting |  BatchSize  |  Lr Schd   |  Acc  |  F1   | Config |                                                                                          Download                                                                                         |
|:-------------:|:-------:|:-----------:|:----------:|:-----:|:-----:| :---: |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|    LaneATT    |   r18   | 4(gpus) * 8 | 100 epochs | 95.85 | 96.69 | [config](configs/laneatt/laneatt_r18_tusimple.py) | [model](https://drive.google.com/file/d/1wwiUsUhibLfOEI-os_Nr6yr2LexsgD8Y/view?usp=drive_link)/[log](https://drive.google.com/file/d/1o8vD-F4nYVUzQXmDbzIR786Qo-YiejFF/view?usp=drive_link) |
|    CLRNet     |   r18   | 4(gpus) * 8 | 70 epochs  | 96.81 | 97.63 | [config](configs/clrnet/clrnet_r18_tusimple.py) | [model](https://drive.google.com/file/d/1mWamlpwjfudb80iMyiqaEqJSqDZf0Ljd/view?usp=drive_link)/[log](https://drive.google.com/file/d/1BlFAgBmd3aOjCqX7Dd9po1fY5dctlLfn/view?usp=drive_link) |
| BezierLaneNet |   r18   | 4(gpus) * 8 | 400 epochs | 95.79 | 96.24 | [config](configs/BezierLaneNet/bezier_r18_tusimple.py) |[model](https://drive.google.com/file/d/12M3ujXg2Bf4BNZ2uPb1KnhiJZ80VVkYS/view?usp=drive_link)/[log](https://drive.google.com/file/d/1589WAqvcIeQ0V_Hl_qWE71EWoJb6VqCL/view?usp=drive_link)| 
|    GANet      |   r18   | 4(gpus) * 8 | 70 epochs | 95.99 | 97.23 | [config](configs/ganet/ganet_r18_tusimple.py) | [model](https://drive.google.com/file/d/15Q1cJxJ4xzXoSfKdZd8vKOAOgKBzRamf/view?usp=drive_link)/[log](https://drive.google.com/file/d/1uGIPfs6kjjO5Ti8DQPBn0917LreQLuRZ/view?usp=drive_link) | 

### Results on CuLane
|  Model   | Setting |  BatchSize  |  Lr Schd  | F1@50 | F1@75 |  mF1  |                        Config                        |                                                                                           Download                                                                                           |
|:--------:| :---:   |:-----------:|:---------:|:-----:|:-----:|:-----:|:----------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  CLRNet  | r18| 4(gpus) * 8 | 15 epochs | 79.09 | 62.11 | 54.92 |    [config](configs/clrnet/clrnet_r18_culane.py)     | [model](https://drive.google.com/file/d/1BrK42iR00GxN-fa4LF5S2E44V_PlIbPI/view?usp=drive_link)/[log](https://drive.google.com/file/d/1wZH6379PcJ5XlGNzxTgAMLKzsF6dDzb8/view?usp=drive_link)  |
| CondLane | r18| 2(gpus) * 4 | 16 epochs | 77.99 | 57.48 | 51.42 |  [config](configs/condlane/condlane_r18_culane.py)   | [model](https://drive.google.com/file/d/1E6tq8QmHTU9uEhlMT8c6FXbIwzOnY2uv/view?usp=drive_link)/[log](https://drive.google.com/file/d/1bNZLSY84yV1xIw4ZrMl9zsoNTnQy8B-5/view?usp=drive_link)  |
| BezierLaneNet | r18| 4(gpus) * 8 | 36 epochs | 73.11 | 44.43 | 42.41 | [config](configs/BezierLaneNet/bezier_r18_culane.py) | [model](https://drive.google.com/file/d/1OIEs4eSKjF5uczVY5l-5GaKCS33QrllQ/view?usp=drive_link)/[log](https://drive.google.com/file/d/15RsMu3BjqShwcfG-S9PJsMKkpNYlmeXs/view?usp=drive_link)  |

### Results on LLAMAS(val)
| Model | Setting |  BatchSize  |  Lr Schd  | F1@50 | F1@75 |  mF1  |                    Config                     | Download |
| :---: | :---:   |:-----------:|:---------:|:-----:|:-----:|:-----:|:---------------------------------------------:|:---:|
|CLRNet | r18| 4(gpus) * 8 | 20 epochs | 96.68 | 85.63 | 71.51 | [config](configs/clrnet/clrnet_r18_llamas.py) | [model](https://drive.google.com/file/d/1zeGEChWCkznS48uZHakg2h2AYXoFHkIF/view?usp=drive_link)/[log](https://drive.google.com/file/d/1pXjOnvGbWT7vX_hdbeIDbtjqtLkf3wx3/view?usp=drive_link)|

#### Notes:
I don't have enough time to do all the experiments and optimize the parameters, so some of the results are not fully aligned. 
I would like to have partners on board to help optimize this project.

## DALNet
code will be released.

https://github.com/Yzichen/mmLaneDet/assets/54573533/35a053fb-2fa4-4b62-9250-c06ec28a09f9


## Acknowledgement
Many thanks to the authors of [mmdetection](https://github.com/open-mmlab/mmdetection) and [lanedet](https://github.com/Turoad/lanedet) .

## Citation

If you find mmLaneNet or DALNet is useful in your research or applications, please consider giving us a star 🌟 and 
citing it by the following BibTeX entry.
```bibtex
@article{yu2023dalnet,
  title={DALNet: A Rail Detection Network Based on Dynamic Anchor Line},
  author={Zichen Yu and Quanli Liu and Wei Wang and Liyong Zhang and Xiaoguang Zhao},
  journal={arXiv preprint arXiv:2308.11381},
  year={2023}
}
```
