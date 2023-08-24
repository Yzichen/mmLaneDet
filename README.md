# MMLaneDet

<div align="center">
  <img src="figs/examp.png"/>
</div><br/>


## Introduction

MMLaneDet is an open source lane detection toolbox based on Pytorch. It contains many 
excellent lane detection models and our DALNet (The code will be made available after acceptance of the paper).

## Preparation
### Environments Preparation
Python == 3.8 \
CUDA == 11.1 \
pytorch == 1.9.1 \
mmcv-full == 1.5.1 \
mmdet == 2.25.0 

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
