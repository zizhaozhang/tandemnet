# TandemNet 
  
This is the implementation for MICCAI 2017 oral paper titled "TandemNet: Distilling Knowledge from Medical Images Using Diagnostic Reports as Optional Semantic References", Zizhao Zhang et al.

## Simple instructions of training the model

### Prepare your training data
TandemNet takes images and corresponding text (diagnosic reports) as inputs. You need to write your own DataLoader.lua. An example with explanations has been provided in utils/DataLoader.lua. 

### Training
    sh scripts/train.sh

### Evaluation
    sh scripts/eval.sh

All results will be saved in the folders inside checkpoints/tandemnet.

## Please consider to cite the paper if it is useful
```
@inproceedings{Zhang2017TandemNet,
title={TandemNet: Distilling Knowledge from Medical Images Using Diagnostic Reports as 
Optional Semantic References},
author={Zhang, Zizhao and Chen, Pingjun and Sapkota, Manish and Yang, Lin},
booktitle={International Conference on Medical Image Computing and 
    Computer-Assisted Intervention (MICCAI)},
year={2017} 
}
```