# TandemNet 
**Deprecated. See new version at https://github.com/zizhaozhang/distill2**

This is the implementation for the Oral paper titled "TandemNet: Distilling Knowledge from Medical Images Using Diagnostic Reports as Optional Semantic References", Zizhao Zhang et al, in MICCAI 2017. Please find more details in the paper.


<img src="https://www.cise.ufl.edu/~zizhao/zzz_files/miccai_2017.png" width="900px"/>



## Simple instructions of training the model

#### Prerequisites
The code is written in Torch7. Install necessary libraries:

  * Torch [https://github.com/torch/distro]
  * Torch-gnuplot [https://github.com/torch/gnuplot]
  * Torch-image [https://github.com/torch/image]
  * Torch-display [https://github.com/szym/display]
  * Lua-cjson [https://www.kyne.com.au/~mark/software/lua-cjson-manual.html]


#### Prepare training data
TandemNet takes images and corresponding text (diagnosic reports) as inputs in order to train. You need to write your own DataLoader.lua. An example with detailed explanations has been provided in utils/DataLoader.lua. 

#### Training
    sh scripts/train.sh

#### Evaluation
    sh scripts/eval.sh

All results will be organized and saved in the folders inside checkpoints/tandemnet.

### Please consider to cite the paper if it is useful
```
@inproceedings{Zhang2017TandemNet,
title={TandemNet: Distilling Knowledge from Medical Images Using Diagnostic Reports as 
Optional Semantic References},
author={Zhang, Zizhao and Chen, Pingjun and Sapkota, Manish and Yang, Lin},
booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
year={2017} 
}
```
