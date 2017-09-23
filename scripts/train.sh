#!/usr/bin/env bash

name=tandemnet
save_dir='checkpoints/'$name
CUDA_VISIBLE_DEVICES=$device_id th train.lua -save_dir $save_dir -batch_size 64 
