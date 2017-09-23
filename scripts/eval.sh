#!/usr/bin/env bash

model_name='tandemnet'
save_dir='checkpoints/'${model_name}
split=test
checkpoint_name=snapshot_epoch7.00Best.t7

CUDA_VISIBLE_DEVICES=$device_id th train.lua -save_dir $save_dir -batch_size 100 -split $split -only_eval 1 -visatt 'true'  -test_batch_size 1 -load_model_from  'checkpoints/'${model_name}'/snapshot/'${checkpoint_name} -remove_text_feats 1 #-verbose 1 -eval_mode 1
    
