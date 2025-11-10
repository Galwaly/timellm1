#!/usr/bin/env bash
# run_once.sh  ——  TimeLLM-ETTh1 单次实验（单 GPU）

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
# 下面保持原来的 python / torchrun 命令

export DS_SKIP_MPI=1
export CUDA_VISIBLE_DEVICES=0        # 仅用 0 号卡
export CUDA_LAUNCH_BLOCKING=1        # 可选：同步报错，调试更直观

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model TimeLLM \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 12 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 16 \
  --learning_rate 0.01 \
  --llm_layers 6 \
  --train_epochs 2 \
  --checkpoints ./checkpoints_12/\
  --patience 100 \
  --model_comment 'TimeLLM-ETTh1'