#!/bin/sh

# greedy
# CUDA_VISIBLE_DEVICES=0 python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --mode 1
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 1
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model instructblip --use_jsd 2 --alpha 0.6 --mode 1
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 1

# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model instructblip --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 2

# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 5
# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 5
CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 7