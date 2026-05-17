#!/bin/sh

# greedy
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --mode 3
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 3
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model instructblip --use_jsd 2 --alpha 0.6 --mode 3
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 3

# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --mode 4
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 4
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model instructblip --use_jsd 2 --alpha 0.6 --mode 4
# CUDA_VISIBLE_DEVICES=1 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 4

# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model instructblip --use_jsd 2 --alpha 0.6 --mode 2
# CUDA_VISIBLE_DEVICES=2 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 2

# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model minigpt4  --use_jsd 2 --alpha 0.6 --mode 6
# CUDA_VISIBLE_DEVICES=3 python pope_eval_mode.py --model qwen-vl --use_jsd 2 --alpha 0.6 --mode 6
CUDA_VISIBLE_DEVICES=4 python pope_eval_mode.py --model minigpt4 --use_jsd 2 --alpha 0.6 --mode 7d