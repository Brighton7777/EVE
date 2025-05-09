#!/bin/sh

# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --use_jsd --model llava-v1.5
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --use_jsd --model llava-v1.5 --beta 0.3
# CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model llava-v1.5 --use_jsd --alpha 0.6 --beta 0.0
# CUDA_VISIBLE_DEVICES=1 python chair_eval_random.py --use_jsd --model llava-v1.5

# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_deco --model llava-v1.5
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_jsd --model minigpt4
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_jsd --model instructblip
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_jsd --model qwen-vl

# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_jsd --model llava-v1.5 --alpha 0.6 --beta 0.9

# beam

# CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model llava-v1.5 --num_beams 5
# CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model minigpt4 --num_beams 5
# CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model instructblip --num_beams 5

# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --model instructblip --use_deco
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --model minigpt4 --use_deco
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --model qwen-vl --use_deco
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera.py --use_jsd --model llava-v1.5 --start_layer 15 --end_layer 32
CUDA_VISIBLE_DEVICES=1 python chair_eval_opera_small.py --model llava-v1.5 --use_jsd 1
# CUDA_VISIBLE_DEVICES=1 python chair_eval_opera_small.py --model llava-v1.5 --use_jsd 2
CUDA_VISIBLE_DEVICES=4 python chair_eval_opera.py --model llava-v1.5 --use_jsd 1 --alpha 1 --beta 0