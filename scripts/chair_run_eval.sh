#!/bin/sh

# greedy
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model llava-v1.5 --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model minigpt4 --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model instructblip --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model qwen-vl --use_jsd 1 --alpha 1.2

# # sample
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model minigpt4 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model instructblip --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 1.2
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3 --part 300

# beam

# CUDA_VISIBLE_DEVICES=2 python chair_eval_opera.py --model llava-v1.5 --num_beams 5 --use_jsd 1 --alpha 19
# CUDA_VISIBLE_DEVICES=2 python chair_eval_opera.py --model minigpt4 --num_beams 5 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=2 python chair_eval_opera.py --model instructblip --num_beams 5 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=0 python chair_eval_opera.py --model qwen-vl --num_beams 5 --use_jsd 1 --alpha 3


CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --use_jsd 1 --alpha 5
CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 5
CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --num_beams 5 --use_jsd 1 --alpha 5

CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --use_jsd 1 --alpha 1
CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 1
CUDA_VISIBLE_DEVICES=3 python chair_eval_opera.py --model qwen-vl --num_beams 5 --use_jsd 1 --alpha 1