#!/bin/sh

# greedy
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --use_jsd --model llava-v1.5 --start_layer 15 --end_layer 32

CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model llava-v1.5
CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model minigpt4
CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model instructblip
CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model qwen-vl

# beam
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model llava-v1.5 --num_beams 5 --pope-type popular
# CUDA_VISIBLE_DEVICES=3 python pope_eval_all.py --model minigpt4 --num_beams 5
# CUDA_VISIBLE_DEVICES=3 python pope_eval_all.py --model instructblip --num_beams 5
# CUDA_VISIBLE_DEVICES=3 python pope_eval_all.py --model qwen-vl --num_beams 5

# sample
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model minigpt4 --sample --top_p 0.9 --temperature 1
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model instructblip --sample --top_p 0.9 --temperature 1
# CUDA_VISIBLE_DEVICES=1 python pope_eval_all.py --model qwen-vl --sample --top_p 0.9 --temperature 1

CUDA_VISIBLE_DEVICES=4 python pope_eval_new.py --model minigpt4 --use_jsd 2 --alpha 0.6
