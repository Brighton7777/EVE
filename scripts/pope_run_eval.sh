#!/bin/sh

# greedy
# CUDA_VISIBLE_DEVICES=4 python pope_eval_new.py --model llava-v1.5 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=4 python pope_eval_new.py --model minigpt4  --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=4 python pope_eval_new.py --model instructblip --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=4 python pope_eval_new.py --model qwen-vl --use_jsd 1 --alpha 3

# beam
CUDA_VISIBLE_DEVICES=0 python pope_eval_new.py --model llava-v1.5 --num_beams 5 --use_jsd 1 --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python pope_eval_new.py --model minigpt4 --num_beams 5 --use_jsd 1 --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python pope_eval_new.py --model instructblip --num_beams 5 --use_jsd 1 --alpha 0.5
CUDA_VISIBLE_DEVICES=0 python pope_eval_new.py --model qwen-vl --num_beams 5 --use_jsd 1 --alpha 0.5

# sample
# CUDA_VISIBLE_DEVICES=5 python pope_eval_new.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=5 python pope_eval_new.py --model minigpt4 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=5 python pope_eval_new.py --model instructblip --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=5 python pope_eval_new.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3