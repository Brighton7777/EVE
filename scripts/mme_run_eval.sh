#!/bin/sh

# CUDA_VISIBLE_DEVICES=0 python mme_eval_all.py --model llava-v1.5 --use_jsd 2 --alpha 1.2

# sample

# ours
CUDA_VISIBLE_DEVICES=4 python mme_eval_all.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
# CUDA_VISIBLE_DEVICES=2 python mme_eval_all.py --model minigpt4 --sample --top_p 0.9 --temperature 1
# CUDA_VISIBLE_DEVICES=4 python mme_eval_all.py --model minigpt4 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3 # oom
CUDA_VISIBLE_DEVICES=4 python mme_eval_all.py --model instructblip --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
CUDA_VISIBLE_DEVICES=4 python mme_eval_all.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3

# # deco
# CUDA_VISIBLE_DEVICES=0 python mme_eval_all.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_deco --alpha 0.6
# CUDA_VISIBLE_DEVICES=2 python mme_eval_all.py --model minigpt4 --sample --top_p 0.9 --temperature 1 --use_deco --alpha 0.6
# CUDA_VISIBLE_DEVICES=0 python mme_eval_all.py --model instructblip --sample --top_p 0.9 --temperature 1 --use_deco --alpha 0.6
# CUDA_VISIBLE_DEVICES=0 python mme_eval_all.py --model qwen-vl --sample --top_p 0.9 --temperature 1 --use_deco --alpha 0.6

# CUDA_VISIBLE_DEVICES=4 python mme_eval_vcd.py --model minigpt4 --use_vcd