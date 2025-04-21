#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python pope_eval_jsd.py --use_jsd --model llava-1.5 --pope-type random

CUDA_VISIBLE_DEVICES=3 python pope_eval_jsd.py --use_jsd --model llava-1.5 --pope-type popular

CUDA_VISIBLE_DEVICES=9 python pope_eval_jsd.py --use_jsd --model llava-1.5 --pope-type adversarial

# python pope_ans.py \
# --random_file ./results/pope/llava-1.5/pope_eval_random_layers_20-29_tokens_512_eos_alpha_0.5_top_p_0.9_top_k_20.jsonl \
# --popular_file ./results/pope/llava-1.5/pope_eval_popular_layers_20-29_tokens_512_eos_alpha_0.5_top_p_0.9_top_k_20.jsonl \
# --adversarial_file ./results/pope/llava-1.5/pope_eval_adversarial_layers_20-29_tokens_512_eos_alpha_0.5_top_p_0.9_top_k_20.jsonl \

# python pope_ans.py \
# --random_file ./results/pope/llava-1.5/pope_eval_random_layers_20-29_tokens_512_eos_alpha_0.6_top_p_0.9_top_k_20deco.jsonl \
# --popular_file ./results/pope/llava-1.5/pope_eval_popular_layers_20-29_tokens_512_eos_alpha_0.6_top_p_0.9_top_k_20deco.jsonl \
# --adversarial_file ./results/pope/llava-1.5/pope_eval_adversarial_layers_20-29_tokens_512_eos_alpha_0.6_top_p_0.9_top_k_20deco.jsonl \