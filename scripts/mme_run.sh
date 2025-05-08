#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python mme_eval_all.py --use_jsd --model llava-v1.5

python mme_calculation.py --results_dir ./results/mme/llava-v1.5/mme_eval_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42