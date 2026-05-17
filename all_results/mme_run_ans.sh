#!/bin/sh

# sample
python mme_calculation.py --results_dir ./all_results/mme/llava-v1.5/mme_eval_tokens_5_sample_jsd_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42
python mme_calculation.py --results_dir ./all_results/mme/qwen-vl/mme_eval_tokens_5_sample_jsd_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42
python mme_calculation.py --results_dir ./all_results/mme/instructblip/mme_eval_tokens_5_sample_eve_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42
python mme_calculation.py --results_dir ./all_results/mme/minigpt4/mme_eval_tokens_5_sample_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42