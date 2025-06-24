#!/bin/sh

python mme_calculation.py --results_dir ./results/mme_new/minigpt4/mme_eval_tokens_5_sample_seed_42
python mme_calculation.py --results_dir ./results/mme_vcd/minigpt4/mme_eval_tokens_5_vcd_top_p_0.9_seed_42
python mme_calculation.py --results_dir ./results/mme_new/minigpt4/mme_eval_tokens_5_sample_deco_seed_42
python mme_calculation.py --results_dir ./results/mme_new/minigpt4/mme_eval_tokens_5_sample_jsd_1_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42
python mme_calculation.py --results_dir ./results/mme_new/minigpt4/mme_eval_tokens_5_sample_jsd_1_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42