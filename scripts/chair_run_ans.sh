#!/bin/sh

# Greedy

# python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_10.0_top_p_0.9_top_k_20_seed_42_pad.jsonl

# Beam


# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_11.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_12.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_13.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_14.0_top_p_0.9_top_k_20_seed_42.jsonl

# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_16.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_17.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_18.0_top_p_0.9_top_k_20_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_19.0_top_p_0.9_top_k_20_seed_42.jsonl

# python chair.py --cap_file ./results/chair_opera_new/instructblip/chair_eval_opera_tokens_512_beams_5_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/minigpt4/chair_eval_opera_tokens_512_beams_5_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/llava-v1.5/chair_eval_opera_tokens_512_beams_5_seed_42.jsonl
# python chair.py --cap_file ./results/chair_opera_new/qwen-vl/chair_eval_opera_tokens_512_beams_5_seed_42.jsonl

# Sample

python chair.py --cap_file ./results/chair_opera_new/qwen-vl/chair_eval_opera_tokens_512_sample_jsd_1_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_new/qwen-vl/chair_eval_opera_tokens_512_beams_5_jsd_1_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
