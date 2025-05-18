#!/bin/sh

# llava
echo llava

#deco
# python chair.py --cap_file ./results/chair_random/llava-v1.5/chair_eval_random_tokens_512_deco_seed_42.jsonl

#ours

python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_10.0_top_p_0.9_top_k_20_seed_42_pad.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_9.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_8.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_7.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_6.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_4.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_2.0_top_p_0.9_top_k_20_seed_42.jsonl

python chair.py --cap_file ./results/chair_opera_test/llava-v1.5/chair_eval_opera_tokens_512_jsd_2_layers_20-29_alpha_2.0_top_p_0.9_top_k_20_seed_42.jsonl


python chair.py --cap_file ./results/chair_opera_test/qwen-vl/chair_eval_opera_tokens_512_sample_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

# # minigpt4
# echo minigpt4

# python chair.py --cap_file ./results/chair_opera/minigpt4/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# python chair.py --cap_file ./results/chair_opera/minigpt4/chair_eval_opera_tokens_512_seed_42.jsonl

# python chair.py --cap_file  ./results/chair_opera/minigpt4/chair_eval_opera_tokens_512_deco_seed_42.jsonl

# # instructblip
# echo instructblip

# python chair.py --cap_file ./results/chair_opera/instructblip/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# python chair.py --cap_file ./results/chair_opera/instructblip/chair_eval_opera_tokens_512_seed_42.jsonl

# python chair.py --cap_file  ./results/chair_opera/instructblip/chair_eval_opera_tokens_512_deco_seed_42.jsonl

# # qwen-vl
# echo qwen-vl

# python chair.py --cap_file ./results/chair_opera/qwen-vl/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# python chair.py --cap_file ./results/chair_opera/qwen-vl/chair_eval_opera_tokens_512_seed_42.jsonl

# python chair.py --cap_file  ./results/chair_opera/qwen-vl/chair_eval_opera_tokens_512_deco_seed_42.jsonl