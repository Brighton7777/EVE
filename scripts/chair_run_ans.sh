#!/bin/sh

# llava
echo llava

python chair.py --cap_file ./results/chair_opera/llava-v1.5/chair_eval_opera_tokens_512_seed_42.jsonl
python chair.py --cap_file ./results/chair_opera/llava-v1.5/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_beta_0.9_top_p_0.9_top_k_20_noprob_seed_42.jsonl

# minigpt4
echo minigpt4

python chair.py --cap_file ./results/chair_opera/minigpt4/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

python chair.py --cap_file ./results/chair_opera/minigpt4/chair_eval_opera_tokens_512_seed_42.jsonl

# instructblip
echo instructblip

python chair.py --cap_file ./results/chair_opera/instructblip/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

python chair.py --cap_file ./results/chair_opera/instructblip/chair_eval_opera_tokens_512_seed_42.jsonl

# qwen-vl
echo qwen-vl

python chair.py --cap_file ./results/chair_opera/qwen-vl/chair_eval_opera_tokens_512_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

python chair.py --cap_file ./results/chair_opera/qwen-vl/chair_eval_opera_tokens_512_seed_42.jsonl