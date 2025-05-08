#!/bin/sh

# echo llava

# echo Vanilla
# python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_seed_42.jsonl

# echo Greedy
# python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# echo Beam
# python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_beams_5_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# echo Nucleus
# python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_sample_top_p_0.9_temp_1_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# echo "#######################"

# echo minigpt4

# echo Vanilla
# python pope_ans_all.py --gen_files ./results/pope_all/minigpt4/pope_eval_type_seed_42.jsonl

# echo Greedy
# python pope_ans_all.py --gen_files ./results/pope_all/minigpt4/pope_eval_type_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# echo "#######################"

# echo instructblip

# echo Vanilla
# python pope_ans_all.py --gen_files ./results/pope_all/instructblip/pope_eval_type_seed_42.jsonl

# echo Greedy
# python pope_ans_all.py --gen_files ./results/pope_all/instructblip/pope_eval_type_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

# echo "#######################"

# echo qwen-vl

# echo Vanilla
# python pope_ans_all.py --gen_files ./results/pope_all/qwen-vl/pope_eval_type_seed_42.jsonl

# echo Greedy
# python pope_ans_all.py --gen_files ./results/pope_all/qwen-vl/pope_eval_type_jsd_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl


# echo sample
# python pope_ans_all.py --gen_files ./results/pope_all/instructblip/pope_eval_type_sample_top_p_0.9_temp_1.0_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all/minigpt4/pope_eval_type_sample_top_p_0.9_temp_1.0_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_sample_top_p_0.9_temp_1.0_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all/qwen-vl/pope_eval_type_sample_top_p_0.9_temp_1.0_seed_42.jsonl

echo beam
python pope_ans_all.py --gen_files ./results/pope_all/instructblip/pope_eval_type_beams_5_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all/minigpt4/pope_eval_type_beams_5_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_beams_5_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all/qwen-vl/pope_eval_type_beams_5_seed_42.jsonl

python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_noprob_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all/llava-v1.5/pope_eval_type_jsd_layers_20-29_alpha_0.6_beta_0.6_top_p_0.9_top_k_20_prob1_seed_42.jsonl