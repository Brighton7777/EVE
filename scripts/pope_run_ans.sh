#!/bin/sh


# echo Greedy
# echo "##############################"
# python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_eve_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_eve_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# echo "##############################"


# echo Beam
# echo "##############################"
# python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
# echo "##############################"

# echo Sample
# echo "##############################"
# python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# echo "##############################"

echo ablation
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_1_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_3_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_1_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_3_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl