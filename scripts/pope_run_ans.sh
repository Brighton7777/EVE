#!/bin/sh


# echo Greedy
# echo "##############################"
# python pope_ans_all.py --gen_files ./results/pope_all_new/instructblip/pope_eval_type_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/minigpt4/pope_eval_type_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/llava-v1.5/pope_eval_type_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/qwen-vl/pope_eval_type_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# echo "##############################"


echo Beam
echo "##############################"
python pope_ans_all.py --gen_files ./results/pope_all_new/instructblip/pope_eval_type_beams_5_jsd_1_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all_new/minigpt4/pope_eval_type_beams_5_jsd_1_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all_new/llava-v1.5/pope_eval_type_beams_5_jsd_1_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./results/pope_all_new/qwen-vl/pope_eval_type_beams_5_jsd_1_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
echo "##############################"

# echo Sample
# echo "##############################"
# python pope_ans_all.py --gen_files ./results/pope_all_new/instructblip/pope_eval_type_sample_top_p_0.9_temp_1.0_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/minigpt4/pope_eval_type_sample_top_p_0.9_temp_1.0_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/llava-v1.5/pope_eval_type_sample_top_p_0.9_temp_1.0_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# python pope_ans_all.py --gen_files ./results/pope_all_new/qwen-vl/pope_eval_type_sample_top_p_0.9_temp_1.0_jsd_2_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
# echo "##############################"

# python pope_ans_all.py --gen_files ./results/zzh/pope_all_new/minigpt4/pope_eval_type_sample_top_p_0.9_temp_1.0_jsd_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl