#!/bin/sh

echo ""
echo "=========================================="
echo "Starting POPE Evaluation Pipeline"
echo "=========================================="
echo ""

echo ">>> Greedy Decoding"
echo "------------------------------------------"
python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_eve_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_eve_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
echo "------------------------------------------"
echo ""

echo ">>> Beam Search (beams=5)"
echo "------------------------------------------"
python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_beams_5_eve_layers_20-29_alpha_0.5_top_p_0.9_top_k_20_seed_42.jsonl
echo "------------------------------------------"
echo ""

echo ">>> Sampling (temp=1.0, top_p=0.9)"
echo "------------------------------------------"
python pope_ans_all.py --gen_files ./all_results/pope/instructblip/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/minigpt4/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/llava-v1.5/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_5.0_top_p_0.9_top_k_20_seed_42.jsonl
python pope_ans_all.py --gen_files ./all_results/pope/qwen-vl/pope_eval_type_sample_top_p_0.9_temp_1.0_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl
echo "------------------------------------------"
echo ""

echo ">>> Ablation Study"
echo "------------------------------------------"
echo "MiniGPT4:"
echo "w/o max_JSD:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_1_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
echo "w/o max_prob:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
echo "w/o max_prob & max_JSD:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/minigpt4/pope_eval_type_eve_mode_3_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl

echo ""
echo "Qwen-VL:"
echo "w/o max_JSD:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_1_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
echo "w/o max_prob:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_2_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
echo "w/o max_prob & max_JSD:"
python pope_ans_all.py --gen_files ./all_results/pope_ablation/qwen-vl/pope_eval_type_eve_mode_3_layers_20-29_alpha_0.6_top_p_0.9_top_k_20_seed_42.jsonl
echo "------------------------------------------"
echo ""

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="