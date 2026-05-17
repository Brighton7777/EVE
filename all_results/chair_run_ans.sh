#!/bin/sh

echo "=== InstructBLIP ==="
echo Greedy
python chair.py --cap_file ./all_results/chair/instructblip/chair_eval_tokens_512_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Beam
python chair.py --cap_file ./all_results/chair/instructblip/chair_eval_tokens_512_beams_5_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Sample
python chair.py --cap_file ./all_results/chair/instructblip/chair_eval_tokens_512_sample_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo ""
echo "=== MiniGPT4 ==="
echo Greedy
python chair.py --cap_file ./all_results/chair/minigpt4/chair_eval_tokens_512_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Beam
python chair.py --cap_file ./all_results/chair/minigpt4/chair_eval_tokens_512_beams_5_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Sample
python chair.py --cap_file ./all_results/chair/minigpt4/chair_eval_tokens_512_sample_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo ""
echo "=== LLaVA-v1.5 ==="
echo Greedy
python chair.py --cap_file ./all_results/chair/llava-v1.5/chair_eval_tokens_512_eve_layers_20-29_alpha_20.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Beam
python chair.py --cap_file ./all_results/chair/llava-v1.5/chair_eval_tokens_512_beams_5_eve_layers_20-29_alpha_12.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Sample
python chair.py --cap_file ./all_results/chair/llava-v1.5/chair_eval_tokens_512_sample_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo ""
echo "=== Qwen-VL ==="
echo Greedy
python chair.py --cap_file ./all_results/chair/qwen-vl/chair_eval_tokens_512_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Beam
python chair.py --cap_file ./all_results/chair/qwen-vl/chair_eval_tokens_512_beams_5_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl

echo Sample
python chair.py --cap_file ./all_results/chair/qwen-vl/chair_eval_tokens_512_sample_eve_layers_20-29_alpha_3.0_top_p_0.9_top_k_20_seed_42.jsonl