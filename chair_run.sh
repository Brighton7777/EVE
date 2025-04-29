#!/bin/sh

# CUDA_VISIBLE_DEVICES=2 python chair_llava_jsd.py
CUDA_VISIBLE_DEVICES=0 python chair_llava_jsd_pai.py --use_jsd

# python chair.py --cap_file ./results/llava-1.5/ori/greedy.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/ori.jsonl

# python chair.py --cap_file ./results/llava-1.5/jsd/jsd-greedy-alpha_0.5-seed_927.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/jsd-greedy-alpha_0.5-seed_927.jsonl

# python chair.py --cap_file ./results/llava-1.5/deco/greedy.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/deco.jsonl

python chair.py --cap_file ./results/chair/eval/llava-1.5/chair_eval_layers_20-29_tokens_512_bs_1_alpha_0.5_top_p_0.9_top_k_20_seed_927.jsonl --cache /data1/zhr/checkpoints/chair/cache.pkl --coco_path /data1/zhr/datasets/coco2014/annotations --save_path ./results/chair/ans/chair_ans_layers_20-29_tokens_512_bs_1_alpha_0.5_top_p_0.9_top_k_20_seed_927.jsonl