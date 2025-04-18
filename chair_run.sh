#!/bin/sh

# python chair.py --cap_file ./results/llava-1.5/ori/greedy.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/ori.jsonl

python chair.py --cap_file ./results/llava-1.5/jsd/jsd-greedy-alpha_0.5-seed_927.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/jsd-greedy-alpha_0.5-seed_927.jsonl

# python chair.py --cap_file ./results/llava-1.5/deco/greedy.jsonl --cache /data2/zhr/checkpoints/chair/cache.pkl --coco_path /data2/zhr/datasets/coco2014/annotations --save_path ./results/chair/deco.jsonl