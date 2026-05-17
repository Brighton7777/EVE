#!/bin/sh


# python get_objects_data.py --cap_file ./results/chair_opera/llava-v1.5/chair_eval_opera_tokens_512_seed_42.jsonl --save_path "./objects_data/llava_data_opera_full.jsonl"
# python get_objects_data.py --cap_file ./objects_data/llava-v1.5/objects_random_tokens_512_seed_42.jsonl --save_path ./objects_data/llava_data_random_full.jsonl

# CUDA_VISIBLE_DEVICES=1 python objects_eval.py --objects_data_path ./objects_data/llava_data_opera.jsonl --threshold_act 0.0 --beta 0.1

# CUDA_VISIBLE_DEVICES=3 python objects_eval.py --objects_data_path ./objects_data/llava_data_opera_full.jsonl --threshold_act 0.0 --part full_val
# CUDA_VISIBLE_DEVICES=1 python objects_eval.py --objects_data_path ./objects_data/llava_data_random_full.jsonl --threshold_act 0.0 --part full
# CUDA_VISIBLE_DEVICES=3 python objects_random_data.py --model llava-v1.5

# CUDA_VISIBLE_DEVICES=3 python objects_eval.py --objects_data_path ./objects_data/llava_data_opera_full.jsonl --threshold_act 0.005 --part full_jsd_var
# CUDA_VISIBLE_DEVICES=3 python objects_eval.py --objects_data_path ./objects_data/llava_data_opera_full.jsonl --threshold_act 1 --part full_var_logits
CUDA_VISIBLE_DEVICES=4 python objects_eval.py --objects_data_path ./objects_data/llava_data_opera_full.jsonl --threshold_act 0 --part jsd_var_without_max
# CUDA_VISIBLE_DEVICES=4 python objects_eval.py --objects_data_path ./objects_data/llava_data_random_full.jsonl --threshold_act 0 --part jsd_var_probs