#!/bin/sh

CUDA_VISIBLE_DEVICES=4 python pope_eval_all.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --beta 0.1
CUDA_VISIBLE_DEVICES=4 python pope_eval_all.py --model llava-v1.5 --use_jsd 2 --alpha 1.5 --beta 0.1
CUDA_VISIBLE_DEVICES=4 python chair_eval_opera.py --model llava-v1.5 --use_jsd 2 --alpha 1.5 --beta 0.1
CUDA_VISIBLE_DEVICES=4 python chair_eval_opera.py --model llava-v1.5 --use_jsd 2 --alpha 0.6 --beta 0.1
CUDA_VISIBLE_DEVICES=4 python chair_eval_opera.py --model llava-v1.5 --use_jsd 2 --alpha 1.5 --beta 0.0
CUDA_VISIBLE_DEVICES=4 python chair_eval_opera.py --model llava-v1.5 --use_jsd 2 --alpha 2.0 --beta 0.1