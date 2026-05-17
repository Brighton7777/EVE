#!/bin/bash

alpha=0.5
end=5.0
step=0.5

while [ $(echo "$alpha <= $end" | bc) -eq 1 ]; do
    printf ">>> Running with alpha = %.1f\n" "$alpha"
    CUDA_VISIBLE_DEVICES=0 python pope_eval_new.py --model llava-v1.5 --use_jsd 1 --alpha "$alpha"
    alpha=$(echo "$alpha + $step" | bc)
done