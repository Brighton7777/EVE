import argparse
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
import shortuuid
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llava.utils import disable_torch_init
from constants import INSTRUCTION_TEMPLATE
from model_loader import ModelLoader

from PIL import Image
import math
import re
from transformers import set_seed


def eval_model(args):
    
    # Model
    disable_torch_init()
    model_loader = ModelLoader(args.model)
    base_dir = "./results/chair_eve/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"chair_eval_eve",
        f"_tokens_{args.max_new_tokens}",
        "_sample" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        f"_eve" if args.use_jsd else "",
        "_deco" if args.use_deco else "",
        f"_layers_{args.start_layer}-{args.end_layer}" if args.use_jsd else "",
        f"_alpha_{args.alpha}" if args.use_jsd else "",
        f"_top_p_{args.threshold_top_p}" if args.use_jsd else "",
        f"_top_k_{args.threshold_top_k}" if args.use_jsd else "",
        f"_seed_{args.seed}",
        f"_{args.part}" if args.part else "",
    ]
    file_name = "".join(file_parts)

    template = INSTRUCTION_TEMPLATE[args.model]

    with open("./opera_log/llava-1.5/ours.jsonl", "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    answers_file = os.path.join(base_dir, file_name + ".jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for data_line in tqdm(data_lines, total=500):
        line = json.loads(data_line)
        idx = line["image_id"]
        image_path = args.data_path + "COCO_val2014_" + str(idx).zfill(12) + ".jpg"
        qs = "Please describe this image in detail."

        if args.model == "llava-v1.5":
            model_loader.vlm_model.config.image_aspect_ratio = None

        if args.model == "qwen-vl":
            qs = "Describe this image in detail."

        questions, kwargs = model_loader.prepare_inputs_for_model(
            template, qs, image_path
        )

        with torch.inference_mode():
            outputs = model_loader.llm_model.generate(
                do_sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                use_deco = args.use_deco,
                use_jsd = args.use_jsd,
                alpha = args.alpha,
                threshold_top_p = args.threshold_top_p, 
                threshold_top_k = args.threshold_top_k,
                early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        output_text = model_loader.decode(outputs)[0]

        ans_file.write(json.dumps({"image_id": idx, "caption": output_text}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data1/zhr/datasets/coco2014/val2014/",
        help="data path",
    )
    parser.add_argument("--temperature", type=float, default=-1) # 可以考虑调大温度试试
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_jsd", type=int, default=0)
    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--part", type=str, default="")
    args = parser.parse_args()
    if args.use_jsd:
        print("use jsd", args.use_jsd)
    if args.use_deco:
        print("use deco")
    assert not (args.use_jsd is True and args.use_deco is True), "use_jsd is True and use_deco is True"
    set_seed(args.seed)
    eval_model(args)

