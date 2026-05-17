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

POPE_PATH = {
    "random": "./pope_coco/coco_pope_random.json",
    "popular": "./pope_coco/coco_pope_popular.json",
    "adversarial": "./pope_coco/coco_pope_adversarial.json",
}

def recorder(out):
    word_list = re.split(r'[^\w]+', out.lower())
    if "yes" in word_list:
        return "Yes"
    else:
        return "No"
    
# def image_parser(args):
#     out = args.image_file.split(args.sep)
#     return out


def eval_model(args, model_loader):

    base_dir = "./results/pope_all_mode_1110/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"pope_eval_{args.pope_type}",
        "_sample" if args.sample else "",
        f"_top_p_{args.top_p}" if args.sample else "",
        f"_temp_{args.temperature}" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        f"_jsd_{args.use_jsd}" if args.use_jsd else "",
        f"_mode_{args.mode}" if args.mode else "",
        "_deco" if args.use_deco else "",
        f"_layers_{args.start_layer}-{args.end_layer}" if args.use_jsd else "",
        f"_alpha_{args.alpha}" if args.use_jsd else "",
        f"_top_p_{args.threshold_top_p}" if args.use_jsd else "",
        f"_top_k_{args.threshold_top_k}" if args.use_jsd else "",
        f"_seed_{args.seed}",
    ]
    file_name = "".join(file_parts)

    template = INSTRUCTION_TEMPLATE[args.model]
    args.pope_path = POPE_PATH[args.pope_type]

    questions = [json.loads(q) for q in open(os.path.expanduser(args.pope_path), "r")]
    answers_file = os.path.join(base_dir, file_name + ".jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_path = args.data_path + line["image"]
        qs = line["text"]
        label = line["label"]
        
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
                max_new_tokens=5,
                use_deco = args.use_deco,
                use_jsd = args.use_jsd,
                alpha = args.alpha,
                mode = args.mode,
                threshold_top_p = args.threshold_top_p, 
                threshold_top_k = args.threshold_top_k,
                early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
        output_text = model_loader.decode(outputs)[0]

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": recorder(output_text),
                                   "label": label,
                                   "image": line["image"],
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--pope-type", type=str, help="model", default="all")
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
    parser.add_argument("--mode", type=int, default=0)

    args = parser.parse_args()
    if args.use_jsd:
        print("use_jsd", args.use_jsd)
    if args.mode:
        print("mode", args.mode)
    if args.use_deco:
        print("use_deco")
    assert not (args.use_jsd is True and args.use_deco is True), "use_jsd is True and use_deco is True"
    
    
    set_seed(args.seed)
    # Model
    disable_torch_init()
    model_loader = ModelLoader(args.model)

    if args.pope_type == "all":
        args.pope_type = "random"
        eval_model(args, model_loader)
        args.pope_type = "popular"
        eval_model(args, model_loader)
        args.pope_type = "adversarial"
        eval_model(args, model_loader)
    else:
        eval_model(args, model_loader)

