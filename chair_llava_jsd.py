import argparse
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import (
    IMAGE_TOKEN_INDEX, # IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN, # DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IM_START_TOKEN, # DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN, # DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER, # IMAGE_PLACEHOLDER = "<image-placeholder>"
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import base64
import requests

from io import BytesIO
import re
import math
import torch.distributed as dist
from utils import dist_util
from utils.logger import create_logger
from glob import glob
from transformers import set_seed


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    # 预处理
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    # 图片列表
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def setup_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



def eval_model(args):
    
    # set up gpu and logging
    
    device = "cuda"

    base_dir = "./results/chair/eval/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"chair_eval_layers_{args.start_layer}-{args.end_layer}_tokens_{args.max_new_tokens}_eos",
        "_sample" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        f"_alpha_{args.alpha}",
        f"_top_p_{args.threshold_top_p}",
        f"_top_k_{args.threshold_top_k}"
    ]

    file_name = "".join(file_parts)
    # Data
    with open("./opera_log/llava-1.5/ours.jsonl", "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device=device)
    for _, data_line in tqdm(enumerate(data_lines),total=500):
        line = json.loads(data_line)
        idx = line["image_id"]
        image_file = args.data_path + "COCO_val2014_" + str(idx).zfill(12) + ".jpg"
        qs = "Please describe this image in detail."

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]            
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().to(device),
                    do_sample=args.sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    # return_dict_in_generate=True,
                    output_hidden_states=True,
                    stopping_criteria=[stopping_criteria],
                    use_jsd = True,
                    alpha = args.alpha,
                    threshold_top_p=args.threshold_top_p, 
                    threshold_top_k=args.threshold_top_k,
                    early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                    return_dict=True
                    )
            
        output_ids = output_sequences
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
        outputs = outputs.strip()
            
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump({"image_id": idx, "caption": outputs}, f)
            f.write("\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-v1.5")
    parser.add_argument("--model-path", type=str, default="/data1/zhr/checkpoints/llava-v1.5-7b")
    parser.add_argument("--data-path", type=str, default="/data1/zhr/datasets/coco2014/val2014/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=-1) # 可以考虑调大温度试试
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=927)

    args = parser.parse_args()
    setup_seeds(args.seed)
    eval_model(args)
