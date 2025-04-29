import argparse
import json
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm


def setup_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True



def eval_model(args):
    setup_seeds(args.seed)
    disable_torch_init()

    model_loader = ModelLoader(args.model)

    coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor)
    coco_loader = torch.utils.data.DataLoader(
        coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    base_dir = "./results/chair/eval/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    file_parts = [
        f"chair_eval_layers_{args.start_layer}-{args.end_layer}_tokens_{args.max_new_tokens}_bs_{args.batch_size}",
        "_sample" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        "_jsd" if args.use_jsd else "",
        f"_alpha_{args.alpha}" if args.use_jsd else "",
        f"_top_p_{args.threshold_top_p}" if args.use_jsd else "",
        f"_top_k_{args.threshold_top_k}" if args.use_jsd else "",
        f"_seed_{args.seed}",
    ]

    file_name = "".join(file_parts)
    file_path = os.path.join(base_dir, file_name + ".jsonl")
    template = INSTRUCTION_TEMPLATE[args.model]
    if args.model == "llava-1.5" or args.model == "shikra":
        template = SYSTEM_MESSAGE + template

    for batch_id, data in tqdm(enumerate(coco_loader), total=500):
        if batch_id == 500:
            break
        img_id = data["img_id"]
        image = data["image"]

        batch_size = img_id.shape[0]
        query = ["Please help me describe the image in detail."] * batch_size
        questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

        with torch.inference_mode():
            outputs = model_loader.llm_model.generate(
                do_sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_jsd=args.use_jsd,
                alpha = args.alpha,
                threshold_top_p=args.threshold_top_p, 
                threshold_top_k=args.threshold_top_k,
                early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        output_text = model_loader.decode(outputs)

        for i in range(len(output_text)):
            with open(file_path, "a") as f:
                json.dump({"image_id": int(img_id[i]), "caption": output_text[i]}, f)
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
    # parser.add_argument("--model", type=str, help="model", default='llava-1.5')
    parser.add_argument("--model", type=str, help="model", default='minigpt4')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # TODO
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data1/zhr/datasets/coco2014/val2014/",
        help="data path",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=-1) # 可以考虑调大温度试试
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_jsd", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    eval_model(args)