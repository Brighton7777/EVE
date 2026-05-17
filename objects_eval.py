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
from chair import CHAIR
import pickle

def is_gt_word(word, gt_words, evaluator):
    if word in gt_words:
        return True
    
    _, node_words, _, _ = evaluator.caption_to_words(word)
    if node_words:
        if node_words[0] in gt_words:
            return True
        
    word_len = len(word)
    for gt_word in gt_words:
        if word_len <= len(gt_word):
            if word == gt_word[:word_len]:
                return True
    return False


def eval_model(args):
    
    # Model
    disable_torch_init()
    model_loader = ModelLoader(args.model)
    evaluator = pickle.load(open("/data1/zhr/checkpoints/chair/cache.pkl", 'rb'))
    base_dir = "./results/objects_1111/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"objects_eval",
        "_random" if "random" in args.objects_data_path else "_opera",
        f"_act_thr_{args.threshold_act}",
        f"_seed_{args.seed}",
        # f"_beta_{args.beta}",
        f"_{args.part}" if args.part else "",
    ]
    file_name = "".join(file_parts)

    template = INSTRUCTION_TEMPLATE[args.model]

    with open(args.objects_data_path, "r", encoding="utf-8") as f:
        data_lines = f.readlines()

    answers_file = os.path.join(base_dir, file_name + ".jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    total_data_num = 0
    jsd_act_data_num = 0
    deco_act_data_num = 0
    random_act_data_num = 0
    jsd_act_gt_word_num = 0
    deco_act_gt_word_num = 0
    random_act_gt_word_num = 0
    ans_file = open(answers_file, "w")
    jsd_val_min = 1000
    jsd_val_gt_min = 1000
    jsd_val_max = -1
    jsd_val_gt_max = -1
    jsd_layer_idx_min = 1000
    jsd_layer_idx_gt_min = 1000
    jsd_layer_idx_max = -1
    jsd_layer_idx_gt_max = -1
    deco_layer_idx_min = 1000
    deco_layer_idx_gt_min = 1000
    deco_layer_idx_max = -1
    deco_layer_idx_gt_max = -1
    jsd_better_idx = []
    deco_better_idx = []
    jsd_probs_better_idx = []
    valid_idx = 1
    for data_line in tqdm(data_lines):
        line = json.loads(data_line)
        img_idx = line["image_id"]
        image_path = args.data_path + "COCO_val2014_" + str(img_idx).zfill(12) + ".jpg"
        qs = "Please describe this image in detail."
        prefix = line["prefix"]
        gt_words = line["mscoco_gt_words"]

        if args.model == "llava-v1.5":
            model_loader.vlm_model.config.image_aspect_ratio = None

        questions, kwargs = model_loader.prepare_inputs_for_model(
            template, qs, image_path, prefix
        )

        with torch.inference_mode():
            output_dict, uncond_output_dict = model_loader.llm_model.generate(
                do_sample=args.sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                use_cache=True,
                max_new_tokens=1,
                use_jsd = args.use_jsd,
                alpha = args.alpha,
                threshold_top_p = args.threshold_top_p, 
                threshold_top_k = args.threshold_top_k,
                early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                return_dict_in_generate=True,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )

        candidate_words = [model_loader.tokenizer.decode(id) for id in output_dict.output_candidate_tokens_ids[0]]
        jsd_probs = output_dict.jsd_probs[0].detach().cpu().numpy()
        deco_probs = output_dict.deco_probs[0].detach().cpu().numpy()
        deco_logits = output_dict.deco_logits[0].detach().cpu().numpy()
        jsd_layer_idx = output_dict.jsd_layer_idx[0]
        deco_layer_idx = output_dict.deco_layer_idx[0]
        jsd_val = output_dict.jsd_val[0].item()
        jsd_logits = output_dict.jsd_logits[0].detach().cpu().numpy()
        jsd_var_probs = output_dict.jsd_var_probs[0].detach().cpu().numpy()
        jsd_var_logits = output_dict.jsd_var_logits[0].detach().cpu().numpy()
        random_layer_idx = output_dict.random_layer_idx[0]
        random_logits = output_dict.random_logits[0].detach().cpu().numpy()
        random_probs = output_dict.random_probs[0].detach().cpu().numpy()

        next_word = candidate_words[0]
        if is_gt_word(next_word, gt_words, evaluator) or len(candidate_words)==1:
            continue
        line["hallucination_word"] = next_word
        line["jsd_act_gt_words"] = []
        line["deco_act_gt_words"] = []
        total_data_num += 1
        jsd_act_gt_num = 0
        deco_act_gt_num = 0
        random_act_gt_num = 0
        
        for i in range(1,len(candidate_words)):
            word = candidate_words[i]
            if is_gt_word(word, gt_words, evaluator):
                # if jsd_probs[i]-jsd_probs[0] > args.threshold_act:
                if jsd_var_probs[i]-jsd_var_probs[0] > args.threshold_act:
                # if jsd_var_logits[i]-jsd_var_logits[0] > args.threshold_act:
                    jsd_act_gt_num += 1
                    line["jsd_act_gt_words"].append(word)
                elif jsd_probs[i]-jsd_probs[0] > args.threshold_act:
                # elif jsd_logits[i]-jsd_logits[0] > args.threshold_act:
                    jsd_probs_better_idx.append((valid_idx, word))

                if deco_probs[i]-deco_probs[0] > args.threshold_act:
                # if deco_logits[i]-deco_logits[0] > args.threshold_act:
                    deco_act_gt_num += 1
                    line["deco_act_gt_words"].append(word)
                if random_probs[i]-random_probs[0] > args.threshold_act:
                # if random_logits[i]-random_logits[0] > args.threshold_act:
                    random_act_gt_num += 1
        
        if random_act_gt_num > 0:
            random_act_data_num += 1

        if jsd_act_gt_num > 0:
            jsd_act_data_num += 1
            jsd_val_gt_min = min(jsd_val_gt_min, jsd_val)
            jsd_val_gt_max = max(jsd_val_gt_max, jsd_val)

            jsd_layer_idx_gt_min = min(jsd_layer_idx_gt_min, jsd_layer_idx)
            jsd_layer_idx_gt_max = max(jsd_layer_idx_gt_max, jsd_layer_idx)

        if deco_act_gt_num > 0:
            deco_act_data_num += 1

            deco_layer_idx_gt_min = min(deco_layer_idx_gt_min, deco_layer_idx)
            deco_layer_idx_gt_max = max(deco_layer_idx_gt_max, deco_layer_idx)

        if jsd_act_gt_num > 0 and deco_act_gt_num==0:
            jsd_better_idx.append(valid_idx)

        if jsd_act_gt_num == 0 and deco_act_gt_num > 0:
            deco_better_idx.append(valid_idx)

        jsd_act_gt_word_num += jsd_act_gt_num
        deco_act_gt_word_num += deco_act_gt_num
        random_act_gt_word_num += random_act_gt_num
        line["jsd_act_gt_num"] = jsd_act_gt_num
        line["deco_act_gt_num"] = deco_act_gt_num
        line["random_act_gt_num"] = random_act_gt_num
        line["jsd_layer_idx"] = jsd_layer_idx
        line["deco_layer_idx"] = deco_layer_idx
        line["random_layer_idx"] = random_layer_idx
        line["jsd_val"] = jsd_val
        line["valid_idx"] = valid_idx
        valid_idx += 1

        jsd_val_min = min(jsd_val_min, jsd_val)
        jsd_val_max = max(jsd_val_max, jsd_val)

        jsd_layer_idx_min = min(jsd_layer_idx, jsd_layer_idx_min)
        jsd_layer_idx_max = max(jsd_layer_idx, jsd_layer_idx_max)
        deco_layer_idx_min = min(deco_layer_idx, deco_layer_idx_min)
        deco_layer_idx_max = max(deco_layer_idx, deco_layer_idx_max)

        ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_dict = {"total_data_num": total_data_num,
                "jsd_act_data_num": jsd_act_data_num,
                "deco_act_data_num": deco_act_data_num,
                "random_act_data_num": random_act_data_num,
                "jsd_act_word_num": jsd_act_gt_word_num,
                "deco_act_word_num": deco_act_gt_word_num,
                "random_act_word_num": random_act_gt_word_num,
                "jsd_act_rate": jsd_act_data_num/total_data_num,
                "deco_act_rate": deco_act_data_num/total_data_num,
                "random_act_rate": random_act_data_num/total_data_num,
                "threshold_act": args.threshold_act,
                "jsd_val_min": jsd_val_min,
                "jsd_val_max": jsd_val_max,
                "jsd_val_gt_min": jsd_val_gt_min,
                "jsd_val_gt_max": jsd_val_gt_max,
                "jsd_better_idx": jsd_better_idx,
                "deco_better_idx": deco_better_idx,
                "jsd_layer_idx_min": jsd_layer_idx_min,
                "jsd_layer_idx_gt_min": jsd_layer_idx_gt_min,
                "jsd_layer_idx_max": jsd_layer_idx_max,
                "jsd_layer_idx_gt_max": jsd_layer_idx_gt_max,
                "deco_layer_idx_min": deco_layer_idx_min,
                "deco_layer_idx_gt_min": deco_layer_idx_gt_min,
                "deco_layer_idx_max": deco_layer_idx_max,
                "deco_layer_idx_gt_max": deco_layer_idx_gt_max,
                "jsd_probs_better_idx": jsd_probs_better_idx,
                }
    print(ans_dict)
    ans_file.write(json.dumps(ans_dict, ensure_ascii=False) + "\n")
    ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Objects evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model", default="llava-v1.5")
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
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--use_jsd", type=int, default=3)
    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--threshold_act", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects_data_path", type=str, default="")
    parser.add_argument("--part", type=str, default="")
    args = parser.parse_args()
    if args.use_jsd:
        print("use jsd", args.use_jsd)
    if args.use_deco:
        print("use deco")
    assert not (args.use_jsd is True and args.use_deco is True), "use_jsd is True and use_deco is True"
    set_seed(args.seed)
    eval_model(args)

