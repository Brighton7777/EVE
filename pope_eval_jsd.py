import argparse
import json
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from constants import INSTRUCTION_TEMPLATE, POPE_CHAT_PATH, SYSTEM_MESSAGE
from eval_data_loader import POPEChatDataSet
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

    # load pope data
    args.pope_path = POPE_CHAT_PATH[args.pope_type]
    pope_dataset = POPEChatDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path, 
        trans=model_loader.image_processor
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    base_dir = "./results/pope/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"pope_eval_{args.pope_type}_layers_{args.start_layer}-{args.end_layer}_tokens_{args.max_new_tokens}_eos",
        "_jsd" if args.use_jsd else "",
        "_sample" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        f"_alpha_{args.alpha}",
        f"_top_p_{args.threshold_top_p}",
        f"_top_k_{args.threshold_top_k}",
        f"_seed_{args.seed}"
    ]

    file_name = "".join(file_parts)
    template = INSTRUCTION_TEMPLATE[args.model]
    if args.model == "llava-1.5" or args.model == "shikra":
        template = SYSTEM_MESSAGE + template

    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        queries = np.array(data["query"])
        label = torch.stack(data["label"])
        chat_id = data["chat_id"].item()
        kwargs = {}

        round = label.size()[0]

        for idx in range(round):
            query = queries[idx, :].tolist()
            lal = label[idx, :].tolist()
            # prepare inputs for model
            questions, kwargs = model_loader.prepare_inputs_for_model(
                template, query, image
            )

            with torch.inference_mode():
                outputs = model_loader.llm_model.generate(
                    do_sample=args.sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=5,
                    # return_dict_in_generate=True,
                    output_hidden_states=True,
                    use_jsd = True,
                    # use_deco = True,
                    alpha = args.alpha,
                    threshold_top_p=args.threshold_top_p, 
                    threshold_top_k=args.threshold_top_k,
                    early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                    return_dict=True,
                    **kwargs,
                )

            output_text = model_loader.decode(outputs)

            for i in range(len(output_text)):
                with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
                    json.dump(
                        {
                            "query": query[i],
                            "label": lal[i],
                            "ans": output_text[i],
                            "question": questions[i],
                            "chat_id": chat_id,
                        },
                        f,
                    )
                    f.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--pope-type", type=str, help="model")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
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
    parser.add_argument("--seed", type=int, default=927)
    args = parser.parse_args()
    eval_model(args)