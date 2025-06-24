import argparse
import os
from tqdm import tqdm
import time
from constants import INSTRUCTION_TEMPLATE
from model_loader_vcd import ModelLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
from transformers import set_seed
import torch

def eval_model(args):
    model_loader = ModelLoader(args.model)
    base_dir = "./results/mme_vcd/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # dump metric file
    file_parts = [
        f"mme_eval",
        f"_tokens_{args.max_new_tokens}",
        "_sample" if args.sample else "",
        f"_beams_{args.num_beams}" if args.num_beams != 1 else "",
        f"_jsd_{args.use_jsd}" if args.use_jsd else "",
        "_deco" if args.use_deco else "",
        "_vcd" if args.use_vcd else "",
        f"_top_p_{args.top_p}",
        f"_seed_{args.seed}",
    ]
    file_name = "".join(file_parts)

    template = INSTRUCTION_TEMPLATE[args.model]

    root = './mme/eval_tool/Your_Results'
    output = os.path.join(base_dir, file_name)
    os.makedirs(output, exist_ok=True)
    for filename in os.listdir(root):
        with open(os.path.join(root, filename), 'r') as fin, open(os.path.join(output, filename), 'w') as fout:
            lines = fin.read().splitlines()
            filename = filename.replace('.txt', '')
            for line in tqdm(lines):
                img, question, gt = line.strip().split('\t')
                img_path = os.path.join(args.data_path, filename, img)
                assert os.path.exists(img_path), img_path
                qs, kwargs = model_loader.prepare_inputs_for_model(
                    template, question, img_path
                )

                with torch.inference_mode():
                    outputs = model_loader.llm_model.generate(
                        do_sample=True,
                        temperature=1,
                        top_p=args.top_p,
                        num_beams=1,
                        max_new_tokens=5,
                        use_cache=True,
                        use_vcd = args.use_vcd,
                        cd_alpha = 1,
                        cd_beta = 0.1,
                        **kwargs
                    )
                response = model_loader.decode(outputs)[0].replace('\n', ' ')

                print(img, question, gt, response, sep='\t', file=fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MME evaluation on LVLMs.")
    parser.add_argument("--model", type=str, help="model", default="minigpt4")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data1/zhr/datasets/mme/images/",
        help="data path",
    )
    parser.add_argument("--temperature", type=float, default=-1) # 可以考虑调大温度试试
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--use_jsd", type=int, default=0)
    parser.add_argument("--use_deco", action="store_true")
    parser.add_argument("--use_vcd", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.use_jsd:
        print("use_jsd", args.use_jsd)
    if args.use_deco:
        print("use_deco")
    if args.use_vcd:
        print("use_vcd")
    assert not (args.use_jsd and args.use_deco), "use_jsd is True and use_deco is True"
    set_seed(args.seed)
    start_time=time.time()
    eval_model(args)
    print(f'Total Time: {(time.time()-start_time)/60:.2f}min')

