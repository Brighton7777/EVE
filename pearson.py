import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from PIL import Image
from transformers import set_seed

def compute_js_divs(layer_softmax, uncond_layer_softmax, layer_log_softmax, uncond_layer_log_softmax, candidate_tokens_ids=None):
    """
    计算两个分布之间的JS散度
    
    参数:
    candidate_tokens_ids: 候选token的索引，如果为None则计算所有索引
    layer_softmax: 层的softmax输出
    uncond_layer_softmax: 无条件层的softmax输出
    layer_log_softmax: 层的log softmax输出
    uncond_layer_log_softmax: 无条件层的log softmax输出
    
    返回:
    cross_layer_js_divs: 跨层的JS散度值
    """
    # 如果candidate_tokens_ids为None，则计算所有索引
    if candidate_tokens_ids is None:
        M = 0.5 * (layer_softmax + uncond_layer_softmax)
        kl1 = F.kl_div(layer_log_softmax, M, reduction='none').mean(-1)  # shape: (num_layers, batch_size)
        kl2 = F.kl_div(uncond_layer_log_softmax, M, reduction='none').mean(-1)
    else:
        # 计算平均分布M
        M = 0.5 * (layer_softmax[..., candidate_tokens_ids] + uncond_layer_softmax[..., candidate_tokens_ids])
        # 计算KL散度
        kl1 = F.kl_div(layer_log_softmax[..., candidate_tokens_ids], M, reduction='none').mean(-1)  # shape: (num_layers, batch_size)
        kl2 = F.kl_div(uncond_layer_log_softmax[..., candidate_tokens_ids], M, reduction='none').mean(-1)
    
    # 计算JS散度
    js_divs = 0.5 * (kl1 + kl2)
    cross_layer_js_divs = js_divs.mean(-1)  # shape: (num_layers,)
    
    return cross_layer_js_divs

def pearson_correlation(x, y):
    """
    计算两个tensor变量之间的皮尔逊相关系数
    
    参数:
    x, y: tensor变量，需要具有相同的形状
    
    返回:
    皮尔逊相关系数 (标量tensor)
    """
    # 确保tensor展平为1D向量
    x = x.flatten()
    y = y.flatten()
    
    # 计算均值
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    
    # 计算分子(协方差)
    cov_xy = torch.mean((x - mean_x) * (y - mean_y))
    
    # 计算分母(标准差的乘积)
    std_x = torch.std(x)
    std_y = torch.std(y)
    
    # 计算皮尔逊相关系数
    correlation = cov_xy / (std_x * std_y)
    
    return correlation

def process_single_image(model_loader, tokenizer, template, image_id, caption, threshold_top_k=20, threshold_top_p=0.9):
    """
    处理单个图像并计算所有token的平均皮尔逊相关系数
    
    返回:
    平均pearson相关系数或None（如果出错）
    """
    try:
        image_path = "/data1/zhr/datasets/coco2014/val2014/" + "COCO_val2014_"+str(image_id).zfill(12) + ".jpg"
        
        query = "Describe this image in detail."
        questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image_path)
        label_ids = tokenizer([caption], return_tensors="pt").to('cuda').input_ids[:,1:]

        use_jsd = 4

        with torch.inference_mode():
            output_dict, uncond_output_dict = model_loader.llm_model.generate(
                do_sample=False,  # 确保采样设置一致
                temperature=-1,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                use_jsd=use_jsd,
                threshold_top_p=threshold_top_p, 
                threshold_top_k=threshold_top_k,
                return_dict=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                label_ids = label_ids,
                **kwargs,
            )

        output_ids = output_dict.sequences
        # outputs = model_loader.decode(output_ids)[0]
        
        # 获取生成序列的实际长度（排除padding等）
        actual_seq_length = label_ids.shape[1]

        token_correlations = []

        for token_idx in range(10, actual_seq_length):  # 从1开始，跳过第一个可能的特殊token
            try:
                layer_logits = []
                uncond_layer_logits = []

                hidden_states = output_dict.hidden_states[token_idx]
                uncond_hidden_states = uncond_output_dict.hidden_states[token_idx]

                if model == "qwen-vl":
                    layer_norm = model_loader.llm_model.transformer.ln_f
                else:
                    layer_norm = model_loader.llm_model.model.norm

                for layer_id in range(0, 33):
                    if layer_id != 32:
                        layer_output = layer_norm(hidden_states[layer_id])
                        uncond_layer_output = layer_norm(uncond_hidden_states[layer_id])
                    else:
                        layer_output = hidden_states[layer_id].clone()
                        uncond_layer_output = uncond_hidden_states[layer_id].clone()
                    logits = model_loader.llm_model.lm_head(layer_output)[:,-1,:]
                    uncond_logits = model_loader.llm_model.lm_head(uncond_layer_output)[:,-1,:]
                    layer_logits.append(logits)
                    uncond_layer_logits.append(uncond_logits)

                stacked_layer_logits = torch.stack(layer_logits, dim=0) # shape: (num_layers, batch_size, vocab_size)
                stacked_uncond_layer_logits = torch.stack(uncond_layer_logits, dim=0)

                layer_softmax = F.softmax(stacked_layer_logits, dim=-1) # shape: (num_layers, batch_size, vocab_size)
                uncond_layer_softmax = F.softmax(stacked_uncond_layer_logits, dim=-1)

                layer_log_softmax = F.log_softmax(stacked_layer_logits, dim=-1)
                uncond_layer_log_softmax =F.log_softmax(stacked_uncond_layer_logits, dim=-1)

                probs = layer_softmax.squeeze()
                # uncond_probs = uncond_layer_softmax.squeeze()

                candidate_tokens_probs, top_k_candidate_tokens_ids = torch.topk(probs[-1], dim=-1, k=threshold_top_k)

                candidate_tokens_cumulative_probs = candidate_tokens_probs.cumsum(dim=-1).float()
                candidate_tokens_indices = torch.searchsorted(candidate_tokens_cumulative_probs, threshold_top_p, right=False)
                candidate_tokens_cutoff_idx = torch.min(candidate_tokens_indices + 1, torch.tensor(threshold_top_k))   

                candidate_tokens_ids = top_k_candidate_tokens_ids[:candidate_tokens_cutoff_idx]

                label_token_id = label_ids[:,token_idx]
                # label_word = [tokenizer.decode(label_token_id.cpu())]

                tokens_ids = candidate_tokens_ids.clone().detach()
                if label_token_id not in candidate_tokens_ids:
                    tokens_ids = torch.cat([label_token_id, tokens_ids])

                # 计算JS散度
                cross_layer_js_divs = compute_js_divs(layer_softmax, uncond_layer_softmax, layer_log_softmax, uncond_layer_log_softmax, candidate_tokens_ids)

                # 计算皮尔逊相关系数
                pearson_corr = pearson_correlation(cross_layer_js_divs, probs[:,label_token_id])
                print(f"Pearson correlation for token {token_idx} for image {image_id}: {pearson_corr}")
                token_correlations.append(pearson_corr.item())
                
            except Exception as e:
                print(f"Error processing token {token_idx} for image {image_id}: {e}")
                continue


        avg_token_correlation = np.mean(token_correlations)
        print(f"Average Pearson correlation for image {image_id}: {avg_token_correlation}")
        return avg_token_correlation
        
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        return None

if __name__ == "__main__":
    
    # settings
    seed = 42
    model = 'llava-v1.5'
    # model = 'minigpt4'
    # model = "instructblip"
    # model = "qwen-vl"
    sample = False
    temperature = -1
    top_p = None
    num_beams = 1
    max_new_tokens = 512

    #hyparams
    threshold_top_p = 0.9
    threshold_top_k = 20
    label_path = '/data1/zhr/datasets/coco2014/annotations/captions_val2014.json'
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    img_cap = label_data['annotations']

    set_seed(seed)
    disable_torch_init()

    model_loader = ModelLoader(model)
    tokenizer = model_loader.tokenizer
    template = INSTRUCTION_TEMPLATE[model]

    if model == "llava-v1.5":
        model_loader.vlm_model.config.image_aspect_ratio = None

    # 遍历img_cap中的图像
    image_correlations = []
    
    # 限制处理数量以节省时间，您可以根据需要调整这个值
    num_images_to_process = 10  # 设置为len(img_cap)处理所有图像
    
    for i in tqdm(range(min(num_images_to_process, len(img_cap))), desc="Processing images"):
        annotation = img_cap[i]
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        avg_correlation = process_single_image(
            model_loader, tokenizer, template, image_id, caption, 
            threshold_top_k=20, threshold_top_p=0.9
        )
        
        if avg_correlation is not None:
            image_correlations.append(avg_correlation)
            
        # 每处理10个图像打印一次中间结果
        if (i + 1) % 10 == 0:
            if len(image_correlations) > 0:
                overall_avg = np.mean(image_correlations)
                print(f"Processed {i+1} images, overall average correlation: {overall_avg:.4f}")

    # 计算并打印最终结果
    if len(image_correlations) > 0:
        overall_avg_correlation = np.mean(image_correlations)
        std_correlation = np.std(image_correlations)
        print(f"\nFinal Results:")
        print(f"Processed {len(image_correlations)} images")
        print(f"Overall Average Pearson Correlation: {overall_avg_correlation:.4f}")
        print(f"Standard Deviation: {std_correlation:.4f}")
    else:
        print("No images were successfully processed.")