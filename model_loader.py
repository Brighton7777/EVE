import os
from collections import namedtuple

import torch
import yaml
from PIL import Image
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    MINIGPT4_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMAGE_TOKEN_LENGTH,
    SHIKRA_IMG_END_TOKEN,
    SHIKRA_IMG_START_TOKEN,
    INSTRUCTION_TEMPLATE_NO_IMG
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llava.model.builder import load_pretrained_model
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from transformers import AutoModelForCausalLM, AutoTokenizer
from mllm.models import load_pretrained
from torchvision import transforms


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model

class BlipModelConfig:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.options = None

def load_blip_model(cfg_path):
    args = BlipModelConfig(cfg_path)
    print('Initialization Model')
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda')
    model.eval()
    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, txt_processors = load_preprocess(processor_cfg)
    # print(vis_processors["eval"].transform)
    print("Done!")

    return model, vis_processors["eval"]

def load_qwen_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, model

def prepare_llava_inputs(template, query, image_path, tokenizer, image_processor, model, prefix):
    image = Image.open(image_path).convert("RGB")
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to("cuda")
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

    qu = template.replace("<question>", query)
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qu)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if prefix:
        prompt += prefix
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    
    img_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1][0].item()
    uncond_input_ids = torch.cat([input_ids[:, :img_idx], input_ids[:, img_idx+1:]], dim=-1)
    uncond_attention_mask = torch.ones(uncond_input_ids.shape[:2], dtype=torch.long, device=uncond_input_ids.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    kwargs = {}
    kwargs["images"] = image_tensor
    kwargs["input_ids"] = input_ids
    kwargs["uncond_input_ids"] = uncond_input_ids
    kwargs["uncond_attention_mask"] = uncond_attention_mask
    kwargs["stopping_criteria"] = [stopping_criteria]

    return prompt, kwargs, input_ids.shape[1]


def prepare_minigpt4_inputs(template, query, image_path, model, image_processor):
    raw_image = Image.open(image_path).convert("RGB")
    image = image_processor(raw_image).unsqueeze(0)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)
    image_tensor = norm(image.to("cuda"))
    qu = [template.replace("<question>", query)]
    batch_size = 1

    uncond_template = INSTRUCTION_TEMPLATE_NO_IMG["minigpt4"]
    uncond_qu = [uncond_template.replace("<question>", query)]
    

    img_embeds, atts_img = model.encode_img(image_tensor)
    inputs_embeds, attention_mask = model.prompt_wrap(
        img_embeds=img_embeds, atts_img=atts_img, prompts=qu
    )

    uncond_inputs_embeds, uncond_attention_mask = model.prompt_wrap(
        img_embeds=None, atts_img=None, prompts=uncond_qu
    )


    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device=inputs_embeds.device)
        * model.llama_tokenizer.bos_token_id
    )
    bos_embeds = model.embed_tokens(bos)
    atts_bos = attention_mask[:, :1]

    # add 1 for bos token
    # img_start_idx = (
    #     model.llama_tokenizer(
    #         qu[0].split("<ImageHere>")[0], return_tensors="pt", add_special_tokens=False
    #     ).input_ids.shape[-1]
    #     + 1
    # )

    inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_bos, attention_mask], dim=1)

    # uncond_inputs_embeds = torch.cat([inputs_embeds[:, :img_start_idx], inputs_embeds[:, img_end_idx:]], dim=1)
    # uncond_attention_mask = torch.cat([attention_mask[:, :img_start_idx], attention_mask[:, img_end_idx:]], dim=1)

    uncond_inputs_embeds = torch.cat([bos_embeds, uncond_inputs_embeds], dim=1)
    uncond_attention_mask = torch.cat([atts_bos, uncond_attention_mask], dim=1)

    kwargs = {}
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["attention_mask"] = attention_mask
    kwargs['uncond_inputs_embeds'] = uncond_inputs_embeds
    kwargs['uncond_attention_mask'] = uncond_attention_mask

    return qu, kwargs, 0

def prepare_instructblip_inputs(template, query, image_path, model, image_processor):
    raw_image = Image.open(image_path).convert("RGB")
    image = image_processor(raw_image).unsqueeze(0)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    norm = transforms.Normalize(mean, std)
    image_tensor = norm(image.to("cuda"))
    qu = [template.replace("<question>", query)]
    
    inputs_embeds, attention_mask, img_start_idx, img_end_idx = model.prompt_wrap(image_tensor, qu)

    uncond_inputs_embeds = torch.cat([inputs_embeds[:, :img_start_idx], inputs_embeds[:, img_end_idx:]], dim=1)
    uncond_attention_mask = torch.cat([attention_mask[:, :img_start_idx], attention_mask[:, img_end_idx:]], dim=1)

    kwargs = {}
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["attention_mask"] = attention_mask
    kwargs['uncond_inputs_embeds'] = uncond_inputs_embeds
    kwargs['uncond_attention_mask'] = uncond_attention_mask

    return qu, kwargs, 0

def qwen_make_context(
    tokenizer,
    query: str,
    history = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def prepare_qwen_inputs(template, query, image_path, tokenizer):
    
    prompt = template.replace("<question>", query).replace("<image_path>", image_path)

    uncond_template = INSTRUCTION_TEMPLATE_NO_IMG["qwen-vl"]
    uncond_prompt = uncond_template.replace("<question>", query)

    system = "You are a helpful assistant. Anwser in English."
    chat_format = "chatml"

    raw_text, context_tokens = qwen_make_context(
        tokenizer,
        prompt,
        system = system,
        chat_format = chat_format
    )

    input_ids = torch.tensor([context_tokens]).to("cuda")

    uncond_raw_text, uncond_context_tokens = qwen_make_context(
        tokenizer,
        uncond_prompt,
        system = system,
        chat_format = chat_format
    )

    uncond_input_ids = torch.tensor([uncond_context_tokens]).to("cuda")
    uncond_attention_mask = torch.ones(uncond_input_ids.shape[:2], dtype=torch.long, device=uncond_input_ids.device)

    kwargs = {}
    kwargs["input_ids"] = input_ids
    kwargs["uncond_input_ids"] = uncond_input_ids
    kwargs['uncond_attention_mask'] = uncond_attention_mask
    kwargs['stop_words_ids'] = [[tokenizer.im_end_id], [tokenizer.im_start_id]]

    return raw_text, kwargs, input_ids.shape[1]


def prepare_qwen_inputs_no_chat(template, query, image_path, tokenizer, model):
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id

    prompt = '<img>{}</img>{} Answer:'.format(image_path, query)
    input_ids = tokenizer([prompt], return_tensors='pt', padding='longest')

    uncond_prompt = '{} Answer:'.format(query)
    uncond_input_ids = tokenizer([uncond_prompt], return_tensors='pt', padding='longest')

    kwargs = {}
    kwargs["input_ids"] = input_ids.input_ids.cuda()
    kwargs["attention_mask"] = input_ids.attention_mask.cuda()
    kwargs["uncond_input_ids"] = uncond_input_ids.input_ids.cuda()
    kwargs['uncond_attention_mask'] = uncond_input_ids.attention_mask.cuda()
    kwargs['pad_token_id'] = tokenizer.eod_id
    kwargs['eos_token_id'] = tokenizer.eod_id

    return prompt, kwargs, input_ids.input_ids.shape[1]


# Example usage:
# prepare_inputs_for_model(args, image, model, tokenizer, kwargs)


class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.input_ids_len = 0
        self.load_model()

    def load_model(self):
        if self.model_name == "llava-v1.5":
            model_path = os.path.expanduser("/data1/zhr/checkpoints/llava-v1.5-7b")
            print('Loading LLava')
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )

        elif self.model_name == "minigpt4":
            cfg_path = "./minigpt4/eval_configs/minigpt4_eval.yaml"
            print('Loading MiniGPT-4')
            model, image_processor = load_blip_model(cfg_path)
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = model.llama_tokenizer, model, image_processor, model.llama_model

        elif self.model_name == "instructblip":
            cfg_path = "./minigpt4/eval_configs/instructblip_eval.yaml"
            print('Loading InstructBlip')
            model, image_processor = load_blip_model(cfg_path)
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = model.llm_tokenizer, model, image_processor, model.llm_model

        elif self.model_name == 'qwen-vl':
            model_path = '/data1/zhr/checkpoints/qwen-vl'
            print('Loading Qwen-VL')
            self.tokenizer, self.llm_model = load_qwen_model(model_path)
            self.vlm_model = self.llm_model

        else:
            raise ValueError(f"Unknown model: {self.model}")

    def prepare_inputs_for_model(self, template, query, image_path, prefix=''):
        if self.model_name == "llava-v1.5":
            questions, kwargs, self.input_ids_len = prepare_llava_inputs(
                template, query, image_path, self.tokenizer, self.image_processor, self.vlm_model, prefix=prefix
            )
        elif self.model_name == "minigpt4":
            questions, kwargs, self.input_ids_len = prepare_minigpt4_inputs(
                template, query, image_path, self.vlm_model, self.image_processor
            )
        elif self.model_name == "instructblip":
            questions, kwargs, self.input_ids_len = prepare_instructblip_inputs(
                template, query, image_path, self.vlm_model, self.image_processor
            )
        elif self.model_name == "qwen-vl":
            questions, kwargs, self.input_ids_len = prepare_qwen_inputs(template, query, image_path, self.tokenizer)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return questions, kwargs


    def decode(self, output_ids):
        # get outputs
        if self.model_name == "llava-v1.5":
            outputs = self.tokenizer.batch_decode(
                    output_ids[:, self.input_ids_len:], skip_special_tokens=True
                )
            output_text = [text.strip() for text in outputs]

        elif self.model_name == "minigpt4":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [
                text.split("###")[0].split("Assistant:")[-1].strip()
                for text in output_text
            ]
        
        elif self.model_name == "instructblip":
            output_ids = output_ids.clone()
            output_ids[output_ids == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            
        elif self.model_name == "qwen-vl":
            output_text = [self.tokenizer.decode(_[self.input_ids_len:].cpu(), skip_special_tokens=True).strip() for _ in output_ids]

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text
