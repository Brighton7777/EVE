# EVE

## Setup

EVE方法代码在`transformers/generation/utils.py`.

```
conda create -n eve python==3.9
conda activate eve
pip install -r requirements.txt
```

## 测试基准
### COCO数据集
数据集路径：`/data1/zhr/datasets/coco2014/val2014/`
### CHAIR
- 生成答案并保存为 jsonl 文件，以 LLaVA-1.5 为例:
```bash
# 贪心搜索
python chair_eval_eve.py --model llava-v1.5 --use_jsd 1 --alpha 0.6
# 集束搜索
python chair_eval_eve.py --model llava-v1.5 --num_beams 5 --use_jsd 1 --alpha 0.6
# 核采样
python chair_eval_eve.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 0.6
```

- 用生成的jsonl文件计算 CHAIR 分数:
```bash
python chair.py --cap_file /path/to/jsonl
```

### POPE
- 生成答案并保存为 jsonl 文件，以 LLaVA-1.5 为例:
```bash
# 贪心搜索
python pope_eval_eve.py --model llava-v1.5 --use_jsd 1 --alpha 0.6
# 集束搜索
python pope_eval_eve.py --model llava-v1.5 --num_beams 5 --use_jsd 1 --alpha 0.6
# 核采样
python pope_eval_eve.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 0.6
```

- 用生成的jsonl文件计算 POPE 分数:
```bash
python pope_ans_all.py --gen_files /path/to/jsonl
```
### MME
- 生成答案并保存各指标 txt 文件，以 LLaVA-1.5 为例:
```bash
# 核采样
python mme_eval_all.py --model llava-v1.5 --sample --top_p 0.9 --temperature 1 --use_jsd 1 --alpha 3
```

- 用生成的包含txt文件的文件夹计算 MME 分数:
```bash
python mme_calculation.py --results_dir /path/to/dir
```

## 可视化
可视化 demo 在 [JSD_examples.ipynb](./JSD_examples.ipynb).