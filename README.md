# EVE

## 论文
论文链接：https://www.onlinelatex.com/6954114316xrfyrsssrzsk#def462

论文图片在 [figures](./figures/)

## 安装 

EVE方法代码在`transformers/generation/utils.py`.

```
# pip 安装
conda create -n eve python==3.9
conda activate eve
pip install -r requirements.txt
# 本地 clone
conda create -n eve --clone /home/zhr/.conda/envs/eve
```

## 测试基准
### COCO数据集
数据集路径：`/data1/zhr/datasets/coco2014/val2014/`

所有评估脚本：
```bash
# chair
bash ./all_results/chair_run_ans.sh
# pope
bash ./all_results/pope_run_ans.sh
# mme
bash ./all_results/mme_run_ans.sh
```

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

## 消融实验

以LLaVA-1.5为例：
```bash
# w/o jsd_max_val
python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --mode 1 
# w/o max_probs
python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --mode 2
# w/o both
python pope_eval_mode.py --model llava-v1.5 --use_jsd 2 --mode 3
```