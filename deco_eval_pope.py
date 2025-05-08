import os
import json
import argparse
from tqdm import tqdm



def calculate_metrics(gt_file_path, gen_file_path):

    # open ground truth answers
    gt_files = [json.loads(q) for q in open(os.path.expanduser(gt_file_path), "r")]

    # open generated answers
    gen_files = [json.loads(q) for q in open(os.path.expanduser(gen_file_path), "r")]

    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(gt_files)
    yes_answers = 0

    # compare answers
    for index, line in enumerate(gt_files):
        idx = line["question_id"]
        gt_answer = line["label"]
        assert idx == gen_files[index]["question_id"]
        gen_answer = gen_files[index]["text"]
        # convert to lowercase
        gt_answer = gt_answer.lower()
        gen_answer = gen_answer.lower()
        # strip
        gt_answer = gt_answer.strip()
        gen_answer = gen_answer.strip()
        # pos = 'yes', neg = 'no'
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            print(f'Warning: unknown gt_answer: {gt_answer}')
            unknown += 1
    # calculate precision, recall, f1, accuracy, and the proportion of 'yes' answers
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    # report results
    print(f'File: {gen_file_path}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'yes: {yes_proportion}')
    print(f'unknow: {unknown_prop}')
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_gt_files", type=str, default="")
    parser.add_argument("--random_gen_files", type=str, default="")
    parser.add_argument("--popular_gt_files", type=str, default="")
    parser.add_argument("--popular_gen_files", type=str, default="")
    parser.add_argument("--adversarial_gt_files", type=str, default="")
    parser.add_argument("--adversarial_gen_files", type=str, default="")
    args = parser.parse_args()
    f1_random = calculate_metrics(args.random_gt_files, args.random_gen_files)
    f1_popular = calculate_metrics(args.popular_gt_files, args.popular_gen_files)
    f1_adversarial = calculate_metrics(args.adversarial_gt_files, args.adversarial_gen_files)
    print("F1:")
    print(f'Random: {f1_random}')
    print(f'Popular: {f1_popular}')
    print(f'Adversarial: {f1_adversarial}')
    print(f'mean: {(f1_random + f1_popular + f1_adversarial) / 3}')
