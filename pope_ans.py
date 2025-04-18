import argparse
import json
import os
import numpy as np

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            pred_list.append(0)
        else:
            pred_list.append(1)

    return pred_list

def acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t\n")
    print("{}\t{}\t{}\t{}\n".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Accuracy: {}\n".format(acc))
    print("Precision: {}\n".format(precision))
    print("Recall: {}\n".format(recall))
    print("F1 score: {}\n".format(f1))
    print("Yes ratio: {}\n".format(yes_ratio))

    return acc, precision, recall, f1, yes_ratio



def calc_file(file):

    lines = open(file).read().split("\n")
    pred_list = []
    label_list = []
    i = 0
    for line in lines:
        i += 1
        if len(line) == 0:
            break
        line = json.loads(line)
        pred_list = recorder([line["ans"]], pred_list)
        if isinstance(line["label"], int):
            label_list += [line["label"]]
        else:
            label_list = recorder([line["label"]], label_list)
    print(f'file name: {os.path.basename(file)}')
    return acc(pred_list, label_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POPE evaluation on LVLMs.")
    parser.add_argument("--random_files", type=str, help="random file")
    parser.add_argument("--popular_files", type=str, help="popular file")
    parser.add_argument("--adversarial_files", type=str, help="adversarial file")
    args = parser.parse_args()
    f1_list = []
    _, _, _, f1, _ = calc_file(args.random_files)
    f1_list.append(f1)
    _, _, _, f1, _ = calc_file(args.popular_files)
    f1_list.append(f1)
    _, _, _, f1, _ = calc_file(args.adversarial_files)
    f1_list.append(f1)
    print("All F1 score: {}\n".format(np.mean(f1_list)))
