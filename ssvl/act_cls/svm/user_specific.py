import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ssvl.act_cls.dataset import SSVEP_TRIAL_SPLIT
import json

def get_dataset(args, pid, fold_idx):
    tr_dataset = SSVEP_TRIAL_SPLIT(
        dataset_path=args.dataset_path,
        feature_type=args.feature_type,
        platform=args.platform,
        pid=pid,
        hz=args.target_hz,
        split="train",
        fold_idx=fold_idx)
    te_dataset = SSVEP_TRIAL_SPLIT(
        dataset_path=args.dataset_path,
        feature_type=args.feature_type,
        platform=args.platform,
        pid=pid,
        hz=args.target_hz,
        split="test",
        fold_idx=fold_idx)

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for data in tr_dataset:
        x_data, y_data = data
        tr_x.append(x_data)
        tr_y.append(y_data)
    for data in te_dataset:
        x_data, y_data = data
        te_x.append(x_data)
        te_y.append(y_data)
    return tr_x, tr_y, te_x, te_y

def main(args):
    results = {}
    pids = ['P01','P02','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13','P14','P15','P16','P17','P18','P20']
    for fold_idx in range(5): # 5 fold cross validation
        for pid in pids:
            tr_x, tr_y, te_x, te_y = get_dataset(args, pid, fold_idx)
            classifier = make_pipeline(StandardScaler(), SVC(random_state=42))
            classifier.fit(tr_x, tr_y)
            predictions = classifier.predict(tr_x)
            train_score = accuracy_score(tr_y, predictions)
            predictions = classifier.predict(te_x)
            test_score = accuracy_score(te_y, predictions)
            results[f"{pid}_fold{fold_idx}"] = {
                "test_score": test_score,
                "train_score": train_score,
                "num_train": len(tr_x),
                "num_test": len(te_x),
                "pid": pid,
                "fold": fold_idx
            }
    
    average_results = {}
    fold_pid_wise_train, fold_pid_wise_test = [], []
    for pid in pids:
        pid_wise_train = np.mean([v["train_score"] for k,v in results.items() if v["pid"] == pid])
        pid_wise_test = np.mean([v["test_score"] for k,v in results.items() if v["pid"] == pid])
        average_results[f"{pid}_fold_wise"] = {
            "test_score": pid_wise_train,
            "train_score": pid_wise_test,
        }
        fold_pid_wise_train.append(pid_wise_train)
        fold_pid_wise_test.append(pid_wise_test)
        
    average_results["score"] = {
        "avg_train": np.mean(fold_pid_wise_train),
        "avg_test": np.mean(fold_pid_wise_test),
    }
    
    average_results.update(results)
    os.makedirs("exp/user_specific", exist_ok=True)
    with open(f"exp/user_specific/{args.platform}_{args.feature_type}_{args.target_hz}_results.json", mode="w") as io:
        json.dump(average_results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="../../../dataset", type=str)
    parser.add_argument("--feature_type", default="psd", type=str)
    parser.add_argument("--platform", default="Sc", type=str)
    parser.add_argument("--target_hz", default="12hz", type=str)
    parser.add_argument("--fold_idx", default=0, type=int)
    # runner 
    args = parser.parse_args()
    main(args)