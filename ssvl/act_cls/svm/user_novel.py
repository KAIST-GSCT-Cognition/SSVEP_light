import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ssvl.act_cls.dataset import SSVEP_PID_SPLIT
import json

def get_dataset(args, train_pid, test_pid):
    tr_dataset = SSVEP_PID_SPLIT(
        dataset_path=args.dataset_path,
        feature_type=args.feature_type,
        platform=args.platform,
        pid_list=train_pid, # pid for data split
        hz=args.target_hz,
        )
    te_dataset = SSVEP_PID_SPLIT(
        dataset_path=args.dataset_path,
        feature_type=args.feature_type,
        platform=args.platform,
        pid_list=[test_pid], # pid for data split
        hz=args.target_hz,
        )

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
    for test_pid in pids:
        train_pid = [i for i in pids if i != test_pid]
        tr_x, tr_y, te_x, te_y = get_dataset(args, train_pid, test_pid)
        classifier = make_pipeline(StandardScaler(), SVC(random_state=42))
        classifier.fit(tr_x, tr_y)
        predictions = classifier.predict(tr_x)
        train_score = accuracy_score(tr_y, predictions)

        predictions = classifier.predict(te_x)
        test_score = accuracy_score(te_y, predictions)
        results[f"{test_pid}_test_fold"] = {
            "test_score": test_score,
            "train_score": train_score,
            "num_train": len(tr_x),
            "num_test": len(te_x),
            "pid": test_pid
        }
    
    avg_train = np.mean([v["train_score"] for k,v in results.items()] )
    avg_test = np.mean([v["test_score"] for k,v in results.items()] )
    results["score"] = {
        "avg_train": avg_train,
        "avg_test": avg_test,
    }
    
    os.makedirs("exp/user_novel", exist_ok=True)
    with open(f"exp/user_novel/{args.platform}_{args.feature_type}_{args.target_hz}_results.json", mode="w") as io:
        json.dump(results, io, indent=4)


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