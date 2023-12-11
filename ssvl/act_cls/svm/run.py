import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
import torch
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ssvl.act_cls.dataset import SSVEP_LIGHT
import json

def get_dataset(args, pid):
    tr_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=pid, hz=args.target_hz, split="train")
    va_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type, platform=args.platform, pid=pid, hz=args.target_hz, split="valid")
    te_dataset = SSVEP_LIGHT(dataset_path=args.dataset_path, feature_type=args.feature_type,  platform=args.platform, pid=pid, hz=args.target_hz, split="test")
    train_set = tr_dataset
    eval_sets = torch.utils.data.ConcatDataset([va_dataset, te_dataset])

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for data in train_set:
        x_data, y_data = data
        tr_x.append(x_data)
        tr_y.append(y_data)
    for data in eval_sets:
        x_data, y_data = data
        te_x.append(x_data)
        te_y.append(y_data)
    return tr_x, tr_y, te_x, te_y

def main(args):
    results = {}
    pids = ['P01', 'P02', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10','P11','P12','P13', 'P14', 'P15','P16','P17','P18','P20', None]
    for pid in pids:
        tr_x, tr_y, te_x, te_y = get_dataset(args, pid)
        if pid == None:
            pid = "dependent"
        classifier = make_pipeline(StandardScaler(), SVC(random_state=42))
        classifier.fit(tr_x, tr_y)
        
        predictions = classifier.predict(tr_x)
        train_score = accuracy_score(tr_y, predictions)

        predictions = classifier.predict(te_x)
        test_score = accuracy_score(te_y, predictions)
        results[pid] = {
            "test_score": test_score,
            "train_score": train_score,
            "num_train": len(tr_x),
            "num_test": len(te_x),
        }
    
    train_10fold = np.mean([v["train_score"] for k,v in results.items() if k != "dependent"] )
    test_10fold = np.mean([v["test_score"] for k,v in results.items() if k != "dependent"] )
    results["fold_wise"] = {
        "test_score": test_10fold,
        "train_score": train_10fold,
        "num_train": 0,
        "num_test": 0,
    }
    os.makedirs("exp", exist_ok=True)
    with open(f"exp/{args.platform}_{args.feature_type}_{args.target_hz}_results.json", mode="w") as io:
        json.dump(results, io, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", default="../../../dataset", type=str)
    parser.add_argument("--feature_type", default="psd", type=str)
    parser.add_argument("--platform", default="Sc", type=str)
    parser.add_argument("--target_hz", default="12hz", type=str)
    # runner 
    args = parser.parse_args()
    main(args)