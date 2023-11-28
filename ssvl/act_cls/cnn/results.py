import os
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
"""
target format
{
    "P01": {
        "test_score": 0.8333333333333334,
        "train_score": 0.8333333333333334,
        "num_train": 48,
        "num_test": 12
    },
    ....
"""
results = {}
for platform in ["VR", "Sc"]:
    for hz in ["7hz", "10hz", "12hz", "all"]:
        trs, vas = [], []
        for pid in ["P01", "P02", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "None"]:
            config = OmegaConf.load(f"./exp/{platform}/{pid}_{hz}/hparams.yaml")
            if pid == "None":
                pid = "dependent"
            else:
                trs.append(config.best_tr)
                vas.append(config.best_val)
            results[pid] = {
                    "test_score": config.best_val,
                    "train_score": config.best_tr,
                }
        results["fold10"] = {
                    "test_score": np.mean(vas),
                    "train_score": np.mean(trs)
                }
        with open(f"exp/{platform}_waveform_{hz}_results.json", mode="w") as io:
            json.dump(results, io, indent=4)