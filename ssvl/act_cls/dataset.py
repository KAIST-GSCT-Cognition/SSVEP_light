import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_DICT = {
    "7hz":0,
    "10hz": 1,
    "12hz": 2,
    "all": -1
}

class SSVEP_LIGHT(Dataset):
    def __init__(self, dataset_path, feature_type, platform, pid, hz, split):
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.pid = pid
        self.split = split
        self.hz = hz
        self.target_label = LABEL_DICT[self.hz]
        self.platform = platform
        self.get_meta()
        self.get_split()

    def get_meta(self):
        feature_meta = json.load(open(os.path.join(self.dataset_path, "feature_meta.json")))
        if self.pid:
            filter_meta = {k:v for k,v in feature_meta.items() if v["platform"] == self.platform}
            if self.target_label != -1:
                filter_meta = {k:v for k,v in filter_meta.items() if v["label"] == self.target_label}
            self.metadata = {k:v for k,v in filter_meta.items() if v["pid"] == self.pid}
        else: 
            filter_meta = {k:v for k,v in feature_meta.items() if v["platform"] == self.platform}
            if self.target_label != -1:
                filter_meta = {k:v for k,v in filter_meta.items() if v["label"] == self.target_label}
            self.metadata = filter_meta
        
    def get_split(self):
        if self.split == "train":
            self.fl = [v for k,v in self.metadata.items() if v["split"] == "train"]
        elif self.split == "valid":
            self.fl = [v for k,v in self.metadata.items() if v["split"] == "valid"]
        elif self.split == "test":
            self.fl = [v for k,v in self.metadata.items() if v["split"] == "test"]

    def load_data(self, item):
        data = torch.load(os.path.join(self.dataset_path, item['path']))
        features = []
        if "psd" in self.feature_type:            
            features.append(data['delta_psd'])
            features.append(data['theta_psd'])
            features.append(data['alpha_psd'])
            features.append(data['beta_psd'])
            features.append(data['gamma_psd'])
            return np.hstack(features)
        else: 
            waveform = np.array(data['waveform']).astype(np.float32)
            waveform = torch.from_numpy(waveform)
            return waveform

    def __getitem__(self, index):
        item = self.fl[index]
        x_data = self.load_data(item)
        if item['is_resting']:
            y_data = 0
        else:
            y_data = 1
        return x_data, y_data

    def __len__(self):
        return len(self.fl)