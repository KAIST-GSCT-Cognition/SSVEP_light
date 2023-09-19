import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SSVEP_LIGHT(Dataset):
    def __init__(self, dataset_path, feature_type, platform, pid, split):
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.pid = pid
        self.split = split
        self.platform = platform
        self.get_meta()
        self.get_split()

    def get_meta(self):
        if self.pid:
            feature_meta = json.load(open(os.path.join(self.dataset_path, "feature_meta.json")))
            filter_meta = {k:v for k,v in feature_meta.items() if v["platform"] == self.platform}
            self.metadata = {k:v for k,v in filter_meta.items() if v["pid"] == self.pid}
        else: 
            feature_meta = json.load(open(os.path.join(self.dataset_path, "feature_meta.json")))
            filter_meta = {k:v for k,v in feature_meta.items() if v["platform"] == self.platform}
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
            # features.append(data['gamma_psd'])
            return np.hstack(features)
        else: 
            return data['waveform']


    def __getitem__(self, index):
        item = self.fl[index]
        x_data = self.load_data(item)
        y_data = item['label']
        return x_data, y_data

    def __len__(self):
        return len(self.metadata)