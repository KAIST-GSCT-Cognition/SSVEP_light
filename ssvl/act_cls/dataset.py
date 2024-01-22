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

class SSVEP_TRIAL_SPLIT(Dataset):
    def __init__(self, dataset_path, feature_type, platform, pid, hz, split, fold_idx=5):
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.pid = pid
        self.split = split
        self.hz = hz
        self.target_label = LABEL_DICT[self.hz]
        self.platform = platform
        self.k = 5
        self.fold_idx = fold_idx
        self.get_meta()
        self.get_split()

    def get_meta(self):
        feature_meta = json.load(open(os.path.join(self.dataset_path, "feature_meta.json")))
        meta_df = pd.DataFrame(feature_meta).T  
        if self.pid:
            filter_meta = meta_df[meta_df["platform"] == self.platform]
            if self.target_label != -1:
                filter_meta = filter_meta[filter_meta["label"] == self.target_label]
            self.metadata = filter_meta[filter_meta["pid"] == self.pid]
        else: 
            filter_meta = meta_df[meta_df["platform"] == self.platform]
            if self.target_label != -1:
                filter_meta = filter_meta[filter_meta["label"] == self.target_label]
            self.metadata = filter_meta
        self.unique_trial = len(set(self.metadata['trial']))
        self.trials = list(range(self.unique_trial))
        self.num_test = self.unique_trial // self.k
        
    def get_split(self):
        test_trial = self.trials[self.fold_idx * self.num_test: (self.fold_idx+1) * self.num_test]
        train_trial = [i for i in self.trials if i not in test_trial]
        if self.split == "train":
            self.fl = self.metadata.set_index("trial").loc[train_trial].reset_index()
        elif self.split == "test":
            self.fl = self.metadata.set_index("trial").loc[test_trial].reset_index()

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
        item = self.fl.iloc[index]
        x_data = self.load_data(item)
        if item['is_resting']:
            y_data = 0
        else:
            y_data = 1
        return x_data, y_data

    def __len__(self):
        return len(self.fl)
    
    

class SSVEP_PID_SPLIT(Dataset):
    def __init__(self, dataset_path, feature_type, platform, pid_list, hz):
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.pid_list = pid_list
        self.hz = hz
        self.target_label = LABEL_DICT[self.hz]
        self.platform = platform
        self.get_meta()
        self.get_split()

    def get_meta(self):
        feature_meta = json.load(open(os.path.join(self.dataset_path, "feature_meta.json")))
        meta_df = pd.DataFrame(feature_meta).T  
        filter_meta = meta_df[meta_df["platform"] == self.platform]
        if self.target_label != -1:
            filter_meta = filter_meta[filter_meta["label"] == self.target_label]
        self.metadata = filter_meta
        
    def get_split(self):
        self.fl = self.metadata.set_index("pid").loc[self.pid_list].reset_index()

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
        item = self.fl.iloc[index]
        x_data = self.load_data(item)
        if item['is_resting']:
            y_data = 0
        else:
            y_data = 1
        return x_data, y_data

    def __len__(self):
        return len(self.fl)