import os
import json
import pandas as pd
import numpy as np
import torch
from scipy.fft import fft
from scipy.signal import welch

# hard cording constant
FNAME = "{pid}_SSVEP ({platform}_{area})"
RENAME = "{pid}_{platform}_{area}"
DATASET_PATH = "../dataset"
SAMPLING_RATE = 256 
DELTA = (0.5, 4)
THETA = (4, 8)
ALPHA = (8, 13)
BETA = (13, 30)
GAMMA = (30, 64)
LABEL_DICT = {
    0: "7hz",
    1: "10hz",
    2: "12hz",
    3: "resting",
}

def welch_psd(channel_data, fs, lower_freq, high_freq):
    freqs, psd = welch(channel_data, fs, nperseg=fs)
    subband_idx = np.where((freqs >= lower_freq) & (freqs <= high_freq))[0]
    channel_wise_psd = psd.T[subband_idx].mean(axis=0)
    return channel_wise_psd

def channel_wise_norm(trial_data):
    norm_waveform = []
    for waveform in trial_data:
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
        norm_waveform.append(waveform)
    return norm_waveform
    
def preprocess(f_name, instance, save_name, is_resting):
    # 3 x 30 x 6 x 1025 == label x trial x channel x 
    data = np.load(os.path.join(DATASET_PATH, "npy", f_name + ".npy"))
    data_dictionary = []
    for label, label_data in enumerate(data):
        hz = LABEL_DICT[label]
        for trial, trial_data in enumerate(label_data):
            trial_data = channel_wise_norm(trial_data) # normalization
            if trial < 24:
                split = "train"
            elif 24 <= trial < 27:
                split = "valid"
            elif 27 <= trial:
                split = "test"
            else:
                split = None
            delta_psd = welch_psd(trial_data, SAMPLING_RATE, DELTA[0], DELTA[1])
            theta_psd = welch_psd(trial_data, SAMPLING_RATE, THETA[0], THETA[1])
            alpha_psd = welch_psd(trial_data, SAMPLING_RATE, ALPHA[0], ALPHA[1])
            beta_psd = welch_psd(trial_data, SAMPLING_RATE, BETA[0], BETA[1])
            gamma_psd = welch_psd(trial_data, SAMPLING_RATE, GAMMA[0], GAMMA[1])
            feature = {
                "waveform": trial_data,
                "delta_psd": delta_psd,
                "theta_psd": theta_psd,
                "alpha_psd": alpha_psd,
                "beta_psd": beta_psd,
                "gamma_psd": gamma_psd,
            }
            data_dictionary.append({
                "id": f"{save_name}_{trial}_{label}",
                "pid": instance.pid,
                "platform": instance.platform,
                "area": instance.area,
                "split": split,
                "label": label,
                "is_resting": is_resting,
                "path": os.path.join("feature", save_name, hz, split, f"trial{trial}.pt")
            })
            os.makedirs(os.path.join(DATASET_PATH, "feature", save_name, hz, split), exist_ok=True)
            torch.save(feature, os.path.join(DATASET_PATH, "feature", save_name, hz, split, f"trial{trial}.pt"))
    return data_dictionary

def main():
    df = pd.read_csv(os.path.join(DATASET_PATH, "metadata.csv"), index_col=0)
    df_target = df[df['area'] == "All"]
    save_metadata = {}
    for idx in range(len(df_target)):
        instance = df_target.iloc[idx]
        f_name = FNAME.format_map(instance)
        resting_name = f_name[:-1] + "_Resting)"
        save_name = RENAME.format_map(instance)
        resting_save_name = save_name[:-1] + "_resting"
        data_normal = preprocess(f_name, instance, save_name, is_resting=False)
        data_resting = preprocess(resting_name, instance, resting_save_name, is_resting=True)
        for i in data_normal:
            save_metadata[i["id"]] = i
        for i in data_resting:
            save_metadata[i["id"]] = i
    with open(f"{DATASET_PATH}/feature_meta.json", mode="w") as io:
        json.dump(save_metadata, io, indent=4)

if __name__ == '__main__':
    main()