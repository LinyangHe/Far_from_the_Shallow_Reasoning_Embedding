#!/usr/bin/env python
"""
Shuffle Analysis for Brain Encoding with Ridge Regression
"""

import os, pickle, logging
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from nilearn.plotting import plot_markers
import librosa, scipy.signal
from tqdm.notebook import tqdm
import gc
import time

# Import the ridge regression function
from ridge_utils.ridge import bootstrap_ridge

# change to your home directory
home_dir = os.path.expanduser("~")

logging.basicConfig(level=logging.INFO, filename=Path(home_dir)/'ridge_analysis.log', format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

dtype = np.float64
preload = True
start_time = time.time()
# ======================================================
# 1. Pre-load Common Data
# ======================================================
# Load transcript (common to all subjects
transcript_path = "./podcast_transcript.csv"
df_transcript = pd.read_csv(transcript_path)
logging.info(f"Transcript loaded with {len(df_transcript)} entries.")

# ======================================================
# 2. Global Parameters for Regression and CV
# ======================================================
# alphas = np.logspace(0, 3, 10)  # candidate ridge penalty values
alphas = np.array([10]) # candidate ridge penalty values

nboots = 5                    # number of bootstrap iterations
chunk_len = 32                # events per contiguous chunk
cv = KFold(n_splits=5, shuffle=False)

# Load word rate feature tables and concatenate them.
sec1_5_feats_path = "./podcast_feats/sec_1.5-selected-podcast/___podcasts-story___.csv"
sec3_feats_path = "./podcast_feats/sec_3-podcast/sec_3/___podcasts-story___.csv"
df_sec1_5 = pd.read_csv(sec1_5_feats_path, sep=",", index_col=0)
df_sec3 = pd.read_csv(sec3_feats_path, sep=",", index_col=0)

wc_sec1_5 = df_sec1_5.index.to_series().astype(str).apply(lambda x: len(x.split())).values
wc_sec3   = df_sec3.index.to_series().astype(str).apply(lambda x: len(x.split())).values
word_rate_features = np.column_stack([wc_sec1_5, wc_sec3])
logging.info(f"Word rate baseline features shape: {word_rate_features.shape}")
# word_rate_features.shape
with open('./word_rate_encoding_results.pkl', "rb") as f:
    word_rate_results = pickle.load(f)
    
# ======================================================
# 3. Function to Run Analysis for a Single Subject
# ======================================================
def run_subject_analysis_shuffle(subject, df_transcript, features_all):
    # results = {"subject": subject}
    results = {}
    debug_mode = False
    if not debug_mode:
        # --- Load Subject ECoG Data ---
        # change the path to your data location
        ecog_fname = f"./podcast_data/sub-{subject}/ieeg/sub-{subject}_task-podcast_desc-highgamma_ieeg.fif"
        print(f"Running analysis for subject {subject}...")
        raw = mne.io.read_raw_fif(ecog_fname, preload=preload, verbose=False)
        print(f"Subject {subject}: Loaded raw data with {len(raw.ch_names)} channels.")
        logging.info(f"Subject {subject}: Loaded raw data with {len(raw.ch_names)} channels.")
        
        # sort by time
        sort_idx = df_transcript['start'].argsort()
        df_transcript = df_transcript.iloc[sort_idx].reset_index(drop=True)
        features_all = features_all[sort_idx]
        features_all = features_all.astype(dtype, copy=False)

        # --- Create Epochs ---
        onset_samples = (df_transcript['start'].values * raw.info['sfreq']).astype(int)
        events = np.column_stack([onset_samples,
                                np.zeros_like(onset_samples, dtype=int),
                                np.ones_like(onset_samples, dtype=int)])
        epochs = mne.Epochs(raw, events, event_id={'word': 1}, tmin=-2.0, tmax=2.0,
                            baseline=None, preload=preload, verbose=False)
        # epochs.load_data()
        # epochs._data = epochs._data.astype(dtype, copy=False)
        
        if len(epochs.events) < len(events):
            logging.warning(f"Subject {subject}: Dropped {len(events)-len(epochs.events)} events during epoching.")
        epochs = epochs.resample(sfreq=32, npad="auto", verbose=False)
        logging.info(f"Subject {subject}: Epochs downsampled to {epochs.info['sfreq']} Hz; shape: {epochs.get_data().shape}")
        
        # --- Align Features ---
        if len(epochs.events) < features_all.shape[0]:
            X_full = features_all[epochs.selection, :]
        else:
            X_full = features_all.copy()
        n_events = len(epochs.events)
        X_full = X_full[:n_events, :]
        X_wc = word_rate_features[:n_events, :]
    
         # Combined features: concatenate features and word rate features.
        # X_combined = np.hstack([X_full, X_wc])
        # print(f"Subject {subject}: Combined features shape: {X_combined.shape}")
        epochs_data = epochs.get_data()   # shape: (n_events, n_channels, n_timepoints)
        n_events, n_channels, n_lags = epochs_data.shape
        Y = epochs_data.reshape(n_events, -1)  # shape: (n_events, n_channels*n_lags)
        
        # --- Define a helper function for regression ---
        def run_regression_shuffle(X_reg, Y_reg):
            fold_corrs = []
            fold_weights = []
            for train_idx, test_idx in cv.split(X_reg):
                X_train = X_reg[train_idx]
                X_test = X_reg[test_idx]
                Y_train = Y_reg[train_idx]
                Y_test = Y_reg[test_idx]
                x_scaler = StandardScaler().fit(X_train)
                X_train_std = x_scaler.transform(X_train)
                X_test_std = x_scaler.transform(X_test)
                y_scaler = StandardScaler().fit(Y_train)
                Y_train_std = y_scaler.transform(Y_train)
                Y_test_std = y_scaler.transform(Y_test)
                n_chunks_fold = int(len(train_idx) * 0.2 / chunk_len)
                if n_chunks_fold < 1:
                    n_chunks_fold = 1
                wt, corrs, _, _, _ = bootstrap_ridge(
                    X_train_std, Y_train_std, X_test_std, Y_test_std,
                    alphas=[10], nboots=1, chunklen=chunk_len, nchunks=n_chunks_fold, single_alpha=True
                )
                fold_corr = corrs.reshape(n_channels, n_lags)
                fold_corrs.append(fold_corr)
                fold_weights.append(wt)
            fold_corrs = np.stack(fold_corrs, axis=0)  # shape: (n_folds, n_channels, n_lags)
            mean_corrs = np.mean(fold_corrs, axis=0)
            performance = np.mean(np.max(mean_corrs, axis=1))
            return performance, fold_corrs, fold_weights

        # ------------------- Shuffle baseline (500×, α=10) -------------------
        n_shuffle = 250
        # n_shuffle = 5 # debug
        null_global = []              # save average peak r
        # null_peak_mat = np.zeros((n_shuffle, n_channels))  # save each electrode peak r
        null_peak_mat = np.empty((n_shuffle, n_channels), dtype=dtype)
        
        # for s in tqdm(range(n_shuffle), desc="Shuffling"):
        for s in range(n_shuffle):
            # estimated time
            time_left = (n_shuffle - s) * (time.time() - start_time) / (s + 1)
            time_left = time_left / 60  # convert to minutes
            print(f"Shuffle {s+1}/{n_shuffle}, estimated time left:{time_left:.2f} mins", end="\n")
            
            idx = np.random.permutation(n_events)
            X_emb_shuf  = X_full[idx]
            X_combined = np.hstack([X_emb_shuf, X_wc])
            shuf_r_mean, shuf_corrs, _ = run_regression_shuffle(X_combined, Y)
            null_global.append(shuf_r_mean)
            
            # mean_corrs = np.mean(shuf_corrs, axis=0)  # shape: (n_channels, n_lags)
            
            word_rate_corrs = word_rate_results[subject]['full_corrs']
            embed_corrs = np.sign(shuf_corrs) * np.sqrt(np.maximum(0, shuf_corrs**2 - word_rate_corrs**2))
            
            mean_corrs = np.mean(embed_corrs, axis=0)  # shape: (n_channels, n_lags)
            null_peak_mat[s] = mean_corrs.max(axis=1)
        null_global = np.array(null_global)
        channel_name = raw.info['ch_names']
        
    else:
        # ------------------- Debug mode (1×, α=10) -------------------
        n_shuffle = 5
        n_channels = 99
        null_global = np.random.rand(n_shuffle)               # save average peak r
        null_peak_mat = np.random.rand(n_shuffle, n_channels)  # save each electrode peak r
        channel_name = ["ch" + str(i) for i in range(n_channels)]
        
    global_thresh = null_global.mean() + 2*null_global.std()     # ≈95%
    null_mean  = null_peak_mat.mean(axis=0)              # (n_channels,)
    null_std   = null_peak_mat.std(axis=0, ddof=1)       # 
    per_ch_thresh = null_mean + 2 * null_std             # (n_channels,)
    per_ch_thresh_95 = np.percentile(null_peak_mat, 95, axis=0)     # 每电极阈值
    results['channel_name'] = channel_name # (n_channels,) 
    # results["null_global"] = null_global
    # results["null_peak_mat"] = null_peak_mat
    
    n_channels = len(channel_name)
    
    results['global_mean'] = np.full(n_channels, null_global.mean()) # (n_channels,)
    results['global_std'] = np.full(n_channels, null_global.std())# (n_channels,)
    results["global_thresh"] = np.full(n_channels, global_thresh) # (n_channels,)
    
    results['per_ch_mean'] = null_mean # (n_channels,)
    results['per_ch_std'] = null_std # (n_channels,)
    results["per_ch_thresh"] = per_ch_thresh # (n_channels,)
    results["per_ch_thresh_95"] = per_ch_thresh_95 # (n_channels,)
    
    del raw, epochs, epochs_data, X_full, Y
    gc.collect()
    return results

# ======================================================
# 4. Loop Over Subjects and Collect Results
# ======================================================
def shuffle_main(name_base, main_features, pca_components, layer_key):
    subject_list = ["01","02","03","04","05","06","07","08","09"]
    # subject_list = ["01"]
    rows = []                         #  (subject, channel)
    output_name = f"{name_base}_{pca_components}_{layer_key}_shuffle.csv"
    output_path = (Path(home_dir) / "LLM_Dual_Stream/Brain_Encoding/Hasson/encoding_results/debug/" / output_name)
    
    for subj in subject_list:
        
        print(f"Running shuffle analysis for subject {subj}...")
        res = run_subject_analysis_shuffle(subj, df_transcript, main_features)

        for ch_idx, ch_name in enumerate(res["channel_name"]):
            rows.append({
                "subject"      : subj,
                "channel_name" : ch_name,
                "global_mean"  : res["global_mean"][ch_idx],   # 全局均值 (相同)
                "global_std"   : res["global_std"][ch_idx],    # 全局 std  (相同)
                "global_thresh"     : res["global_thresh"][ch_idx], # 全局阈值 (相同)
                "per_ch_mean"   : res["per_ch_mean"][ch_idx],
                "per_ch_std"    : res["per_ch_std"][ch_idx],
                "per_ch_thresh": res["per_ch_thresh"][ch_idx],
                "per_ch_thresh_95": res["per_ch_thresh_95"][ch_idx],
            })

    df = pd.DataFrame(rows,
                    columns=["subject","channel_name",
                            "global_mean","global_std","global_thresh",
                            "per_ch_mean","per_ch_std","per_ch_thresh", "per_ch_thresh_95"])
    df.to_csv(output_path, index=False)
    logging.info(f"Saved shuffle baseline to {output_path}")
    
    return

def shuffle_main_per_subj(name_base, main_features, pca_components, subj, layer_key):
    # subject_list = ["01","02","03","04","05","06","07","08","09"]
    rows = []                         # 每行一个 (subject, channel)

    # for subj in subject_list:
    # output_name = f"{control}{name_base}_{pca_components}_{layer_key}_{subj}_shuffle.csv" #old
    # output_path = (Path(home_dir) / "Data/encoding_results_0_4_20_30_500_pt_wr/sh/" / output_name) #old
    
    output_name = f"{name_base}_{pca_components}_{layer_key}_{subj}_shuffle.csv" #old
    output_path = (Path(home_dir) / "Data/encoding_results_0_6_20_30_500_neurips_good_layer/shuffle_new/camera_ready_new" / output_name) #old
    print(f"Running shuffle analysis for subject {subj}...")
    res = run_subject_analysis_shuffle(subj, df_transcript, main_features)

    for ch_idx, ch_name in enumerate(res["channel_name"]):
        rows.append({
            "subject"      : subj,
            "channel_name" : ch_name,
            "global_mean"  : res["global_mean"][ch_idx],   # 
            "global_std"   : res["global_std"][ch_idx],    #
            "global_thresh"     : res["global_thresh"][ch_idx], #
            "per_ch_mean"   : res["per_ch_mean"][ch_idx],
            "per_ch_std"    : res["per_ch_std"][ch_idx],
            "per_ch_thresh": res["per_ch_thresh"][ch_idx],
            "per_ch_thresh_95": res["per_ch_thresh_95"][ch_idx],
        })

    df = pd.DataFrame(rows,
                    columns=["subject","channel_name",
                            "global_mean","global_std","global_thresh",
                            "per_ch_mean","per_ch_std","per_ch_thresh", "per_ch_thresh_95"])
    df.to_csv(output_path, index=False)
    logging.info(f"Saved shuffle baseline to {output_path}")
    return


# args pars function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run shuffle analysis")
    parser.add_argument("--name_base", type=str, required=True, help="Base name for the dataset")
    parser.add_argument("--layer_key", type=str, required=True, help="Layer key for the data")
    parser.add_argument("--pca_components", type=int, default=500, help="Number of PCA components")
    parser.add_argument("--subj", type=str, required=True,  help="Subject ID for analysis")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    name_base = args.name_base
    layer_key = args.layer_key
    subj = args.subj
    pca_components = args.pca_components
    
    data_name = 'Hasson_Qwen2.5-14B_'+name_base+'.pkl'
    # data_path = Path(home_dir) / 'Data' / 'Hasson_pretrained_ridge' / data_name # old
    # data_path = Path(home_dir) / 'Data' / 'Hasson_no_scale_tuned' / data_name # camera-ready 
    data_path = Path(home_dir) / 'Data' / 'Hasson_good_layer' / data_name # camera-ready 
    
    
    with open(data_path, 'rb') as f:
        hasson_data_test = pickle.load(f)

    features = np.stack(hasson_data_test[layer_key].values)
    # pca_components = 500

    if features.shape[1]> pca_components:
        pca = PCA(n_components=pca_components)
        reduced_features = pca.fit_transform(features)
    else:
        reduced_features = features
    print(f"original/reduced features shape: {features.shape}, {reduced_features.shape}")

    shuffle_main_per_subj(name_base, reduced_features, pca_components, subj, layer_key)