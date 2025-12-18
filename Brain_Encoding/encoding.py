#!/usr/bin/env python
"""
Code for brain encoding analysis using ridge regression with variance partitioning.
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
def run_subject_analysis_main(subject, df_transcript, features_all):
    """
    Run encoding analysis for a single subject using ridge regression with variance partitioning.
    """

    results = {"subject": subject}
    # --- Load Subject ECoG Data ---
    # change to your local path
    ecog_fname = f"./podcast_data/sub-{subject}/ieeg/sub-{subject}_task-podcast_desc-highgamma_ieeg.fif"
    print(f"Running analysis for subject {subject}...")
    raw = mne.io.read_raw_fif(ecog_fname, preload=True, verbose=False)
    print(f"Subject {subject}: Loaded raw data with {len(raw.ch_names)} channels.")
    logging.info(f"Subject {subject}: Loaded raw data with {len(raw.ch_names)} channels.")
    
    # sort by time
    sort_idx = df_transcript['start'].argsort()
    df_transcript = df_transcript.iloc[sort_idx].reset_index(drop=True)
    features_all = features_all[sort_idx]

    # --- Create Epochs ---
    onset_samples = (df_transcript['start'].values * raw.info['sfreq']).astype(int)
    events = np.column_stack([onset_samples,
                              np.zeros_like(onset_samples, dtype=int),
                              np.ones_like(onset_samples, dtype=int)])
    epochs = mne.Epochs(raw, events, event_id={'word': 1}, tmin=-2.0, tmax=2.0,
                        baseline=None, preload=True, verbose=False)
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
    
    # Combined features: concatenate QA and word rate features.
    X_combined = np.hstack([X_full, X_wc])
    print(f"Subject {subject}: Combined features shape: {X_combined.shape}")
    
    epochs_data = epochs.get_data()   # shape: (n_events, n_channels, n_timepoints)
    n_events, n_channels, n_lags = epochs_data.shape
    Y = epochs_data.reshape(n_events, -1)  # shape: (n_events, n_channels*n_lags)
    
    # --- Define a helper function for regression ---
    def run_regression(X_reg, Y_reg):
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
                alphas, nboots=nboots, chunklen=chunk_len, nchunks=n_chunks_fold, single_alpha=False
            )
            fold_corr = corrs.reshape(n_channels, n_lags)
            fold_corrs.append(fold_corr)
            fold_weights.append(wt)
            print(wt.shape)
        fold_corrs = np.stack(fold_corrs, axis=0)  # shape: (n_folds, n_channels, n_lags)
        mean_corrs = np.mean(fold_corrs, axis=0)
        performance = np.mean(np.max(mean_corrs, axis=1))
        return performance, fold_corrs, fold_weights
        
    # --- Run Regressions ---
    full_perf, full_corrs, full_weights = run_regression(X_combined, Y)
    full_r2 = full_corrs ** 2
    word_rate_corrs = word_rate_results[subject]['full_corrs']
    wr_r2 = word_rate_corrs ** 2
    
    # method 2: average before variance partitioning
    mean_delta_r2 = np.maximum(0, np.mean(full_r2 - wr_r2, axis=0))  # average across folds
    mean_full_corrs = np.mean(full_corrs, axis=0)  # average across folds
    embed_mean_corrs = np.sign(mean_full_corrs) * np.sqrt(mean_delta_r2)
    embed_perf = np.mean(np.max(embed_mean_corrs, axis=1))  # max cor across time lag
    
    
    # --- Variance Partitioning ---
    # Save performance metrics and results.
    results["full_perf"] = full_perf
    results["full_corrs"] = full_corrs
    # results["full_weights"] = full_weights # if you want to save weights, uncomment this line
    results["n_channels"] = n_channels
    results["n_lags"] = n_lags
    results['embed_mean_corrs'] = embed_mean_corrs
    results['embed_perf'] = embed_perf
    return results

# ======================================================
# 4. Loop Over Subjects and Collect Results
# ======================================================
def encoding_main(name_base, main_features, pca_components, layer_key):
    subject_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    # subject_list = ["01"] # debugging
    
    output_name = name_base+f'_{pca_components}_{layer_key}.pkl'
    
    output_path = Path(home_dir)/'Data'/f'encoding_results_{pca_components}_neurips_qwen2.5_1.5b'/ output_name
    
    print(f"Output path: {output_path}")
    # test if output_path exists, if not, create it
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_path.parent}")
    else:
        print(f"Directory already exists: {output_path.parent}")
    
    all_results = {}
    for subj in subject_list:
        res = run_subject_analysis_main(subj, df_transcript, main_features)
        all_results[subj] = res
        
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    logging.info(f"Saved analysis results to {output_path}")

# args pars function
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run shuffle analysis")
    parser.add_argument("--name_base", type=str, required=True, help="Base name for the dataset")
    parser.add_argument("--layer_key", type=str, required=True, help="Layer key for the data")
    parser.add_argument("--pca_components", type=int, default=500, help="Number of PCA components")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    name_base = args.name_base
    layer_key = args.layer_key
    pca_components = args.pca_components
        
    data_name = 'Hasson_Qwen2.5-14B_'+name_base+'.pkl'
    data_path = Path(home_dir) / 'Data' / 'Hasson_good_layer' / data_name
    
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

    encoding_main(name_base, reduced_features, pca_components, layer_key)