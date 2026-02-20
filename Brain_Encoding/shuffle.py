#!/usr/bin/env python
"""
Shuffle analysis for brain encoding with ridge regression.

Estimates null distributions by permuting embeddings while keeping word-rate
features intact, producing per-channel significance thresholds.
"""

import os, pickle, logging, argparse, time, gc
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from pathlib import Path

# Import the ridge regression function
from ridge_utils.ridge import bootstrap_ridge

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so the code works regardless of the
# caller's working directory.  Users can override every path via CLI flags.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# ======================================================
# Global Parameters
# ======================================================
dtype = np.float64
alphas = np.array([10])
nboots = 5
chunk_len = 32
cv = KFold(n_splits=5, shuffle=False)


# ======================================================
# Data Loading
# ======================================================
def load_common_data(transcript_path, sec1_5_feats_path, sec3_feats_path,
                     word_rate_file):
    """Load transcript, word-rate features, and pre-computed word-rate results.

    Parameters
    ----------
    transcript_path : str or Path
        Path to ``podcast_transcript.csv``.
    sec1_5_feats_path : str or Path
        Path to the 1.5 s word-rate feature CSV.
    sec3_feats_path : str or Path
        Path to the 3 s word-rate feature CSV.
    word_rate_file : str or Path
        Path to pickled word-rate encoding results.

    Returns
    -------
    df_transcript : DataFrame
    word_rate_features : ndarray
    word_rate_results : dict
    """
    df_transcript = pd.read_csv(transcript_path)
    logging.info(f"Transcript loaded with {len(df_transcript)} entries.")

    df_sec1_5 = pd.read_csv(sec1_5_feats_path, sep=",", index_col=0)
    df_sec3   = pd.read_csv(sec3_feats_path, sep=",", index_col=0)
    wc_sec1_5 = df_sec1_5.index.to_series().astype(str).apply(
        lambda x: len(x.split())).values
    wc_sec3   = df_sec3.index.to_series().astype(str).apply(
        lambda x: len(x.split())).values
    word_rate_features = np.column_stack([wc_sec1_5, wc_sec3])
    logging.info(f"Word rate baseline features shape: {word_rate_features.shape}")

    with open(word_rate_file, "rb") as f:
        word_rate_results = pickle.load(f)

    return df_transcript, word_rate_features, word_rate_results


# ======================================================
# Single-Subject Shuffle Analysis
# ======================================================
def run_subject_analysis_shuffle(subject, df_transcript, features_all,
                                 word_rate_features, word_rate_results,
                                 ecog_dir, n_shuffle=250):
    """Run shuffle-based null distribution analysis for one subject.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g. ``"03"``).
    df_transcript : DataFrame
        Word-level transcript.
    features_all : ndarray
        Embedding features aligned to transcript rows.
    word_rate_features : ndarray
        Baseline word-rate regressor matrix.
    word_rate_results : dict
        Pre-computed word-rate-only results keyed by subject.
    ecog_dir : str or Path
        Root directory containing ``sub-XX/ieeg/`` folders.
    n_shuffle : int
        Number of permutation iterations (default: 250).
    """
    results = {}

    ecog_fname = str(
        Path(ecog_dir) / f"sub-{subject}" / "ieeg"
        / f"sub-{subject}_task-podcast_desc-highgamma_ieeg.fif"
    )
    print(f"Running analysis for subject {subject}...")
    raw = mne.io.read_raw_fif(ecog_fname, preload=True, verbose=False)
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
                        baseline=None, preload=True, verbose=False)
    if len(epochs.events) < len(events):
        logging.warning(
            f"Subject {subject}: Dropped {len(events)-len(epochs.events)} events during epoching."
        )
    epochs = epochs.resample(sfreq=32, npad="auto", verbose=False)
    logging.info(
        f"Subject {subject}: Epochs downsampled to {epochs.info['sfreq']} Hz; "
        f"shape: {epochs.get_data().shape}"
    )

    # --- Align Features ---
    if len(epochs.events) < features_all.shape[0]:
        X_full = features_all[epochs.selection, :]
    else:
        X_full = features_all.copy()
    n_events = len(epochs.events)
    X_full = X_full[:n_events, :]
    X_wc = word_rate_features[:n_events, :]

    epochs_data = epochs.get_data()   # shape: (n_events, n_channels, n_timepoints)
    n_events, n_channels, n_lags = epochs_data.shape
    Y = epochs_data.reshape(n_events, -1)  # shape: (n_events, n_channels*n_lags)

    # --- Helper for ridge regression ---
    def run_regression_shuffle(X_reg, Y_reg):
        fold_corrs = []
        fold_weights = []
        for train_idx, test_idx in cv.split(X_reg):
            X_train, X_test = X_reg[train_idx], X_reg[test_idx]
            Y_train, Y_test = Y_reg[train_idx], Y_reg[test_idx]
            x_scaler = StandardScaler().fit(X_train)
            X_train_std = x_scaler.transform(X_train)
            X_test_std  = x_scaler.transform(X_test)
            y_scaler = StandardScaler().fit(Y_train)
            Y_train_std = y_scaler.transform(Y_train)
            Y_test_std  = y_scaler.transform(Y_test)
            n_chunks_fold = max(1, int(len(train_idx) * 0.2 / chunk_len))
            wt, corrs, _, _, _ = bootstrap_ridge(
                X_train_std, Y_train_std, X_test_std, Y_test_std,
                alphas=[10], nboots=1, chunklen=chunk_len,
                nchunks=n_chunks_fold, single_alpha=True
            )
            fold_corrs.append(corrs.reshape(n_channels, n_lags))
            fold_weights.append(wt)
        fold_corrs = np.stack(fold_corrs, axis=0)  # (n_folds, n_channels, n_lags)
        mean_corrs = np.mean(fold_corrs, axis=0)
        performance = np.mean(np.max(mean_corrs, axis=1))
        return performance, fold_corrs, fold_weights

    # ---- Shuffle baseline ----
    start_time = time.time()
    null_global = []
    null_peak_mat = np.empty((n_shuffle, n_channels), dtype=dtype)

    for s in range(n_shuffle):
        elapsed = time.time() - start_time
        time_left = (n_shuffle - s) * elapsed / max(s, 1) / 60
        print(f"Shuffle {s+1}/{n_shuffle}, estimated time left: {time_left:.2f} mins")

        idx = np.random.permutation(n_events)
        X_emb_shuf = X_full[idx]
        X_combined = np.hstack([X_emb_shuf, X_wc])
        shuf_r_mean, shuf_corrs, _ = run_regression_shuffle(X_combined, Y)
        null_global.append(shuf_r_mean)

        word_rate_corrs = word_rate_results[subject]['full_corrs']
        embed_corrs = np.sign(shuf_corrs) * np.sqrt(
            np.maximum(0, shuf_corrs**2 - word_rate_corrs**2))
        mean_corrs = np.mean(embed_corrs, axis=0)  # (n_channels, n_lags)
        null_peak_mat[s] = mean_corrs.max(axis=1)

    null_global = np.array(null_global)
    channel_name = raw.info['ch_names']
    n_channels = len(channel_name)

    global_thresh = null_global.mean() + 2 * null_global.std()
    null_mean  = null_peak_mat.mean(axis=0)
    null_std   = null_peak_mat.std(axis=0, ddof=1)
    per_ch_thresh = null_mean + 2 * null_std
    per_ch_thresh_95 = np.percentile(null_peak_mat, 95, axis=0)

    results['channel_name']     = channel_name
    results['global_mean']      = np.full(n_channels, null_global.mean())
    results['global_std']       = np.full(n_channels, null_global.std())
    results["global_thresh"]    = np.full(n_channels, global_thresh)
    results['per_ch_mean']      = null_mean
    results['per_ch_std']       = null_std
    results["per_ch_thresh"]    = per_ch_thresh
    results["per_ch_thresh_95"] = per_ch_thresh_95

    del raw, epochs, epochs_data, X_full, Y
    gc.collect()
    return results


# ======================================================
# Main Loops
# ======================================================
def shuffle_main(name_base, main_features, pca_components, layer_key,
                 df_transcript, word_rate_features, word_rate_results,
                 ecog_dir, output_dir, n_shuffle=250):
    """Run shuffle for all subjects and save a combined CSV."""
    subject_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    rows = []
    output_name = f"{name_base}_{pca_components}_{layer_key}_shuffle.csv"
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for subj in subject_list:
        print(f"Running shuffle analysis for subject {subj}...")
        res = run_subject_analysis_shuffle(
            subj, df_transcript, main_features,
            word_rate_features, word_rate_results, ecog_dir, n_shuffle
        )
        for ch_idx, ch_name in enumerate(res["channel_name"]):
            rows.append({
                "subject":        subj,
                "channel_name":   ch_name,
                "global_mean":    res["global_mean"][ch_idx],
                "global_std":     res["global_std"][ch_idx],
                "global_thresh":  res["global_thresh"][ch_idx],
                "per_ch_mean":    res["per_ch_mean"][ch_idx],
                "per_ch_std":     res["per_ch_std"][ch_idx],
                "per_ch_thresh":  res["per_ch_thresh"][ch_idx],
                "per_ch_thresh_95": res["per_ch_thresh_95"][ch_idx],
            })

    df = pd.DataFrame(rows, columns=[
        "subject", "channel_name",
        "global_mean", "global_std", "global_thresh",
        "per_ch_mean", "per_ch_std", "per_ch_thresh", "per_ch_thresh_95"
    ])
    df.to_csv(output_path, index=False)
    logging.info(f"Saved shuffle baseline to {output_path}")


def shuffle_main_per_subj(name_base, main_features, pca_components, subj,
                          layer_key, df_transcript, word_rate_features,
                          word_rate_results, ecog_dir, output_dir,
                          n_shuffle=250):
    """Run shuffle for a single subject and save a per-subject CSV."""
    rows = []
    output_name = f"{name_base}_{pca_components}_{layer_key}_{subj}_shuffle.csv"
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running shuffle analysis for subject {subj}...")
    res = run_subject_analysis_shuffle(
        subj, df_transcript, main_features,
        word_rate_features, word_rate_results, ecog_dir, n_shuffle
    )

    for ch_idx, ch_name in enumerate(res["channel_name"]):
        rows.append({
            "subject":        subj,
            "channel_name":   ch_name,
            "global_mean":    res["global_mean"][ch_idx],
            "global_std":     res["global_std"][ch_idx],
            "global_thresh":  res["global_thresh"][ch_idx],
            "per_ch_mean":    res["per_ch_mean"][ch_idx],
            "per_ch_std":     res["per_ch_std"][ch_idx],
            "per_ch_thresh":  res["per_ch_thresh"][ch_idx],
            "per_ch_thresh_95": res["per_ch_thresh_95"][ch_idx],
        })

    df = pd.DataFrame(rows, columns=[
        "subject", "channel_name",
        "global_mean", "global_std", "global_thresh",
        "per_ch_mean", "per_ch_std", "per_ch_thresh", "per_ch_thresh_95"
    ])
    df.to_csv(output_path, index=False)
    logging.info(f"Saved shuffle baseline to {output_path}")


# ======================================================
# CLI
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shuffle-based null-distribution analysis for brain encoding"
    )
    # --- experiment knobs ---
    parser.add_argument("--name_base", type=str, required=True,
                        help="Residual identifier: windowSize_sourceLayer_targetLayer "
                             "(e.g. 50_20_30)")
    parser.add_argument("--layer_key", type=str, required=True,
                        help="Stream to load: layer_<idx> or 'residual'")
    parser.add_argument("--pca_components", type=int, default=500,
                        help="Number of PCA components (default: 500)")
    parser.add_argument("--subj", type=str, required=True,
                        help="Subject ID (e.g. 03)")
    parser.add_argument("--n_shuffle", type=int, default=250,
                        help="Number of shuffle iterations (default: 250)")

    # --- paths (all have sensible defaults relative to this script) ---
    parser.add_argument("--embedding_dir", type=str,
                        default=str(SCRIPT_DIR / "embeddings"),
                        help="Directory containing precomputed embedding .pkl files "
                             "(default: <script_dir>/embeddings)")
    parser.add_argument("--ecog_dir", type=str,
                        default=str(SCRIPT_DIR / "podcast_data"),
                        help="Root of the OpenNeuro-style ECoG dataset (contains sub-XX/) "
                             "(default: <script_dir>/podcast_data)")
    parser.add_argument("--output_dir", type=str,
                        default=str(SCRIPT_DIR / "shuffle_results"),
                        help="Directory to save shuffle result CSV files "
                             "(default: <script_dir>/shuffle_results)")
    parser.add_argument("--transcript_path", type=str,
                        default=str(SCRIPT_DIR / "podcast_transcript.csv"),
                        help="Path to the word-level transcript CSV")
    parser.add_argument("--word_rate_file", type=str,
                        default=str(SCRIPT_DIR / "word_rate_encoding_results.pkl"),
                        help="Path to precomputed word-rate encoding results")
    parser.add_argument("--sec1_5_feats_path", type=str,
                        default=str(SCRIPT_DIR / "podcast_feats"
                                    / "sec_1.5-selected-podcast"
                                    / "___podcasts-story___.csv"),
                        help="Path to 1.5 s word-rate feature CSV")
    parser.add_argument("--sec3_feats_path", type=str,
                        default=str(SCRIPT_DIR / "podcast_feats"
                                    / "sec_3-podcast" / "sec_3"
                                    / "___podcasts-story___.csv"),
                        help="Path to 3 s word-rate feature CSV")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-14B",
                        help="Model identifier used in embedding filename "
                             "(default: Qwen2.5-14B)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --- configure logging ---
    log_path = Path(args.output_dir) / "shuffle.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_path),
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    # --- load common data ---
    df_transcript, word_rate_features, word_rate_results = load_common_data(
        args.transcript_path,
        args.sec1_5_feats_path,
        args.sec3_feats_path,
        args.word_rate_file,
    )

    # --- load embedding features ---
    data_name = f"Hasson_{args.model_name}_{args.name_base}.pkl"
    data_path = Path(args.embedding_dir) / data_name
    print(f"Loading embeddings from {data_path}")

    with open(data_path, 'rb') as f:
        hasson_data = pickle.load(f)

    features = np.stack(hasson_data[args.layer_key].values)
    pca_components = args.pca_components

    if features.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        reduced_features = pca.fit_transform(features)
    else:
        reduced_features = features
    print(f"Original/reduced features shape: {features.shape}, {reduced_features.shape}")

    shuffle_main_per_subj(
        args.name_base, reduced_features, pca_components, args.subj,
        args.layer_key, df_transcript, word_rate_features, word_rate_results,
        args.ecog_dir, args.output_dir, args.n_shuffle,
    )
