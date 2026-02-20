#!/usr/bin/env python
"""
Brain encoding analysis using ridge regression with variance partitioning.

Aligns LLM embeddings with intracranial ECoG responses to naturalistic podcasts.
Ridge regression with cross-validation measures per-channel predictive performance,
and variance partitioning isolates embedding contributions beyond word-rate baselines.
"""

import os, pickle, logging, argparse, time, gc
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from pathlib import Path
from tqdm.notebook import tqdm

# Import the ridge regression function
from ridge_utils.ridge import bootstrap_ridge

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so the code works regardless of the
# caller's working directory.  Users can override every path via CLI flags.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# ======================================================
# Global Parameters for Regression and CV
# ======================================================
dtype = np.float64
alphas = np.array([10])           # candidate ridge penalty values
nboots = 5                        # number of bootstrap iterations
chunk_len = 32                    # events per contiguous chunk
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
        Path to pickled word-rate encoding results (``word_rate_encoding_results.pkl``).

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
# Single-Subject Analysis
# ======================================================
def run_subject_analysis_main(subject, df_transcript, features_all,
                              word_rate_features, word_rate_results,
                              ecog_dir):
    """
    Run encoding analysis for a single subject using ridge regression
    with variance partitioning.

    Parameters
    ----------
    subject : str
        Subject identifier (e.g. ``"03"``).
    df_transcript : DataFrame
        Word-level transcript with a ``start`` column (seconds).
    features_all : ndarray, shape (n_words, n_features)
        Embedding features aligned to transcript rows.
    word_rate_features : ndarray
        Word-rate baseline regressor matrix.
    word_rate_results : dict
        Pre-computed word-rate-only encoding results keyed by subject.
    ecog_dir : str or Path
        Root directory containing ``sub-XX/ieeg/`` folders.
    """
    results = {"subject": subject}

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

    # Combined features: concatenate embedding and word rate features.
    X_combined = np.hstack([X_full, X_wc])
    print(f"Subject {subject}: Combined features shape: {X_combined.shape}")

    epochs_data = epochs.get_data()   # shape: (n_events, n_channels, n_timepoints)
    n_events, n_channels, n_lags = epochs_data.shape
    Y = epochs_data.reshape(n_events, -1)  # shape: (n_events, n_channels*n_lags)

    # --- Helper for ridge regression ---
    def run_regression(X_reg, Y_reg):
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
                alphas, nboots=nboots, chunklen=chunk_len,
                nchunks=n_chunks_fold, single_alpha=False
            )
            fold_corrs.append(corrs.reshape(n_channels, n_lags))
            fold_weights.append(wt)
            print(wt.shape)
        fold_corrs = np.stack(fold_corrs, axis=0)  # (n_folds, n_channels, n_lags)
        mean_corrs = np.mean(fold_corrs, axis=0)
        performance = np.mean(np.max(mean_corrs, axis=1))
        return performance, fold_corrs, fold_weights

    # --- Run Regressions ---
    full_perf, full_corrs, full_weights = run_regression(X_combined, Y)
    full_r2 = full_corrs ** 2
    word_rate_corrs = word_rate_results[subject]['full_corrs']
    wr_r2 = word_rate_corrs ** 2

    # Variance partitioning: average before partitioning
    mean_delta_r2 = np.maximum(0, np.mean(full_r2 - wr_r2, axis=0))
    mean_full_corrs = np.mean(full_corrs, axis=0)
    embed_mean_corrs = np.sign(mean_full_corrs) * np.sqrt(mean_delta_r2)
    embed_perf = np.mean(np.max(embed_mean_corrs, axis=1))

    results["full_perf"]        = full_perf
    results["full_corrs"]       = full_corrs
    results["n_channels"]       = n_channels
    results["n_lags"]           = n_lags
    results['embed_mean_corrs'] = embed_mean_corrs
    results['embed_perf']       = embed_perf
    return results


# ======================================================
# Main Loop
# ======================================================
def encoding_main(name_base, main_features, pca_components, layer_key,
                  df_transcript, word_rate_features, word_rate_results,
                  ecog_dir, output_dir):
    """Iterate over all subjects and save combined results.

    Parameters
    ----------
    output_dir : str or Path
        Directory where the result ``.pkl`` file will be written.
    """
    subject_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

    output_name = f"{name_base}_{pca_components}_{layer_key}.pkl"
    output_path = Path(output_dir) / output_name

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_path.parent}")
    print(f"Output path: {output_path}")

    all_results = {}
    for subj in subject_list:
        res = run_subject_analysis_main(
            subj, df_transcript, main_features,
            word_rate_features, word_rate_results, ecog_dir
        )
        all_results[subj] = res

    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    logging.info(f"Saved analysis results to {output_path}")


# ======================================================
# CLI
# ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run brain encoding analysis with ridge regression"
    )
    # --- experiment knobs ---
    parser.add_argument("--name_base", type=str, required=True,
                        help="Residual identifier: windowSize_sourceLayer_targetLayer "
                             "(e.g. 50_20_30)")
    parser.add_argument("--layer_key", type=str, required=True,
                        help="Stream to load: layer_<idx> for raw activations, "
                             "or 'residual' for the disentangled stream")
    parser.add_argument("--pca_components", type=int, default=500,
                        help="Number of PCA components (default: 500)")

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
                        default=str(SCRIPT_DIR / "encoding_results"),
                        help="Directory to save encoding result .pkl files "
                             "(default: <script_dir>/encoding_results)")
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
    log_path = Path(args.output_dir) / "encoding.log"
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

    encoding_main(
        args.name_base, reduced_features, pca_components, args.layer_key,
        df_transcript, word_rate_features, word_rate_results,
        args.ecog_dir, args.output_dir,
    )
