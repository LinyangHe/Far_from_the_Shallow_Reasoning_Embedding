# Far from the Shallow: Brain-Predictive Reasoning Embedding

Code accompanying the NeurIPS 2025 paper **“[Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement.](https://arxiv.org/pdf/2510.22860)”** The repository contains everything needed to (1) construct reasoning-focused residual embeddings from large language models, (2) probe them on neurolinguistic and commonsense benchmarks, and (3) evaluate how well they predict intracranial brain recordings during naturalistic language comprehension.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Map](#repository-map)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Core Workflows](#core-workflows)
	- [Residual Reasoning Embedding](#1-residual-reasoning-embedding)
	- [Neurolinguistic Probing](#2-neurolinguistic-probing)
	- [Brain Encoding & Shuffle Baselines](#3-brain-encoding--shuffle-baselines)
6. [Citation](#citation)

## Project Overview

- **Residual disentanglement.** `Residual_Embedding/` contains utilities to regress out lower-level signals between LLM layers, yielding reasoning-specific residuals. These representations isolate deeper computations while controlling for lexical or syntactic confounds.
- **Neurolinguistic & reasoning probes.** `LLM_Probing/` evaluates either native embeddings or residual streams on BLiMP, COMPS, Logic-LLM, ProntoQA, and WinoGrande. The probing stack supports sentence or speech inputs, PCA bottlenecks, bag-of-words baselines, and thinking-mode toggles.
- **Brain encoding.** `Brain_Encoding/` aligns the embeddings with intracranial ECoG responses to naturalistic podcasts. Scripts implement ridge regression with variance partitioning and a shuffle-derived significance baseline.

Together these pieces demonstrate that disentangled residual embeddings better explain both behavioral probing tasks and neural data, supporting the dual-stream reasoning hypothesis laid out in the paper.

## Repository Map

```
├── Brain_Encoding/            # Ridge-based encoding and shuffle baselines for ECoG
│   ├── encoding.py            # Main analysis loop over subjects
│   ├── shuffle.py             # Null distribution via shuffled embeddings
│   └── run_encoding.ipynb     # Notebook launcher (Slurm + Jupyter workflows)
├── LLM_Probing/
│   ├── code/
│   │   ├── config.py          # Experiment configuration object
│   │   ├── model_loader.py    # HF model/processor loader (text & speech)
│   │   ├── neuroling_probing.py  # Main probing runner
│   │   ├── utils.py           # Helpers for export, etc.
│   │   └── neuroling_probing_main.ipynb  # Notebook front-end
│   ├── plot/                  # Plotting scripts & example figures
│   └── data/                  # Benchmark JSONL/CSV files (BLiMP, COMPS, ...)
├── Residual_Embedding/
│   ├── residual_reasoning.py  # ResidualReasoningConstructor class
│   ├── residual_reasoning_main.ipynb
│   └── utils.py               # Token ↦ word aggregation helpers
└── README.md
```

## Environment Setup

Tested with Python 3.10, PyTorch ≥ 2.2, CUDA 12, and Ubuntu 22.04/Windows Subsystem for Linux. Adjust versions as needed for your system.

1. **Create an environment** (example with Conda):

	```powershell
	conda create -n shallow_env python=3.10 -y
	conda activate shallow_env
	```

2. **Install core dependencies** (PyTorch build should match your CUDA toolkit):

	```powershell
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt
	```

	> If you do not wish to maintain a `requirements.txt`, minimally install: `transformers`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `mne`, `nilearn`, `librosa`, `scipy`, `matplotlib`, `tqdm`, `joblib`, `h5py`, `cupy-cuda12x` (optional), and `torchaudio`.

3. **Authenticate with Hugging Face (optional but recommended):**

	```powershell
	huggingface-cli login
	```

4. **Set cache/data paths** referenced in the scripts (e.g., `~/Data/Hasson_good_layer` for embeddings and `./podcast_data` for ECoG files).

## Data Preparation

| Resource | Location/Format | Notes |
| --- | --- | --- |
| **Podcast transcript** | `Brain_Encoding/podcast_transcript.csv` | Word-level timestamps aligned to naturalistic audio. |
| **Word-rate features** | `Brain_Encoding/podcast_feats/` | Baseline regressors from 1.5 s & 3 s windows. |
| **ECoG data** | `Brain_Encoding/podcast_data/sub-XX/ieeg/*.fif` | Download Hasson podcast iEEG from OpenNeuro dataset [ds005574](https://openneuro.org/datasets/ds005574); unzip so each `sub-XX` folder lives under `Brain_Encoding/podcast_data/`. |
| **Residual training corpora** | user-provided text | Paths configured when instantiating `ResidualReasoningConstructor`. |
| **Benchmark datasets** | `LLM_Probing/data/` | Includes BLiMP (jsonl), COMPS, Logic-LLM, ProntoQA, WinoGrande. Ensure licensing/usage terms are satisfied. |
| **Precomputed embeddings** | `~/Data/Hasson_good_layer/*.pkl` | Produced by running the residual constructor or other upstream scripts. |

> Tip: the OpenNeuro CLI can pull a single subject via `openneuro download --dataset ds005574 --include sub-03`. After download, mirror the folder structure shown above so the scripts can find `./podcast_data/sub-XX/ieeg/sub-XX_task-podcast_desc-highgamma_ieeg.fif`.

Update file paths or symlinks if your data live elsewhere. All scripts accept absolute paths, so feel free to edit the defaults before launching large jobs.

## Core Workflows

### 1. Residual Reasoning Embedding

`Residual_Embedding/residual_reasoning.py` defines `ResidualReasoningConstructor`, which:

- extracts hidden states from a target LLM for each token in a corpus,
- learns ridge mappings from shallower layers to deeper layers (optionally with PCA),
- computes layer-wise residuals that emphasize syntactic, semantic, or reasoning streams.

Minimal usage example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from Residual_Embedding.residual_reasoning import ResidualReasoningConstructor

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", torch_dtype="auto", device_map="auto")

rr = ResidualReasoningConstructor(
	 tokenizer=tokenizer,
	 model=model,
	 source_target_layers=[(0, 6), (6, 20), (20, 30)],
	 context_size=50,
	 param_grid={"alpha": [0.1, 1.0, 10.0]},
	 cv_splits=4,
	 pca_components=512,
	 device="cuda"
)

residuals = rr.get_residuals(corpus=["Naturalistic text here ..."], return_hidden_states=False)
```

The accompanying notebook `Residual_Embedding/residual_reasoning_main.ipynb` walks through tuning, logging CV metrics, and exporting `.pkl` files referenced by the brain encoding scripts.

### 2. Neurolinguistic Probing

The probing pipeline lives in `LLM_Probing/code/`. Key components:

- `model_loader.py`: loads text or speech models (Qwen2.5, Qwen2-Audio, Whisper, HuBERT, etc.) with quantization and residual toggles.
- `config.py`: centralizes experiment knobs (layers, pooling, tasks, PCA dim, residual caches, etc.).
- `neuroling_probing.py`: orchestrates embedding extraction, PCA, and cross-validated logistic regression with per-fold metrics.

Example command (PowerShell) to probe BLiMP with residual filters:

```powershell
cd LLM_Probing/code
python neuroling_probing.py `
  --task blimp `
  --model_name Qwen/Qwen2.5-14B `
  --cache_dir C:/hf_cache `
  --hf_token $env:HF_TOKEN `
  --output_file_name ../results/blimp_qwen2p5.csv `
  --pca_dim 256 `
  --residual_mode True `
  --shuffle False
```

For interactive debugging or plotting, open `neuroling_probing_main.ipynb`, which mirrors the script steps with richer commentary.

### 3. Brain Encoding & Shuffle Baselines

`Brain_Encoding/encoding.py` implements ridge regression with variance partitioning to measure how well embeddings (concatenated with word-rate baselines) predict channel-wise ECoG responses. Use the same reduced features that feed into the probing tasks to maintain consistency.

```powershell
cd Brain_Encoding
python encoding.py `
	--name_base 50_20_30 `
	--layer_key residual `
	--pca_components 500
```

`shuffle.py` estimates null distributions by permuting embeddings while keeping word-rate features intact:

```powershell
python shuffle.py `
	--name_base 50_20_30 `
	--layer_key residual `
	--pca_components 500 `
	--subj 03
```

> Choose `name_base`/`layer_key` pairs from the table in the next section (e.g., `('50_0_6', 'layer_0')` for lexicon, `('50_6_20', 'residual')` for meaning). Keep `pca_components` aligned with the features you exported.

> Naming cheat sheet: `name_base` follows the pattern `windowSize_sourceLayer_targetLayer`, so `50_20_30` means a 50-token context window with residuals trained from layer 20 to layer 30. `layer_key` tells the script which stream to load from disk: use `layer_0`/`layer_30` when you want raw activations from those layers, or `residual` when you want the disentangled representation produced by the ridge mapping between the two layers encoded in `name_base`.

Outputs (per subject) include variance-partitioned correlations, residual embedding performance, and per-channel significance thresholds saved under `~/Data/encoding_results_*`.

#### Batching experiments with `run_encoding.ipynb`

The notebook `Brain_Encoding/run_encoding.ipynb` mirrors the Slurm launchers and adds “Jupyter version” cells that execute jobs inline. The naming style used throughout the repo follows the template:

- `subject_list = ["01", …, "09"]` to sweep every Hasson subject.
- `name_base_layer_key_list = [
	('50_0_6', 'layer_0'),      # lexicon baseline
	('50_0_6', 'residual'),     # syntax residuals
	('50_6_20', 'residual'),    # meaning residuals
	('50_20_30', 'layer_30'),   # reasoning (raw layer)
	('50_20_30', 'residual')    # reasoning residuals
	]`
	- The `50_X_Y` prefix encodes the context window (50 tokens) and the source→target layer pair of the residual constructor.
	- `layer_0`/`layer_30` refer to native activations; `residual` indicates the disentangled representation for that span.
- `pca_components = 500` is the default dimensionality used both for exported `.pkl` features and the encoding regression.

Running the “Jupyter version” cells will:

1. Create log directories (`shuffle_logs/`, `encoding_logs/`).
2. Iterate over the subject and `(name_base, layer_key)` grid.
3. Invoke `python -u shuffle.py …` or `python -u encoding.py …` directly via `subprocess.run`, streaming stdout/stderr into per-job log files named `shuffle_{subj}_{name_base}_{layer_key}.txt` and `encoding_{name_base}_{layer_key}_{pca}.txt`.

Use the same parameter names when launching from the command line to keep filenames, checkpoints, and downstream analysis scripts aligned.


## Citation

Please cite the paper if you use this repository:

```
@article{he2025far,
  title={Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement},
  author={He, Linyang and Zhong, Tianjun and Antonello, Richard and Mischler, Gavin and Goldblum, Micah and Mesgarani, Nima},
  journal={arXiv preprint arXiv:2510.22860},
  year={2025}
}
```

For questions or collaboration inquiries, please open an issue or email the corresponding author listed in the manuscript.
