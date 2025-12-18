import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import product
import json
import os
import gc


def mean_cosine_similarity(y_true, y_pred):
    # Normalize row vectors to unit length
    y_true_norm = y_true / np.linalg.norm(y_true, axis=1, keepdims=True)
    y_pred_norm = y_pred / np.linalg.norm(y_pred, axis=1, keepdims=True)

    # Element-wise dot product across rows → cosine similarity for each pair
    sims = np.sum(y_true_norm * y_pred_norm, axis=1)  # shape: (n_samples,)
    return np.mean(sims)

cosine_scorer = make_scorer(mean_cosine_similarity, greater_is_better=True)


class ResidualReasoningConstructor:
    def __init__(
        self,
        tokenizer,
        model,
        source_target_layers: list[tuple[int]],
        context_size: int = 50,
        param_grid: dict = {"alpha": [0.1, 1.0, 10.0]},
        cv_splits: int = 4,
        pca_components: int = None,
        pretrained_ridge_models: list[Ridge] = None,
        device: str | torch.device = "cpu",
        log_path: str = None
    ):
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and device == "cuda") 
            else "cpu"
        )
        self.tokenizer = tokenizer
        if model:
            self.model = model.to(self.device)
            self.model.eval()
        self.source_target_layers = source_target_layers
        self.layers = {layer for pair in self.source_target_layers for layer in pair}
        self.context_size = context_size
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.pca_components = pca_components
        self.pretrained_ridge_models = pretrained_ridge_models
        self.log_path = log_path
        self.ridge_models = self.pretrained_ridge_models or []
        self.return_hidden_states = False

    def _extract_hidden_states(self, corpus: str | list[str], average_layer_0: bool = False):
        # tokenize the corpus
        print("Tokenizing the corpus")
        # corpus already splitted
        if self.context_size == -1:
            input_chunk_list = []
            for sentence in corpus:
                sentence_inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=False,
                    padding=False,
                    add_special_tokens=False
                )
                input_chunk = sentence_inputs["input_ids"]
                input_chunk_list.append(input_chunk)
            total_length = len(input_chunk_list)

        # fixed context window        
        else:
            inputs = self.tokenizer(
                corpus,
                return_tensors="pt",
                truncation=False,
                padding=False,
                add_special_tokens=False
            )
            input_ids = inputs["input_ids"].squeeze(0)
            total_length = input_ids.size(0)

        # extract hidden states
        hidden_dict = {}
                
        for token_index in tqdm(range(total_length), desc="Extracting hidden states"):
        # for token_index in range(total_length):
            if self.context_size == -1:
                input_chunk = input_chunk_list[token_index].to(self.device)
            else:
                start = max(0, token_index - self.context_size + 1)
                input_chunk = input_ids[start:token_index + 1]
                input_chunk = input_chunk.unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_chunk).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_chunk, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states

            for layer in self.layers:
                if average_layer_0 and layer == 0:
                    hidden_token = hidden_states[layer][0].mean(dim=0).cpu()
                else:
                    hidden_token = hidden_states[layer][0, -1].cpu()
                hidden_dict.setdefault(layer, []).append(hidden_token)

        hidden = {}
        for layer in list(hidden_dict.keys()):
            hidden_list = hidden_dict[layer]
            layer_hidden = torch.stack(hidden_list, dim=0).to(torch.float32).numpy()
            hidden[layer] = layer_hidden

            del hidden_dict[layer]
            del hidden_list  # Optional, to be safe
            gc.collect()

        return hidden


    def _fit(self, corpus: str | list[str], hidden=None, no_fit=False, no_scale=False, return_hidden_pair_list=True):
        if not hidden:
            hidden = self._extract_hidden_states(corpus)
        if self.pca_components:
            pca = PCA(self.pca_components)
        if return_hidden_pair_list:
            hidden_pair_list = []

        for ridge_index, (source_layer, target_layer) in enumerate(self.source_target_layers):
            print(f"Fitting Ridge Model: source_layer = {source_layer}, target_layer = {target_layer}")
            if self.context_size != -1:
                source_hidden = hidden[source_layer][self.context_size:]
            else:
                source_hidden = hidden[source_layer]
            
            if no_scale:
                source_hidden_scaled = source_hidden
                if return_hidden_pair_list:
                    source_hidden_scaled_full = hidden[source_layer]
            else:
                scaler = StandardScaler()
                source_hidden_scaled = scaler.fit_transform(source_hidden)
                if return_hidden_pair_list:
                    source_hidden_scaled_full = scaler.transform(hidden[source_layer])
                del scaler

            if self.context_size != -1:
                target_hidden = hidden[target_layer][self.context_size:]
            else:
                target_hidden = hidden[target_layer]
            if return_hidden_pair_list:
                target_hidden_full = hidden[target_layer]
            if (not self.return_hidden_states) and (ridge_index >= len(self.source_target_layers) - 1):
                del hidden

            if self.pca_components:
                print(f"Conducting PCA dimension reduction: {source_hidden.shape[-1]} --> {self.pca_components}")
                source_hidden_scaled = pca.fit_transform(source_hidden_scaled)
                if return_hidden_pair_list:
                    source_hidden_scaled_full = pca.transform(source_hidden_scaled_full)
                target_hidden = pca.fit_transform(target_hidden)
                if return_hidden_pair_list:
                    target_hidden_full = pca.transform(target_hidden_full)
                del pca
            if return_hidden_pair_list:
                hidden_pair_list.append((source_hidden_scaled_full, target_hidden_full))

            if self.pretrained_ridge_models:
                print("Using pretrained ridge models.")
            elif no_fit:
                print("Model fitting skipped.")
            else:
                ridge = Ridge()
                grid = GridSearchCV(
                    ridge, 
                    self.param_grid, 
                    cv=KFold(n_splits=self.cv_splits, shuffle=True, random_state=42), 
                    scoring={
                        'mse': 'neg_mean_squared_error',
                        'r2': 'r2',
                        'cosine': cosine_scorer
                    },
                    refit="mse",
                    verbose=3
                )
                grid.fit(source_hidden_scaled, target_hidden)
                self.ridge_models.append(grid.best_estimator_)

                # log
                if self.log_path:
                    cv_log = {}
                    cv_results = grid.cv_results_
                    for param_index in range(len(cv_results["params"])):
                        for metric in grid.scoring.keys():
                            param_log = cv_log.setdefault(str(cv_results["params"][param_index]), {})
                            param_log[metric] = cv_results[f"mean_test_{metric}"][param_index]

                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)

                    with open(os.path.join(self.log_path, "cv_log.jsonl"), "a") as f:
                        f.write(f"Config: context_size = {self.context_size}, source_layer = {source_layer}, target_layer = {target_layer}, refit = {grid.refit}, no_scale = {no_scale}\n")
                        f.write(json.dumps(cv_log) + "\n")
                        f.write(f"Best hyperparameters: {grid.best_params_}\n\n")
                print(f"Best hyperparameters: {grid.best_params_}")

        if self.return_hidden_states:
            return hidden, hidden_pair_list
        if return_hidden_pair_list:
            return hidden_pair_list

    def _compute_residuals(self, source_hidden_scaled, target_hidden, ridge_index) -> np.ndarray:
        target_hidden_predicted = self.ridge_models[ridge_index].predict(source_hidden_scaled)
        residuals = target_hidden - target_hidden_predicted

        return residuals

    def get_residuals(self, corpus: str | list[str], return_hidden_states: bool = False) -> np.ndarray:
        if return_hidden_states:
            self.return_hidden_states = True
            hidden, hidden_pair_list = self._fit(corpus)
        else:
            hidden_pair_list = self._fit(corpus)

        residual_list = []

        for ridge_index, (source_hidden_scaled, target_hidden) in enumerate(hidden_pair_list):
            residual = self._compute_residuals(source_hidden_scaled, target_hidden, ridge_index)
            residual_list.append(residual)

        if return_hidden_states:
            return hidden, residual_list
        
        return residual_list
