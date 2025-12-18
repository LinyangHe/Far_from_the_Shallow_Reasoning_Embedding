import os
import string
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchaudio
import pickle
import gc
import joblib
import scipy.io as sio
import utils
import h5py


from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline

import cupy as cp
# from cuml.decomposition import PCA as cuPCA
# from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.feature_extraction.text import CountVectorizer # for bag-of-words



class NeurolingProbing:
    """
    Manager for running multiple linguistic probing tasks, including sentence embedding
    extraction, probing classification, and task loading/execution utilities.
    """

    def __init__(self, config, model, tokenizer=None, processor=None):
        """
        :param config: ExperimentConfig instance
        :param model:  loaded pretrained language model
        :param tokenizer: matching tokenizer for the model
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.classifiers = [
            ("Logistic Regression", LogisticRegression(max_iter=10000))
        ]
        # self.classifiers_gpu = [
            # ("cuLogistic Regression", cuLogisticRegression(max_iter=10000))
        # ]

    @staticmethod
    def _find_files(folder_path, endswith='.csv'):
        """
        Locate every file under the given folder (recursively) whose name ends with the
        provided suffix.
        """
        found_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(endswith):
                    found_files.append(os.path.join(root, file))
        return found_files

    @staticmethod
    def _remove_end_punctuation(sentence):
        """
        Remove trailing punctuation marks from a sentence.
        """
        if isinstance(sentence, str):
            while sentence and sentence[-1] in string.punctuation:
                sentence = sentence[:-1]
        return sentence

    def get_sentence_embeddings(self, sentences):
        """
        Run a batch of sentences through the model to obtain the final sentence vectors.
        :param sentences: raw sentence list that will be tokenized
        :return: sentence-level vector representations
        """
        if self.config.thinking_mode_off:
            sentences = [s + " /no_think" for s in sentences]
        encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.config.device)
        
        self.model.eval()
        if self.config.speech_mode:
            input_ids = encoded_inputs['input_values'] 
        else:
            input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        num_sent = len(input_ids)
        batch_size = self.config.batch_size
        num_batches = num_sent // batch_size + int(num_sent % batch_size != 0)

        activation = []
        print(f'Number of batches: {num_batches}, Number of sentences: {num_sent}')
        for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sent)
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]

            with torch.no_grad():
                if self.config.speech_mode:
                    outputs = self.model(
                    input_values=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True
                    )
                else:
                    outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=self.config.output_hidden_states,
                    output_attentions=self.config.output_attentions
                    )

            if self.config.sent_pooling == 'last':
                if self.config.target_layer is None:
                    cat_dim = 1
                    embedding_batch = torch.stack(outputs.hidden_states)
                    # print(embedding_batch.shape, batch_attention_mask.shape)  
                    lengths = batch_attention_mask.sum(dim=1)
                    expanded_lengths = lengths.unsqueeze(0).unsqueeze(-1).expand(
                        embedding_batch.size(0),
                        embedding_batch.size(1),
                        1
                    )
                    last_word_embeddings = torch.gather(
                        embedding_batch,
                        2,
                        expanded_lengths.unsqueeze(-1).expand(
                            -1, -1, -1, embedding_batch.size(3)
                        ) - 1
                    ).squeeze(2)
                    
                else:
                    cat_dim = 1
                    embedding_batch = outputs.hidden_states[self.config.target_layer]
                    lengths = batch_attention_mask.sum(dim=1)
                    last_word_embeddings = embedding_batch[torch.arange(embedding_batch.size(0)), lengths - 1]
                    last_word_embeddings = last_word_embeddings.unsqueeze(0)
                # check if has NaN
                if torch.isnan(last_word_embeddings).sum().item() > 0:
                    print(f'NaN found in last_word_embeddings: {torch.isnan(last_word_embeddings).sum().item()}')
                    return
                activation.append(last_word_embeddings.cpu().detach())    
            elif self.config.sent_pooling == 'mean':
                if self.config.target_layer is None:
                    cat_dim = 1  # dimension 1 corresponds to batch_size
                    embedding_batch = torch.stack(outputs.hidden_states)  # Shape: (num_layers, batch_size, seq_len, hidden_size)
                    expanded_mask = batch_attention_mask.unsqueeze(0).unsqueeze(-1)  # Shape: (1, batch_size, seq_len, 1)
                    expanded_mask = expanded_mask.expand_as(embedding_batch)  # Shape: (num_layers, batch_size, seq_len, hidden_size)
                    masked_embeddings = embedding_batch * expanded_mask  # Mask padded positions as 0
                    sentence_lengths = batch_attention_mask.sum(dim=1, keepdim=True).unsqueeze(0)  # Shape: (1, batch_size, 1)
                    mean_embeddings = masked_embeddings.sum(dim=2) / sentence_lengths  # Shape: (num_layers, batch_size, hidden_size)
                else:
                    cat_dim = 1  # dimension 0 represents the number of sentences
                    embedding_batch = outputs.hidden_states[self.config.target_layer]  # Shape: (batch_size, seq_len, hidden_size)
                    expanded_mask = batch_attention_mask.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
                    masked_embeddings = embedding_batch * expanded_mask  # Mask padded positions as 0
                    sentence_lengths = batch_attention_mask.sum(dim=1, keepdim=True)  # Shape: (batch_size, 1)
                    mean_embeddings = masked_embeddings.sum(dim=1) / sentence_lengths  # Shape: (batch_size, hidden_size)
                    mean_embeddings = mean_embeddings.unsqueeze(0)
                activation.append(mean_embeddings.cpu().detach())
            else:
                raise ValueError(f"Invalid pooling method: {self.config.sent_pooling}")     
                    
            del outputs
            torch.cuda.empty_cache()

        ############### Tianjun Added ###############
        # if self.config.residual_mode or not self.config.use_residual_cache:
        if self.config.residual_mode:
            selected_layers = [0, 4, 20, 30]
            activation = [a[selected_layers, ...] for a in activation]
        #############################################

        activation = torch.cat(activation, dim=cat_dim)
        
        print(f"Shape of sentence embeddings: {activation.size()}")
        
        activation = activation.cpu().detach().to(torch.float32).numpy()
        
        return activation
    
    def get_sentence_embeddings_bow(self, sentences):
        """
        Compute sentence representations using a bag-of-words encoding.
        :param sentences: list of sentences
        :return: bag-of-words representation for each sentence
        """
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(sentences).toarray().astype(np.float32)
        # Add a layer dimension so the shape matches other embeddings
        bow_matrix = bow_matrix.reshape(1, -1, bow_matrix.shape[-1])  # (1, num_sentences, vocab_size)
        print(f"Shape of bag-of-words matrix: {bow_matrix.shape}")
        return bow_matrix
    
    def get_speech_embeddings(self, inputs):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.model.dtype):
                if "whisper" in self.config.model_name:
                    outputs = self.model(**inputs, output_hidden_states=True).hidden_states
                    outputs = torch.stack(outputs) # ([7, 1, 1500, 512])
                    if self.config.target_layer is not None:
                        outputs = outputs[self.config.target_layer]  
                    attn_mask = inputs["attention_mask"]  # shape: [batch, seq]
                    if self.config.sent_pooling == 'last':
                        inputs_last_indices = (attn_mask.sum(dim=1) - 1)
                        last_indices = inputs_last_indices/2 
                        last_indices = last_indices.to(dtype=torch.int64)
                        embedding = outputs[:, :, last_indices, :]
                    elif self.config.sent_pooling == 'mean':
                        embedding = torch.mean(outputs, dim=2)
                    else:
                        raise ValueError(f"Invalid pooling method: {self.config.sent_pooling}")
        
                else:
                    outputs = self.model(**inputs, output_hidden_states=True).hidden_states
                    outputs = torch.stack(outputs) # torch.Size([25, 1, 116, 1024]) [L, B, seq, hidden]
                    if self.config.target_layer is not None:
                        outputs = outputs[self.config.target_layer].unsqueeze(0) # add a dimension
                        
                    if self.config.sent_pooling == 'last':
                        embedding = outputs[:, :, -1, :]
                    elif self.config.sent_pooling == 'mean':
                        embedding =torch.mean(outputs, dim=2)
                    else:
                        raise ValueError(f"Invalid pooling method: {self.config.sent_pooling}")
                
                return embedding
    
    def decoding_probing(self, activation, y, comps_ids=None):
        """
        Main probing routine (designed to avoid PCA data leakage):
        1. Use the provided sentence embeddings;
        2. Optionally project with a transfer matrix;
        3. Run cross-validation with Pipeline(PCA→Classifier) on each layer;
        4. Append the results to the output file.
        """
        if self.config.residual_mode:
            if self.config.task == 'comps':
                residuals_path = "/home/lh3288/project/PJ2504_Dual_stream/Data/COMPS_multi-source/"
                task = "base"
                # task = "wugs_dist"

                # Syntax
                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_0_6_avg_0_no_scale.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                syntax_activation = torch.from_numpy(array_2d)
                print(f"Syntax MSE: {(syntax_activation ** 2).mean()}")

                # Meaning
                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_[0, 6]_20_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                meaning_activation = torch.from_numpy(array_2d)
                print(f"Meaning MSE: {(meaning_activation ** 2).mean()}")

                # Reasoning
                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_[0, 6, 20]_30_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                reasoning_activation = torch.from_numpy(array_2d)
                print(f"Reasoning MSE: {(reasoning_activation ** 2).mean()}")

            elif self.config.task == 'blimp':
                residuals_path = "/home/lh3288/project/PJ2504_Dual_stream/Data/BLiMP_scale/"
                print("Yes, using 6!!!")
                task = "BLiMP"
                start, end = (comps_ids - 1) * 2000, comps_ids * 2000

                # Syntax
                with open(residuals_path + f"BLiMP_Qwen2.5-14B_-1_0_6_filter.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                syntax_activation = torch.from_numpy(array_2d)
                print(f"Syntax MSE: {(syntax_activation ** 2).mean()}")

                # Meaning
                with open(residuals_path + f"BLiMP_Qwen2.5-14B_-1_6_20_filter.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                meaning_activation = torch.from_numpy(array_2d)
                print(f"Meaning MSE: {(meaning_activation ** 2).mean()}")

                # Reasoning
                with open(residuals_path + f"BLiMP_Qwen2.5-14B_-1_20_30_filter.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                reasoning_activation = torch.from_numpy(array_2d)
                print(f"Reasoning MSE: {(reasoning_activation ** 2).mean()}")

            # Stack all three on CPU
            activation = torch.stack(
                [syntax_activation, meaning_activation, reasoning_activation],
                dim=0
            )
            print(activation.shape)
        # —— 1. Optional projection —— 
        if self.config.transfer_matrix is not None:
            activation = self.config.transfer_matrix.fc1(activation)

        print(f'\nActivation shape: {activation.shape}')
        layer_num = activation.shape[0]

        # —— 2. Configure cross-validation —— 
        n_splits    = 5
        # n_components = 50    # set to 0 if dimensionality reduction is not desired
        n_components = self.config.pca_dim  # -1 means skip PCA
        cv          = StratifiedKFold(n_splits=n_splits, shuffle=self.config.shuffle, random_state=42 if self.config.shuffle else None)
        scoring     = {
            'accuracy':'accuracy',
            'precision':'precision',
            'recall':'recall',
            'f1':'f1'
        }

        # —— 3. Cross-validate each layer —— 
        for layer_id in range(layer_num):
            print(f'\n▶ Layer {layer_id}')
            X = activation[layer_id]
            # Flatten then convert to numpy
            X = X.reshape(X.shape[0], -1)
            if isinstance(X, torch.Tensor):
                X = X.cpu().detach().numpy()

            if layer_id == 0:
                print(f"  X shape: {X.shape},  y shape: {y.shape}")

            for classifier_name, classifier in self.classifiers:
                # Build the PCA + classifier pipeline
                steps = []
                if n_components > 0:
                    steps.append(('pca', PCA(n_components=n_components)))
                steps.append(('clf', classifier))
                pipe = Pipeline(steps)

                # Ensure the pipeline handles both fit/transform and fit/predict
                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

                # —— 4. Write results to file —— 
                with open(self.config.output_file_name, "a", encoding="utf-8") as f_out:
                    # Add header if the file is empty
                    if os.path.getsize(self.config.output_file_name) == 0:
                        if self.config.task in {"proofwriter", "logic-llm", "prontoqa", "winogrande"}:
                            header = (
                                "gpt_layer,classifier,mode,speaker,"
                                "accuracy,precision,recall,f1,"
                                "acc_fold1,acc_fold2,acc_fold3,acc_fold4,acc_fold5,"
                                "pre_fold1,pre_fold2,pre_fold3,pre_fold4,pre_fold5,"
                                "recall_fol1,recall_fol2,recall_fol3,recall_fol4,recall_fol5,"
                                "f1_fold1,f1_fold2,f1_fold3,f1_fold4,f1_fold5"
                            )
                        else:
                            header = (
                                "lang,field,linguisitics_form,tse_type,"
                                "gpt_layer,classifier,mode,speaker,"
                                "accuracy,precision,recall,f1,"
                                "acc_fold1,acc_fold2,acc_fold3,acc_fold4,acc_fold5,"
                                "pre_fold1,pre_fold2,pre_fold3,pre_fold4,pre_fold5,"
                                "recall_fol1,recall_fol2,recall_fol3,recall_fol4,recall_fol5,"
                                "f1_fold1,f1_fold2,f1_fold3,f1_fold4,f1_fold5"
                            )
                        print(header, file=f_out)

                    # Metadata prefix
                    if self.config.task in {"proofwriter", "logic-llm", "prontoqa", "winogrande"}:
                        prefix = (
                            f"{layer_id},{classifier_name},"
                            f"{self.config.sent_pooling},{self.config.speaker}"
                        )
                    else:
                        prefix = (
                            f"{self.config.file_id},{self.config.field},"
                            f"{self.config.ling_form},{self.config.type_name},"
                            f"{layer_id},{classifier_name},"
                            f"{self.config.sent_pooling},{self.config.speaker}"
                        )
                    # Average metrics
                    avg_scores = ",".join(f"{scores[f'test_{m}'].mean():.4f}" 
                                        for m in scoring)
                    # Per-fold metrics
                    fold_scores = ",".join(
                        ",".join(f"{s:.4f}" for s in scores[f'test_{m}'])
                        for m in scoring
                    )
                    print(f"{prefix},{avg_scores},{fold_scores}", file=f_out)

            print("-" * 60)

    # Clear CUDA memory
        del activation
        torch.cuda.empty_cache()
                
    def decoding_probing_old(self, activation, y):
        """
        Legacy probing pipeline:
        1. Consume precomputed sentence embeddings;
        2. Optionally apply the transfer_matrix projection;
        3. Run cross-validation on each layer and log the metrics.
        """
        # Apply the transfer matrix to the activations if provided
        if self.config.transfer_matrix is not None:
            activation = self.config.transfer_matrix.fc1(activation)

        print(f'\nActivation shape: {activation.shape}')
        layer_num = activation.shape[0]
        n_splits = 5
        n_components = -1  # -1 means PCA is skipped

        for layer_id in range(layer_num):
            print(f'Run cross validation using layer {layer_id}')
            X = activation[layer_id]
            # X shape: (batch_size, hidden_size)
            X = X.reshape(X.shape[0], -1)
            if isinstance(X, torch.Tensor):
                X = X.cpu().detach().numpy()

            if layer_id == 0:
                print(f"X shape: {X.shape}, y shape: {y.shape}")

            # Optional PCA dimensionality reduction
            if n_components > 0 and n_components < X.shape[0]:
                pca = PCA(n_components=n_components)
                X = pca.fit_transform(X)
            elif n_components == -1:
                pass
            else:
                # Dynamically adjust n_components if desired
                n_components = int(X.shape[0] * 0.7)
                pca = PCA(n_components=n_components)
                X = pca.fit_transform(X)

            for classifier_name, classifier in self.classifiers:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=self.config.shuffle, random_state=42 if self.config.shuffle else None)
                scoring = {
                    'accuracy': 'accuracy',
                    'precision': 'precision',
                    'recall': 'recall',
                    'f1': 'f1'
                }

                scores = cross_validate(classifier, X, y, cv=cv, scoring=scoring)

                # Write metrics to the CSV output
                with open(self.config.output_file_name, "a", encoding="utf-8") as f_out:
                    # if it is a new file, write the header
                    if os.path.getsize(self.config.output_file_name) == 0:
                        print("lang,field,linguisitics_form,tse_type,gpt_layer,classifier,mode,speaker,accuracy,precision,recall,f1,acc_fold1,acc_fold2,acc_fold3,acc_fold4,acc_fold5,pre_fold1,pre_fold2,pre_fold3,pre_fold4,pre_fold5,recall_fol1,recall_fol2,recall_fol3,recall_fol4,recall_fol5,f1_fold1,f1_fold2,f1_fold3,f1_fold4,f1_fold5,", file=f_out)
                    print(
                        f"{self.config.file_id},{self.config.field},{self.config.ling_form},{self.config.type_name},"
                        f"{layer_id},{classifier_name},{self.config.sent_pooling},{self.config.speaker}",
                        end=",",
                        file=f_out
                    )
                    for metric in scoring.keys():
                        avg_score = scores[f'test_{metric}'].mean()
                        print(f"{avg_score}", end=",", file=f_out)

                    for metric in scoring.keys():
                        for score in scores[f'test_{metric}']:
                            print(f"{score}", end=",", file=f_out)

                    print("", file=f_out)  # newline

            print("*" * 60)

        # Clear CUDA memory
        del activation
        torch.cuda.empty_cache()
    
    def decoding_probing_gpu(self, activation, y, comps_ids=None):
        ############### Tianjun Changed ###############
        # added `comps_ids` to parameters
        ###############################################
        """
        use cuML to run the logsitic regression on GPU
        """
        # cp.cuda.Device(1).use() # use GPU 1
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
        
        ############### Tianjun Added ###############
        if not self.config.residual_mode:
        #############################################
            # count Nan in the activation
            nan_cnt = torch.isnan(activation).sum().item()
            if nan_cnt > 0:
                print(f'NaN found in activation: {nan_cnt}')
                return
            else:
                print(f'No NaN found in activation.')
            
            inf_cnt = torch.isinf(activation).sum().item()
            if inf_cnt > 0:
                print(f'Inf found in activation: {inf_cnt}')
                return
            else:
                print(f'No Inf found in activation.')

        ############### Tianjun Added ###############
        if self.config.residual_mode:
            # if self.config.task == 'blimp':
            #     layer_0 = activation[0]
            #     layer_4 = activation[1]
            #     layer_20 = activation[2]
            #     layer_30 = activation[3]

            #     ridge_path = "/home/lh3288/project/PJ2504_Dual_stream/Data/pretrained_ridge_models/"

            #     syntax_ridge_model = joblib.load(ridge_path + "BLiMP_0_4.joblib")
            #     syntax_pred = syntax_ridge_model.predict(layer_0.to(dtype=torch.float32).cpu().numpy())
            #     syntax_pred = torch.tensor(syntax_pred, device=layer_4.device, dtype=torch.float32)
            #     syntax_activation = layer_4 - syntax_pred
            #     print(f"Syntax MSE: {(syntax_activation**2).mean().item()}")

            #     meaning_ridge_model = joblib.load(ridge_path + "BLiMP_4_20.joblib")
            #     meaning_pred = meaning_ridge_model.predict(layer_4.to(dtype=torch.float32).cpu().numpy())
            #     meaning_pred = torch.tensor(meaning_pred, device=layer_20.device, dtype=torch.float32)
            #     meaning_activation = layer_20 - meaning_pred
            #     print(f"Meaning MSE: {(meaning_activation**2).mean().item()}")

            #     reasoning_ridge_model = joblib.load(ridge_path + "BLiMP_20_30.joblib")
            #     reasoning_pred = reasoning_ridge_model.predict(layer_20.to(dtype=torch.float32).cpu().numpy())
            #     reasoning_pred = torch.tensor(reasoning_pred, device=layer_30.device, dtype=torch.float32)
            #     reasoning_activation = layer_30 - reasoning_pred
            #     print(f"Reasoning MSE: {(reasoning_activation**2).mean().item()}")

            #     activation = torch.stack([syntax_activation, meaning_activation, reasoning_activation], dim=0)

                
            if self.config.task == 'comps':
                # residuals_path = f"/home/lh3288/project/PJ2504_Dual_stream/Data/COMPS_good_layer/"
                residuals_path = f"/home/lh3288/project/PJ2504_Dual_stream/Data/COMPS_multi-source/"
                # task = "base"
                task = "wugs_dist"

                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_0_6_avg_0_no_scale.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                syntax_activation = torch.from_numpy(array_2d)
                print(f"Syntax MSE: {(syntax_activation**2).mean()}")

                # with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_6_20_multi-source.pkl", "rb") as file:
                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_[0, 6]_20_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                meaning_activation = torch.from_numpy(array_2d)
                print(f"Meaning MSE: {(meaning_activation**2).mean()}")

                # with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_20_30_multi-source.pkl", "rb") as file:
                with open(residuals_path + f"comps_{task}_Qwen2.5-14B_-1_[0, 6, 20]_30_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][comps_ids].values)
                reasoning_activation = torch.from_numpy(array_2d)
                print(f"Reasoning MSE: {(reasoning_activation**2).mean()}")

            if self.config.task == 'blimp':
                residuals_path = f"/home/lh3288/project/PJ2504_Dual_stream/Data/BLiMP_good_task/"
                task = "BLiMP"
                start, end = (comps_ids - 1) * 2000, comps_ids * 2000

                with open(residuals_path + f"{task}_Qwen2.5-14B_-1_0_6_avg_0_no_scale.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                syntax_activation = torch.from_numpy(array_2d)
                print(f"Syntax MSE: {(syntax_activation**2).mean()}")

                # with open("/home/lh3288/project/PJ2504_Dual_stream/Data/BLiMP_linear_weights/" + f"{task}_Qwen2.5-14B_-1_4_20_5100.pkl", "rb") as file:
                # with open(residuals_path + f"{task}_Qwen2.5-14B_-1_6_20_no_scale.pkl", "rb") as file:
                with open(residuals_path + f"{task}_Qwen2.5-14B_-1_[0, 6]_20_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                meaning_activation = torch.from_numpy(array_2d)
                print(f"Meaning MSE: {(meaning_activation**2).mean()}")

                # with open(residuals_path + f"{task}_Qwen2.5-14B_-1_20_30_no_scale.pkl", "rb") as file:
                with open(residuals_path + f"{task}_Qwen2.5-14B_-1_[0, 6, 20]_30_multi-source.pkl", "rb") as file:
                    residual = pickle.load(file)
                array_2d = np.stack(residual["residual"][start:end].values)
                reasoning_activation = torch.from_numpy(array_2d)
                print(f"Reasoning MSE: {(reasoning_activation**2).mean()}")

                # activation = torch.stack([syntax_activation, meaning_activation, reasoning_activation], dim=0) 
            activation = torch.stack([syntax_activation, meaning_activation, reasoning_activation], dim=0) 
            print(activation.shape)
        #############################################

        if self.config.transfer_matrix is not None:
            activation = self.config.transfer_matrix.fc1(activation)
        try:    
            activation = activation.cpu().detach().numpy()  # convert to a NumPy array
        except:
            pass            
        # activation = activation.to(dtype=torch.float32)
        # activation = activation.cpu().detach().numpy()  
        ############### Tianjun Changed ###############
        layer_start = 0
        layer_end = activation.shape[0]
        ###############################################
        activation = activation[layer_start:layer_end]  # run first 10 layers
        # activation = activation[layer_start:12]  # run first 10 layers
        layer_num = activation.shape[0]
        n_splits = 5
        n_components = self.config.pca_dim  # -1 disables PCA

        for layer_id in tqdm(range(layer_num), desc="Processing layers", unit="layer"):
            X = activation[layer_id]
            X = X.reshape(X.shape[0], -1)  # ensure X is a 2D matrix
            
            # print(f"X shape: {X.shape}, y shape: {y.shape}, dtype: {X.dtype}, {y.dtype}")
            
            # if n_components > 0:
            #     pca = cuPCA(n_components=n_components)
            #     X = pca.fit_transform(cp.asarray(X))
            # elif n_components == -1:
            #     pass
            # else:
            #     n_components = int(X.shape[0] * 0.7)
            #     pca = cuPCA(n_components=n_components)
            #     X = pca.fit_transform(cp.asarray(X))

            # X, y = shuffle(X, y, random_state=42)
            ############### Tianjun Added ###############
            if not self.config.save_linear_weights:
            #############################################
                cv = StratifiedKFold(n_splits=n_splits, shuffle=self.config.shuffle, random_state=42 if self.config.shuffle else None)
                for classifier_name, classifier in self.classifiers_gpu:  # custom classifiers
                    scores = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': []
                    }

                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        
                        if n_components > 0:
                            # CUDA version
                            # pca = cuPCA(n_components=n_components)
                            # X_train = pca.fit_transform(cp.asarray(X_train))
                            # X_test = pca.transform(cp.asarray(X_test))
                            # CPU version
                            pca = PCA(n_components=n_components)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)
                            X_train = cp.asarray(X_train)
                            X_test = cp.asarray(X_test)
                        else:
                            X_train = cp.asarray(X_train)
                            X_test = cp.asarray(X_test)

                        # X_train = cp.asarray(X_train)
                        # X_test = cp.asarray(X_test)
                        y_train = cp.asarray(y_train)
                        y_test = cp.asarray(y_test)

                        classifier.fit(X_train, y_train)

                        y_pred = classifier.predict(X_test).get()

                        scores['accuracy'].append(accuracy_score(y_test.get(), y_pred))
                        scores['precision'].append(precision_score(y_test.get(), y_pred))
                        scores['recall'].append(recall_score(y_test.get(), y_pred))
                        scores['f1'].append(f1_score(y_test.get(), y_pred))
                        
                        del X_train, X_test, y_train, y_test, y_pred
                        cp._default_memory_pool.free_all_blocks()
                        cp.cuda.Device().synchronize()

                    output_layer = layer_id + layer_start if self.config.target_layer is None else self.config.target_layer
                    with open(self.config.output_file_name, "a", encoding="utf-8") as f_out:
                        # if it is a new file, write the header
                        if os.path.getsize(self.config.output_file_name) == 0:
                            print("lang,tse_id,field,linguisitics_form,tse_type,gpt_layer,classifier,mode,speaker,accuracy,precision,recall,f1,acc_fold1,acc_fold2,acc_fold3,acc_fold4,acc_fold5,pre_fold1,pre_fold2,pre_fold3,pre_fold4,pre_fold5,recall_fol1,recall_fol2,recall_fol3,recall_fol4,recall_fol5,f1_fold1,f1_fold2,f1_fold3,f1_fold4,f1_fold5,", file=f_out)
                        print(
                            f"{self.config.lang},{self.config.file_id},{self.config.field},{self.config.ling_form},{self.config.type_name},"
                            f"{output_layer},{classifier_name},{self.config.sent_pooling},{self.config.speaker}",
                            end=",",
                            file=f_out
                        )
                        for metric, values in scores.items():
                            avg_score = np.mean(values)
                            print(f"{avg_score}", end=",", file=f_out)

                        for metric, values in scores.items():
                            for score in values:
                                print(f"{score}", end=",", file=f_out)

                        print("", file=f_out)  # newline

            ############### Tianjun Added ###############
            else:
                for classifier_name, classifier in self.classifiers_gpu:
                    if n_components > 0:
                        pca = PCA(n_components=n_components)
                        X = pca.fit_transform(X)
                        joblib.dump(pca.components_, f"/home/lh3288/project/PJ2504_Dual_stream/Data/linear_weights/pca_syntax_{comps_ids}_{classifier_name}_{layer_id}")
                    
                    X = cp.asarray(X)
                    y = cp.asarray(y)
                    classifier.fit(X, y)
                    df = {
                        "X": X,
                        "y": y,
                        "classifier": classifier
                    }
                    joblib.dump(df, f"/home/lh3288/project/PJ2504_Dual_stream/Data/linear_models/syntax_{comps_ids}_{classifier_name}_{layer_id}")
                    # joblib.dump(cp.asnumpy(classifier.coef_), f"/home/lh3288/project/PJ2504_Dual_stream/Data/linear_weights/linear_syntax_{comps_ids}_{classifier_name}_{layer_id}")
            #############################################

        print("Task finished.")

        # empty cuML memory
        cp._default_memory_pool.free_all_blocks()
        cp.cuda.Device().synchronize()
        
        return

    def run_blimp(self, data_path):
        """
        Run the BLiMP dataset probing tasks.
        """
        jsonl_files = self._find_files(data_path, endswith='.jsonl')
        # print(jsonl_files)
        task_num = len(jsonl_files)
        print("Number of BLiMP files:", task_num)

        ############### Tianjun Added ###############
        blimp_id = 0
        ###############################################
        for file_id in range(task_num):
            ############### Tianjun Added ###############
            if self.config.residual_mode:
                blimp_id += 1
            ###############################################

            sentences, labels = [], []
            file_path = jsonl_files[file_id]
            df = pd.read_json(file_path, lines=True)
            bow_results = pd.read_csv('./results/blimp/blimp_base_bow_-1_bow.csv')
            type_name = df["UID"][0]
            if self.config.filter_mode:
                bow_f1 = bow_results[bow_results['tse_type'] == type_name]['f1'].values[0]
                if bow_f1 > 0.6:
                    print(f"Skipping {type_name} with bow f1: {bow_f1}")
                    if self.config.blimp_residual_filter:
                        blimp_id -= 1
                    continue
                else:
                    print(f"Processing {type_name} with bow f1: {bow_f1}")
                
            print(f"Processing field: {df.field[0]}, "
                  f"linguistic term: {df.linguistics_term[0]}, "
                  f"task: {df.UID[0]}")

            # df['sentence_good'] = df['sentence_good'].apply(self._remove_end_punctuation)
            # df['sentence_bad'] = df['sentence_bad'].apply(self._remove_end_punctuation)

            sentences.extend(df['sentence_good'].tolist() + df['sentence_bad'].tolist())
            labels.extend([1] * len(df) + [0] * len(df))
            
            # for good, bad in zip(df['sentence_good'], df['sentence_bad']):
                # sentences.extend([good, bad])
                # labels.extend([1, 0])
            
            labels = np.array(labels)
            print('Total sentences:', len(sentences))

            self.config.update({
                'file_id': -1,
                'type_name': type_name,
                'mode': 'last',
                'field': df.field[0],
                'ling_form': df.linguistics_term[0]
            })
            # encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.config.device)
            # print(f"Encoded inputs shape: {encoded_inputs['input_ids'].shape}")
            ############### Tianjun Changed ###############
            if self.config.residual_mode:
                activation = None
            elif self.config.bow_mode:
                activation = self.get_sentence_embeddings_bow(sentences)
            else:
                activation = self.get_sentence_embeddings(sentences)
            ###############################################
            
            if self.config.save_to_mat:
                # turn activation to float32
                if isinstance(activation, torch.Tensor):
                    activation = activation.cpu().detach().to(torch.float32).numpy()
                elif isinstance(activation, np.ndarray):
                    activation = activation.astype(np.float32)
                # save type_name, field, ling_form, sentences, labels, activation to a .mat file
                data_to_mat = []
                data_to_mat.append({
                    'field': df.field[0],
                    'phenomenon': df.linguistics_term[0],
                    'task_name': type_name,
                    'sentences_good': df['sentence_good'].tolist(),
                    'sentences_bad': df['sentence_bad'].tolist(),
                    # 'labels': labels,
                    'activation': activation
                })
                data_to_mat = pd.DataFrame(data_to_mat)
                data_to_mat = utils.df_to_mat(data_to_mat)
                sio.savemat(self.config.embed_path.replace('.mat', f'_{type_name}.mat'), {f'{type_name}': data_to_mat})
                continue # skip the decoding step if saving to mat

            if self.config.gpu_classify:
                self.decoding_probing_gpu(activation, labels, comps_ids=blimp_id)
            else:
                self.decoding_probing(activation, labels, comps_ids=blimp_id)
                
            del activation
            gc.collect()
            torch.cuda.empty_cache()
            
        # if self.config.save_to_mat:
        #     data_to_mat = pd.DataFrame(data_to_mat)
        #     data_to_mat = utils.df_to_mat(data_to_mat)
            # sio.savemat(self.config.embed_path, {'data': data_to_mat})
            # with h5py.File(self.config.embed_path.replace('.mat', '.h5'), 'w') as f:
                # f.create_dataset('data', data=data_to_mat)
        return

    def run_blimp_single_task(self, data_path, file_id):
        """
        Run a single BLiMP probing task specified by file index.
        """
        jsonl_files = self._find_files(data_path, endswith='.jsonl')
        # print(jsonl_files)
        task_num = len(jsonl_files)
        print("Number of BLiMP files:", task_num)
        # sort the jsonl files
        jsonl_files.sort()
        
        sentences, labels = [], []
        file_path = jsonl_files[file_id]
        df = pd.read_json(file_path, lines=True)
        type_name = df["UID"][0]
        print(f"Processing field: {df.field[0]}, "
                f"linguistic term: {df.linguistics_term[0]}, "
                f"task: {df.UID[0]}")

        df['sentence_good'] = df['sentence_good'].apply(self._remove_end_punctuation)
        df['sentence_bad'] = df['sentence_bad'].apply(self._remove_end_punctuation)

        sentences.extend(df['sentence_good'].tolist() + df['sentence_bad'].tolist())
        labels.extend([1] * len(df) + [0] * len(df))
        labels = np.array(labels)
        print('Total sentences:', len(sentences))

        self.config.update({
            'file_id': -1,
            'type_name': type_name,
            'mode': 'last',
            'field': df.field[0],
            'ling_form': df.linguistics_term[0]
        })
        # encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.config.device)
        # print(f"Encoded inputs shape: {encoded_inputs['input_ids'].shape}")
        activation = self.get_sentence_embeddings(sentences)
        # self.decoding_probing(activation, labels)
        
        if self.config.gpu_classify:
            self.decoding_probing_gpu(activation, labels)
        else:
            self.decoding_probing(activation, labels)
            
        del activation
        # gc.collect()
        torch.cuda.empty_cache()
        return
    
    def run_climp(self, data_path):
        """
        Run the CLiMP dataset probing workflow.
        """
        csv_files = self._find_files(data_path, endswith='.csv')
        task_num = len(csv_files)
        print("Number of CLiMP files:", task_num)

    # Update self.config fields as needed per task
        for file_id in range(task_num):
            file_path = csv_files[file_id]
            df = pd.read_csv(file_path)

            # Read the phenomenon label
            phenomenon = df['Phenomenon'][0]
            print(f'Processing {file_id}-th task: {phenomenon}')

            df['Sentence'] = df['Sentence'].apply(self._remove_end_punctuation)
            good_sent = df[df['Grammatical'] == 1]['Sentence'].tolist()
            bad_sent = df[df['Grammatical'] == 0]['Sentence'].tolist()
            sentences = good_sent + bad_sent
            labels = np.hstack([
                np.ones(len(good_sent)), 
                np.zeros(len(bad_sent))
            ])

            print('Sentence number:', len(sentences))

            self.config.update({
                'file_id': file_id,
                'type_name': phenomenon,
                'mode': 'last',
                'field': 'syntax',
                # If the CSV has a Category column, prefer df['Category'][0]
                'ling_form': df['Category'][0] if 'Category' in df.columns else 'Unknown'
            })

            self.decoding_probing(sentences, labels)

    def run_comps(self, file_name):
        """
        conceptual minimal pair probing task
        """
        # df = pd.read_json(file_name, lines=True)
        if self.config.lang != 'en':
            df = pd.read_json(file_name, orient="index")
        elif self.config.task == 'comps':
            df = pd.read_json(file_name, lines=True)
        elif self.config.task == 'comps_wugs':
            df_before = pd.read_json(file_name+'comps_wugs_dist-before.jsonl', lines=True)
            df_between = pd.read_json(file_name+'comps_wugs_dist-in-between.jsonl', lines=True)
        else:
            raise ValueError(f"Task {self.config.task} is not supported. Please check the task name.")
        
        tasks = ['taxonomic', 'overlap', 'co-occurrence', 'random']
        # tasks = ['co-occurrence']
        
        for task in tasks:
            print(f'task: {self.config.task}| lang: {self.config.lang} | level: {task}')
            if self.config.embed_path is not None:
                activation_path = self.config.embed_path + f"/{self.config.model_name}/{self.config.lang}_{task}.pickle"
            else:
                activation_path = None
                
            if activation_path is not None and os.path.exists(activation_path):
                print(f'Loading {task} embeddings from {activation_path}')
                activation, y = pickle.load(open(activation_path, 'rb'))
            else:
                if self.config.task == 'comps':
                    ############### Tianjun Added ###############
                    comps_ids = (df['negative_sample_type'] == task)
                    comps_ids = pd.Series(
                        list(comps_ids.values) * 2,  # duplicate values
                        index=range(len(comps_ids) * 2)  # reset index to avoid alignment issues
                    )
                    #############################################

                    df_task = df[df['negative_sample_type'] == task]
                    sentences_task_good = df_task['prefix_acceptable'] + ' ' + df_task['property_phrase']
                    sentences_task_bad = df_task['prefix_unacceptable'] + ' ' + df_task['property_phrase']
                elif self.config.task == 'mcomps':
                    df_task = df[df['negative_sample_type'] == task]
                    sentences_task_good = df_task['acceptable_sent']
                    sentences_task_bad = df_task['unacceptable_sent']
                elif self.config.task == 'comps_wugs':
                    df_before_task = df_before[df_before['negative_sample_type'] == task]
                    df_between_task = df_between[df_between['negative_sample_type'] == task]
                    sentences_task_good = df_before_task['prefix_acceptable'] + ' ' + df_before_task['property_phrase']
                    sentences_task_good += df_between_task['prefix_acceptable'] + ' ' + df_between_task['property_phrase']
                    sentences_task_bad = df_before_task['prefix_unacceptable'] + ' ' + df_before_task['property_phrase']
                    sentences_task_bad += df_between_task['prefix_unacceptable'] + ' ' + df_between_task['property_phrase']
                else:   
                    raise ValueError(f"Task {self.config.task} is not supported. Please check the task name.")
                
                print(f'Processing {task} task, number of good sentences: {len(sentences_task_good)}, number of bad sentences: {len(sentences_task_bad)}')
                sentences_task_good = sentences_task_good.tolist()
                sentences_task_bad = sentences_task_bad.tolist()
                sentences = sentences_task_good + sentences_task_bad             
                # sentences = sentences_task_good.tolist() + sentences_task_bad.tolist()
                
                sentences = [i[:-1] for i in sentences] # remove the period at the end of the sentence
                
                y = np.concatenate((np.ones(len(sentences_task_good)), np.zeros(len(sentences_task_bad))))
                print(f'Number of sentences: {len(sentences)}')
                print(f'sentence example: {sentences[0]}')
                # set autocast
                ############### Tianjun Changed ###############
                if self.config.residual_mode:
                    activation = None
                elif self.config.bow_mode:
                    activation = self.get_sentence_embeddings_bow(sentences)
                else:
                    activation = self.get_sentence_embeddings(sentences)
                    
                if activation_path is not None:
                    if not os.path.exists(self.config.embed_path + f"/{self.config.model_name}/"):
                        os.makedirs(self.config.embed_path + f"/{self.config.model_name}/")
                
                    with open(activation_path, 'wb') as f:
                        pickle.dump([activation, y], f) #
                ###############################################
            
            self.config.update({
                    'file_id': 0,
                    'type_name': task,
                    'mode': 'last',
                    'field': 'concept',
                    'ling_form': task
                })
                           
            if self.config.save_to_mat:
                if isinstance(activation, torch.Tensor):
                    activation = activation.cpu().detach().to(torch.float32).numpy()
                elif isinstance(activation, np.ndarray):
                    activation = activation.astype(np.float32)

                data_to_mat = []
                data_to_mat.append({
                    'field': 'concept',
                    'phenomenon': task,
                    'task_name': task,
                    'sentences_good': sentences_task_good,
                    'sentences_bad': sentences_task_bad,
                    'activation': activation})
                
                data_to_mat = pd.DataFrame(data_to_mat)
                data_to_mat = utils.df_to_mat(data_to_mat)
                safe_task_name = task.replace('-', '_')
                sio.savemat(self.config.mat_path.replace('.mat', f'_{safe_task_name}.mat'), {f'{safe_task_name}': data_to_mat})
                continue
             
            if self.config.gpu_classify:
                self.decoding_probing_gpu(activation, y, comps_ids=comps_ids)
            else:
                self.decoding_probing(activation, y, comps_ids=comps_ids)
            
            del activation
            torch.cuda.empty_cache()
        return
    
    def run_proofwriter(self, file_name):
        df = pd.read_json(file_name, lines=True)
        sentences = (df['context'] + ' ' + df['question']).tolist()

        # Prepare binary labels: 1 if answer is True, else 0
        y = np.array([
            1 if ans else 0
            for ans in df['answer']
        ])

        # Extract activations (skip if in residual mode)
        if self.config.residual_mode:
            activation = None
        else:
            with torch.cuda.amp.autocast():
                activation = self.get_sentence_embeddings(sentences)

        # Run the probing
        if getattr(self.config, 'gpu_classify', False):
            self.decoding_probing_gpu(activation, y)
        else:
            self.decoding_probing(activation, y)

        del activation
        torch.cuda.empty_cache()

    def run_prontoqa(self, file_name):
        df = pd.read_json(file_name, lines=True)
        sentences = df['question'].tolist()

        # Prepare binary labels: 1 if answer is True, else 0
        y = np.array([
            1 if ans else 0
            for ans in df['answer']
        ])

        # Extract activations (skip if in residual mode)
        if self.config.residual_mode:
            activation = None
        elif self.config.bow_mode:
            activation = self.get_sentence_embeddings_bow(sentences)
        else:
            activation = self.get_sentence_embeddings(sentences)

        # Run the probing
        if getattr(self.config, 'gpu_classify', False):
            self.decoding_probing_gpu(activation, y)
        else:
            self.decoding_probing(activation, y)

        del activation
        torch.cuda.empty_cache()

    def run_winogrande(self, file_name):
        self.run_prontoqa(file_name)
    
    def neuroling_probing_speech(self, compute_embed_only=False):
        if self.config.task == 'blimp_speech':
            labels = pd.read_csv(self.config.task_label_path)
        elif self.config.task == 'comps_speech':
            labels = pd.DataFrame({'Field':['concept', 'concept', 'concept', 'concept'],
                                   'Ling_form':['taxonomic', 'overlap', 'co-occurrence', 'random'],
                                   'Task':['taxonomic', 'overlap', 'co-occurrence', 'random'],
                                   'Task_id':[0, 1, 2, 3]})
        else:
            raise ValueError(f"Task {self.config.task} is not supported. Please check the task name.")
            
        if self.config.test_mode:
            num_pair = 10
        elif self.config.task == 'comps_speech':
            num_pair = 49340
        elif self.config.task == 'blimp_speech':
            num_pair = 1000
        else:
            pass    
        # num_pair = 10 if self.config.test_mode else 1000
        # for index, row in tqdm(labels.iterrows(), desc="Processing rows", total=len(labels)):
        for index, row in labels.iterrows():
            field = row.Field
            ling_form = row.Ling_form
            type_name = row.Task
            
            print(f'Processing {index}-th task, field: {field} | ling_form: {ling_form} | type_name: {type_name}')
            
            # check if pickle file exists, if yes load them directly without processing
            activation_save_file = self.config.speech_data_path + f"/embed/{self.config.model_name}/{self.config.speaker}_{type_name}.pickle"
            
            if os.path.exists(activation_save_file):
                # delete the pickle file if you want to recompute the embeddings
                # os.remove(activation_save_file)
                # print('deleted')
                if compute_embed_only:
                    print(f'{type_name} embedding computed already, skip computing.')
                    continue
                with open(activation_save_file, 'rb') as f:
                    final_activations, final_labels = pickle.load(f)
                print(f'Embedding loaded. Shape: {final_activations.shape}, Labels: {final_labels.shape}')
            
            else:
                audio_paths = []
                audio_labels = []
                
                # for pair_id in tqdm(range(1000), desc=f"Collecting paths for {type_name}", leave=False):
                for pair_id in range(num_pair):   
                    if self.config.task == 'comps_speech':
                        base_dir = os.path.join(
                        self.config.speech_data_path,
                        "KokoroTTS", 
                        self.config.speaker, 
                        'comps-'+type_name, 
                        f"{pair_id}"
                    )
                    else:              
                        base_dir = os.path.join(
                            self.config.speech_data_path,
                            "KokoroTTS", 
                            self.config.speaker, 
                            type_name, 
                            f"{pair_id}"
                        )
                        
                    if os.path.exists(good_path := os.path.join(base_dir, "good.wav")):
                        audio_paths.append(good_path)
                        audio_labels.append(1)
                        
                    if os.path.exists(bad_path := os.path.join(base_dir, "bad.wav")):
                        audio_paths.append(bad_path)
                        audio_labels.append(0)

                batch_size = self.config.batch_size  # adjust based on GPU memory
                all_activations = []
                
                for batch_idx in tqdm(range(0, len(audio_paths), batch_size), desc=f"Processing batches for {type_name}"):
                # for batch_idx in range(0, len(audio_paths), batch_size):
                    batch_paths = audio_paths[batch_idx:batch_idx+batch_size]
                    
                    waveforms = []
                    sample_rates = set()
                    
                    for path in batch_paths:
                        waveform, sr = torchaudio.load(path)
                        waveforms.append(waveform.squeeze(0).numpy())  # [1, T] -> [T]
                        sample_rates.add(sr)
                        
                    assert len(sample_rates) == 1, "Multiple sample rates found"
                    target_sr = sample_rates.pop()
                    
                    # print(len(waveforms), target_sr)
                    if "s2t" in self.config.model_name:
                        encoded_inputs = self.processor(
                            waveforms,
                            sampling_rate=target_sr,
                            return_tensors="pt",
                            padding="longest",
                            return_attention_mask=True, 
                        ).to(self.model.dtype).to(self.config.device)
                    else:
                        encoded_inputs = self.processor(
                            waveforms,
                            sampling_rate=target_sr,
                            return_tensors="pt",
                            padding="max_length" if "whisper" in self.config.model_name else True,
                            return_attention_mask=True, 
                        ).to(self.model.dtype).to(self.config.device)
                    
                    # temporary debug
                    # import pdb
                    # pdb.set_trace()
                    
                    batch_activations = self.get_speech_embeddings(encoded_inputs)
                    all_activations.append(batch_activations.cpu().detach().numpy())
                    
                    del encoded_inputs, batch_activations
                    torch.cuda.empty_cache()

                # final_activations = torch.cat(all_activations, dim=1)
                final_activations = np.concatenate(all_activations, axis=1)
                final_labels = np.array(audio_labels)
                del all_activations
                torch.cuda.empty_cache()
            
                print(f'Embedding Finished. Shape: {final_activations.shape}, Labels: {final_labels.shape}')
                if not os.path.exists(self.config.speech_data_path + f"/embed/{self.config.model_name}/"):
                    os.makedirs(self.config.speech_data_path + f"/embed/{self.config.model_name}/")                
                with open(activation_save_file, 'wb') as f:
                    pickle.dump([final_activations, final_labels], f)
                    
                
            self.config.update({
                'file_id': row.Task_id,
                'field': field,
                'ling_form': ling_form,
                'type_name': type_name
            })
            
            if not compute_embed_only: # run the probing task when not loading embeddings only
                if self.config.gpu_classify:
                    self.decoding_probing_gpu(final_activations, final_labels)
                else:            
                    self.decoding_probing(final_activations, final_labels)
                            
            del final_activations
            torch.cuda.empty_cache()
        return    

    def run_neuroling_probing(self, compute_embed_only=False):
        if 'speech' in self.config.task:
            #  temporarily use neuroling_probing_ASR for speech probing task
            # self.neuroling_probing_ASR(compute_embed_only=compute_embed_only) 
            self.neuroling_probing_speech(compute_embed_only=compute_embed_only)
        return
        