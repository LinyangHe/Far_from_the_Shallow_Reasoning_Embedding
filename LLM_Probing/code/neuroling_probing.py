import os
import string
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import gc
import joblib
import tempfile


from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer



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
        encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.config.device)
        
        self.model.eval()
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
                outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=self.config.output_hidden_states
                )

            if self.config.sent_pooling == 'last':
                if self.config.target_layer is None:
                    cat_dim = 1
                    embedding_batch = torch.stack(outputs.hidden_states)
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

        activation = torch.cat(activation, dim=cat_dim)
        
        print(f"Shape of sentence embeddings: {activation.size()}")
        
        # Keep as torch tensor to avoid a single large numpy allocation.
        activation = activation.cpu().detach().to(torch.float32)

        return activation

    def get_sentence_embeddings_memmap(self, sentences):
        """
        Stream embeddings into per-layer memmap files to avoid large in-memory tensors.
        """
        encoded_inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.config.device)

        self.model.eval()
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        num_sent = len(input_ids)
        batch_size = self.config.batch_size
        num_batches = num_sent // batch_size + int(num_sent % batch_size != 0)

        memmap_dir = self.config.memmap_dir
        cleanup = False
        if not memmap_dir:
            memmap_dir = tempfile.mkdtemp(prefix="probing_memmap_")
            cleanup = True
        os.makedirs(memmap_dir, exist_ok=True)

        memmap_files = []
        memmaps = []
        num_layers = None
        hidden_size = None

        print(f'Number of batches: {num_batches}, Number of sentences: {num_sent}')
        for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_sent)
            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]

            with torch.no_grad():
                outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=self.config.output_hidden_states
                )

            if self.config.sent_pooling == 'last':
                if self.config.target_layer is None:
                    embedding_batch = torch.stack(outputs.hidden_states)
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
                    embedding_batch = last_word_embeddings
                else:
                    embedding_batch = outputs.hidden_states[self.config.target_layer]
                    lengths = batch_attention_mask.sum(dim=1)
                    last_word_embeddings = embedding_batch[torch.arange(len(lengths)), lengths - 1]
                    embedding_batch = last_word_embeddings.unsqueeze(0)
            elif self.config.sent_pooling == 'mean':
                if self.config.target_layer is None:
                    embedding_batch = torch.stack(outputs.hidden_states)
                    expanded_mask = batch_attention_mask.unsqueeze(0).unsqueeze(-1)
                    expanded_mask = expanded_mask.expand_as(embedding_batch)
                    masked_embeddings = embedding_batch * expanded_mask
                    sentence_lengths = batch_attention_mask.sum(dim=1, keepdim=True).unsqueeze(0)
                    mean_embeddings = masked_embeddings.sum(dim=2) / sentence_lengths
                    embedding_batch = mean_embeddings
                else:
                    embedding_batch = outputs.hidden_states[self.config.target_layer]
                    expanded_mask = batch_attention_mask.unsqueeze(-1)
                    masked_embeddings = embedding_batch * expanded_mask
                    sentence_lengths = batch_attention_mask.sum(dim=1, keepdim=True)
                    mean_embeddings = masked_embeddings.sum(dim=1) / sentence_lengths
                    embedding_batch = mean_embeddings.unsqueeze(0)
            else:
                raise ValueError(f"Invalid pooling method: {self.config.sent_pooling}")

            if num_layers is None:
                num_layers = embedding_batch.size(0)
                hidden_size = embedding_batch.size(-1)
                for layer_id in range(num_layers):
                    file_path = os.path.join(memmap_dir, f"layer_{layer_id}.dat")
                    memmap_files.append(file_path)
                    memmaps.append(
                        np.memmap(file_path, dtype=np.float32, mode='w+', shape=(num_sent, hidden_size))
                    )

            embedding_batch = embedding_batch.detach().cpu().to(torch.float32).numpy()
            for layer_id in range(num_layers):
                memmaps[layer_id][start_idx:end_idx, :] = embedding_batch[layer_id]

            del outputs
            torch.cuda.empty_cache()

        for mm in memmaps:
            mm.flush()

        return {
            "memmap_files": memmap_files,
            "num_layers": num_layers,
            "num_sent": num_sent,
            "hidden_size": hidden_size,
            "cleanup": cleanup,
            "memmap_dir": memmap_dir,
        }
    
    def get_sentence_embeddings_bow(self, sentences):
        """
        Compute sentence representations using a bag-of-words encoding.
        :param sentences: list of sentences
        :return: bag-of-words representation for each sentence
        """
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(sentences).toarray().astype(np.float32)
        bow_matrix = bow_matrix.reshape(1, -1, bow_matrix.shape[-1])  # (1, num_sentences, vocab_size)
        print(f"Shape of bag-of-words matrix: {bow_matrix.shape}")
        return bow_matrix

    def _load_residuals(self, prefix, selection):
        if not prefix:
            raise ValueError("Residual prefix must be set when residual_mode is enabled.")
        layers = self.config.residuals_source_target_layers

        residual_arrays = []
        for source_layer, target_layer in layers:
            file_path = f"{prefix}_{source_layer}_{target_layer}.pkl"
            with open(file_path, "rb") as file:
                residual_df = pickle.load(file)

            if isinstance(selection, slice):
                selected = residual_df["residual"].iloc[selection]
            else:
                mask = np.asarray(selection)
                selected = residual_df["residual"].iloc[mask]

            residual_arrays.append(np.stack(selected.values))

        return np.stack(residual_arrays, axis=0)
    
    def decoding_probing(self, activation, y, comps_ids=None):
        """
        Main probing routine (designed to avoid PCA data leakage):
        1. Use the provided sentence embeddings;
        2. Optionally project with a transfer matrix;
        3. Run cross-validation with Pipeline(PCA→Classifier) on each layer;
        4. Append the results to the output file.
        """
        if self.config.residual_mode:
            if self.config.task == "blimp":
                if comps_ids is None:
                    raise ValueError("blimp_id is required for residual_mode in BLiMP.")
                selection = slice((comps_ids - 1) * 2000, comps_ids * 2000)
                activation = self._load_residuals(self.config.residuals_blimp_prefix, selection)
            elif self.config.task == "comps":
                if comps_ids is None:
                    raise ValueError("comps_ids is required for residual_mode in COMPS.")
                activation = self._load_residuals(self.config.residuals_comps_prefix, comps_ids)
            else:
                raise ValueError(f"Task {self.config.task} does not support residual_mode.")
        elif activation is None:
            raise ValueError("Activation is required when residual_mode is disabled.")

        if isinstance(activation, dict) and "memmap_files" in activation:
            layer_num = activation["num_layers"]
        else:
            print(f'\nActivation shape: {activation.shape}')
            layer_num = activation.shape[0]

        n_splits    = 5
        n_components = self.config.pca_dim  # -1 means skip PCA
        cv          = StratifiedKFold(n_splits=n_splits, shuffle=self.config.shuffle, random_state=42 if self.config.shuffle else None)
        scoring     = {
            'accuracy':'accuracy',
            'precision':'precision',
            'recall':'recall',
            'f1':'f1'
        }

        for layer_id in range(layer_num):
            print(f'\n▶ Layer {layer_id}')
            if isinstance(activation, dict) and "memmap_files" in activation:
                memmap_file = activation["memmap_files"][layer_id]
                X = np.memmap(
                    memmap_file,
                    dtype=np.float32,
                    mode='r',
                    shape=(activation["num_sent"], activation["hidden_size"])
                )
                X = X.reshape(X.shape[0], -1)
            else:
                X = activation[layer_id]
                X = X.reshape(X.shape[0], -1)
                if isinstance(X, torch.Tensor):
                    X = X.cpu().detach().numpy()

            if layer_id == 0:
                print(f"  X shape: {X.shape},  y shape: {y.shape}")

            for classifier_name, classifier in self.classifiers:
                steps = []
                if n_components > 0:
                    steps.append(('pca', PCA(n_components=n_components)))
                steps.append(('clf', classifier))
                pipe = Pipeline(steps)

                scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)

                with open(self.config.output_file_name, "a", encoding="utf-8") as f_out:
                    if os.path.getsize(self.config.output_file_name) == 0:
                        header = (
                            "tse_type,gpt_layer,classifier,mode,"
                            "accuracy,precision,recall,f1,"
                            "acc_fold1,acc_fold2,acc_fold3,acc_fold4,acc_fold5,"
                            "pre_fold1,pre_fold2,pre_fold3,pre_fold4,pre_fold5,"
                            "recall_fol1,recall_fol2,recall_fol3,recall_fol4,recall_fol5,"
                            "f1_fold1,f1_fold2,f1_fold3,f1_fold4,f1_fold5"
                        )
                        print(header, file=f_out)

                    prefix = (
                        f"{self.config.type_name},{layer_id},{classifier_name},"
                        f"{self.config.sent_pooling}"
                    )
                    avg_scores = ",".join(f"{scores[f'test_{m}'].mean():.4f}" 
                                        for m in scoring)
                    fold_scores = ",".join(
                        ",".join(f"{s:.4f}" for s in scores[f'test_{m}'])
                        for m in scoring
                    )
                    print(f"{prefix},{avg_scores},{fold_scores}", file=f_out)

            print("-" * 60)
        if isinstance(activation, dict) and activation.get("cleanup"):
            for memmap_file in activation["memmap_files"]:
                try:
                    os.remove(memmap_file)
                except FileNotFoundError:
                    pass
            try:
                os.rmdir(activation["memmap_dir"])
            except OSError:
                pass
        del activation
        torch.cuda.empty_cache()
    
    def run_blimp(self, data_path):
        """
        Run the BLiMP dataset probing tasks.
        """
        jsonl_files = self._find_files(data_path, endswith='.jsonl')
        task_num = len(jsonl_files)
        print("Number of BLiMP files:", task_num)

        blimp_id = 0
        bow_results = None
        if self.config.filter_mode:
            bow_results = pd.read_csv(self.config.blimp_bow_results_path)

        for file_id in range(task_num):
            if self.config.residual_mode:
                blimp_id += 1
            sentences, labels = [], []
            file_path = jsonl_files[file_id]
            df = pd.read_json(file_path, lines=True)
            type_name = df["UID"][0]
            if self.config.filter_mode:
                bow_f1 = bow_results[bow_results['tse_type'] == type_name]['f1'].values[0]
                if bow_f1 > 0.6:
                    print(f"Skipping {type_name} with bow f1: {bow_f1}")
                    if self.config.residual_mode:
                        blimp_id -= 1
                    continue
                else:
                    print(f"Processing {type_name} with bow f1: {bow_f1}")
                
            print(f"Processing field: {df.field[0]}, "
                  f"linguistic term: {df.linguistics_term[0]}, "
                  f"task: {df.UID[0]}")

            sentences.extend(df['sentence_good'].tolist() + df['sentence_bad'].tolist())
            labels.extend([1] * len(df) + [0] * len(df))
            labels = np.array(labels)
            print('Total sentences:', len(sentences))

            self.config.type_name = type_name
            if self.config.residual_mode:
                activation = None
            elif self.config.bow_mode:
                activation = self.get_sentence_embeddings_bow(sentences)
            elif self.config.use_memmap:
                activation = self.get_sentence_embeddings_memmap(sentences)
            else:
                activation = self.get_sentence_embeddings(sentences)
            
            if self.config.residual_mode:
                self.decoding_probing(activation, labels, comps_ids=blimp_id)
            else:
                self.decoding_probing(activation, labels)
                
            del activation
            gc.collect()
            torch.cuda.empty_cache()
            
        return

    def run_comps(self, file_name):
        """
        conceptual minimal pair probing task
        """
        if self.config.task in {"comps_base", "comps_wugs"}:
            file_map = {
                "comps_base": "comps_base.jsonl",
                "comps_wugs": "comps_wugs_dist.jsonl"
            }
            file_suffix = file_map[self.config.task]
            if file_name and file_name.endswith(".jsonl"):
                comps_path = file_name
            else:
                comps_path = os.path.join(file_name, file_suffix)
            df = pd.read_json(comps_path, lines=True)
        elif self.config.task == 'comps':
            df = pd.read_json(file_name, lines=True)
        else:
            raise ValueError(f"Task {self.config.task} is not supported. Please check the task name.")
        
        tasks = ['taxonomic', 'overlap', 'co-occurrence', 'random']
        
        for task in tasks:
            print(f'task: {self.config.task} | level: {task}')
            comps_ids = None
            if self.config.task in {"comps", "comps_base", "comps_wugs"}:
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
            else:   
                raise ValueError(f"Task {self.config.task} is not supported. Please check the task name.")
            
            print(f'Processing {task} task, number of good sentences: {len(sentences_task_good)}, number of bad sentences: {len(sentences_task_bad)}')
            sentences_task_good = sentences_task_good.tolist()
            sentences_task_bad = sentences_task_bad.tolist()
            sentences = sentences_task_good + sentences_task_bad             
            sentences = [i[:-1] for i in sentences] # remove the period at the end of the sentence
            
            y = np.concatenate((np.ones(len(sentences_task_good)), np.zeros(len(sentences_task_bad))))
            print(f'Number of sentences: {len(sentences)}')
            print(f'sentence example: {sentences[0]}')
            if self.config.residual_mode:
                activation = None
            elif self.config.bow_mode:
                activation = self.get_sentence_embeddings_bow(sentences)
            elif self.config.use_memmap:
                activation = self.get_sentence_embeddings_memmap(sentences)
            else:
                activation = self.get_sentence_embeddings(sentences)

            self.config.type_name = task
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

        with torch.cuda.amp.autocast():
            activation = self.get_sentence_embeddings(sentences)

        # Run the probing
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

        if self.config.bow_mode:
            activation = self.get_sentence_embeddings_bow(sentences)
        else:
            activation = self.get_sentence_embeddings(sentences)

        # Run the probing
        self.decoding_probing(activation, y)

        del activation
        torch.cuda.empty_cache()

    def run_winogrande(self, file_name):
        self.run_prontoqa(file_name)
    
    def run_neuroling_probing(self, compute_embed_only=False):
        return
        