import os
import string
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
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
        if self.config.thinking_mode_off:
            sentences = [s + " /no_think" for s in sentences]
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
                output_hidden_states=self.config.output_hidden_states,
                output_attentions=self.config.output_attentions
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
        bow_matrix = bow_matrix.reshape(1, -1, bow_matrix.shape[-1])  # (1, num_sentences, vocab_size)
        print(f"Shape of bag-of-words matrix: {bow_matrix.shape}")
        return bow_matrix
    
    def decoding_probing(self, activation, y, comps_ids=None):
        """
        Main probing routine (designed to avoid PCA data leakage):
        1. Use the provided sentence embeddings;
        2. Optionally project with a transfer matrix;
        3. Run cross-validation with Pipeline(PCA→Classifier) on each layer;
        4. Append the results to the output file.
        """
        if self.config.transfer_matrix is not None:
            activation = self.config.transfer_matrix.fc1(activation)

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
                    avg_scores = ",".join(f"{scores[f'test_{m}'].mean():.4f}" 
                                        for m in scoring)
                    fold_scores = ",".join(
                        ",".join(f"{s:.4f}" for s in scores[f'test_{m}'])
                        for m in scoring
                    )
                    print(f"{prefix},{avg_scores},{fold_scores}", file=f_out)

            print("-" * 60)
        del activation
        torch.cuda.empty_cache()
    
    def run_blimp(self, data_path):
        """
        Run the BLiMP dataset probing tasks.
        """
        jsonl_files = self._find_files(data_path, endswith='.jsonl')
        task_num = len(jsonl_files)
        print("Number of BLiMP files:", task_num)

        for file_id in range(task_num):
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
            if self.config.bow_mode:
                activation = self.get_sentence_embeddings_bow(sentences)
            else:
                activation = self.get_sentence_embeddings(sentences)
            
            if self.config.save_to_mat:
                if isinstance(activation, torch.Tensor):
                    activation = activation.cpu().detach().to(torch.float32).numpy()
                elif isinstance(activation, np.ndarray):
                    activation = activation.astype(np.float32)
                data_to_mat = []
                data_to_mat.append({
                    'field': df.field[0],
                    'phenomenon': df.linguistics_term[0],
                    'task_name': type_name,
                    'sentences_good': df['sentence_good'].tolist(),
                    'sentences_bad': df['sentence_bad'].tolist(),
                    'activation': activation
                })
                data_to_mat = pd.DataFrame(data_to_mat)
                data_to_mat = utils.df_to_mat(data_to_mat)
                sio.savemat(self.config.embed_path.replace('.mat', f'_{type_name}.mat'), {f'{type_name}': data_to_mat})
                continue # skip the decoding step if saving to mat

            self.decoding_probing(activation, labels)
                
            del activation
            gc.collect()
            torch.cuda.empty_cache()
            
        return

    def run_blimp_single_task(self, data_path, file_id):
        """
        Run a single BLiMP probing task specified by file index.
        """
        jsonl_files = self._find_files(data_path, endswith='.jsonl')
        task_num = len(jsonl_files)
        print("Number of BLiMP files:", task_num)
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
        activation = self.get_sentence_embeddings(sentences)
        
        self.decoding_probing(activation, labels)
            
        del activation
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
                sentences = [i[:-1] for i in sentences] # remove the period at the end of the sentence
                
                y = np.concatenate((np.ones(len(sentences_task_good)), np.zeros(len(sentences_task_bad))))
                print(f'Number of sentences: {len(sentences)}')
                print(f'sentence example: {sentences[0]}')
                if self.config.bow_mode:
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
        