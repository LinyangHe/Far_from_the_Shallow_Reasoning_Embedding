import numpy as np
import scipy.io as sio
import pandas as pd
from residual_reasoning import ResidualReasoningConstructor

def get_words_from_bpe_tokens(tokenizer, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    words = []
    word2token_map = []

    current_word = ""
    current_token_indices = []

    for i, token in enumerate(tokens):
        if isinstance(token, (bytes, bytearray)):
            token = token.decode("utf-8")
        else:
            token = token

        if token.startswith("Ġ") or token.startswith(" ") or (i == 0):
            if current_token_indices:
                words.append(current_word)
                word2token_map.append(current_token_indices)

            current_word = token.lstrip("Ġ") if token.startswith("Ġ") else token.lstrip()
            current_token_indices = [i]
        else:
            current_word += token
            current_token_indices.append(i)

    if current_token_indices:
        words.append(current_word)
        word2token_map.append(current_token_indices)
    return words, word2token_map


def get_word_level_embedding(
    rr: ResidualReasoningConstructor,
    corpus: str,
):
    inputs = rr.tokenizer(corpus, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"][0]
    
    words, word2token_map = get_words_from_bpe_tokens(rr.tokenizer, input_ids)
    hidden, residual_list = rr.get_residuals(corpus, return_hidden_states=True)  # list of np.ndarray, pick the first
    df_list = []

    for residual_index, (source_layer, target_layer) in enumerate(rr.source_target_layers):
    
        layerX = hidden[source_layer]              # shape: (n_tokens, hidden_dim)
        layerY = hidden[target_layer]   # shape: (n_tokens, hidden_dim)
        residuals = residual_list[residual_index]

        data = []
        for i, (word, token_idxs) in enumerate(zip(words, word2token_map)):
            embX = layerX[token_idxs].mean(axis=0)
            embY = layerY[token_idxs].mean(axis=0)
            res = residuals[token_idxs].mean(axis=0)
            data.append({
                "word_index": i,
                "word": word,
                f"layer_{source_layer}": embX,
                f"layer_{target_layer}": embY,
                "residual": res
            })

        df = pd.DataFrame(data)
        df_list.append(df)
    
    return df_list
