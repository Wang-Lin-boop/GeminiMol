import sys
import json
import pandas as pd
from collections import Counter

def token_embedding(input_string, vocab_dict, max_seq_len):
    # Convert input sentence to a list of numerical indices
    indices = [vocab_dict['[CLS]']]
    i = 0
    while i < len(input_string):
        best_match = None
        for word, index in vocab_dict.items():
            if input_string.startswith(word, i):
                if not best_match or len(word) > len(best_match[0]):
                    best_match = (word, index)
        if best_match:
            indices.append(best_match[1])
            i += len(best_match[0])
        else:
            new_token = input_string[i]  # 未匹配的字符
            if new_token not in vocab_dict:
                new_index = len(vocab_dict)  # 分配一个新的索引
                vocab_dict[new_token] = new_index  # 将新的字符和索引添加到词汇表
            indices.append(vocab_dict[new_token])  # 将新的字符的索引添加到 indices 列表中
            i += 1
        if len(indices) == max_seq_len:
            break
    pad_len = max_seq_len - len(indices)
    indices += [vocab_dict['<EOS>']]
    indices += [vocab_dict['<PAD>']] * pad_len
    return indices, vocab_dict

def token_embedding_batch(input_strings, vocab_dict, max_seq_len):
    indices_batch = []
    for input_string in input_strings:
        indices, vocab_dict = token_embedding(input_string, vocab_dict, max_seq_len)
        indices_batch.append(indices)
    return indices_batch, vocab_dict

def generate_weights(vocabulary, training_data):
    counter = Counter()
    for sentence in training_data:
        counter.update(sentence)
    weights = {}
    total_tokens = sum(counter.values())
    for token, index in vocabulary.items():
        token_count = counter.get(index, 0)
        token_weight = total_tokens / token_count if token_count > 0 else 0.0
        weights[token] = token_weight
    return weights

if __name__ == '__main__':
    data_set = pd.read_csv(f"{sys.argv[1]}/MolDecoder.smi")
    smiles_list = data_set['smiles'].tolist()
    vocab_dict = {
            "<PAD>": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "<EOS>": 3,
            "C": 4,
            "O": 5,
            "N": 6,
            "S": 7,
            "P": 8,
            "F": 9,
            "Cl": 10,
            "Br": 11,
            "I": 12,
            "c": 13,
            "o": 14,
            "n": 15,
            "s": 16,
            "[nH]": 17,
            "(=O)": 18,
            "[O-]": 19,
            "[C@]": 20,
            "[C@@]": 21,
            "[C@H]": 22,
            "[C@@H]": 23,
            "[NH+]": 24,
            "[NH2+]": 25,
            "[NH3+]": 26,
            "[N@H+]": 27,
            "[N@@H+]": 28,
            "[N+]": 29,
            "[N-]": 30,
            "\\": 31,
            "/": 32,
            "=": 33,
            "#": 34,
            "~": 35,
            "%": 36,
            ".": 37,
            "-": 38,
            "+": 39,
            "@": 40,
            "]": 41,
            "[": 42,
            "(": 43,
            ")": 44,
            "1": 45,
            "2": 46,
            "3": 47,
            "4": 48,
            "5": 49,
            "6": 50,
            "7": 51,
            "8": 52,
            "9": 53,
            "0": 54,
            "Si": 55,
            "H": 56
    }
    token_database, vocab_dict = token_embedding_batch(smiles_list, vocab_dict, 256)
    token_weights = generate_weights(vocab_dict, token_database)
    n_token_weights_dict = {k: v for k, v in token_weights.items() if v > 0.0}
    n_vocab_dict = {k: v for k, v in vocab_dict.items() if k in list(n_token_weights_dict.keys())}
    n_vocab_dict = {list(n_vocab_dict.keys())[i]:i for i in range(len(n_vocab_dict))}
    with open(f'token_weights.json', 'w', encoding='utf-8') as f:
        json.dump(n_token_weights_dict, f, ensure_ascii=False, indent=4)
    with open(f'vocabularies.json', 'w', encoding='utf-8') as f:
        json.dump(n_vocab_dict, f, ensure_ascii=False, indent=4)
    