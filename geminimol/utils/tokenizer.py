import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from chem import is_valid_smiles, is_single_ring_system, gen_smiles_list

def save_vocabulary_to_json(vocab_dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

def unpaired_brackets(string):
    stack = []
    brackets_map = {')': '(', ']': '['}
    for char in string:
        if char in ('(', '['):
            stack.append(char)
        elif char in (')', ']'):
            if not stack:
                return False
            if stack[-1] == brackets_map[char]:
                stack.pop()
            else:
                return False
    return len(stack) == 0

def extract_smiles_vocabulary(string_list, cutoff, min_count):
    vocab = {}
    for s in string_list:
        for i in range(len(s)):
            for j in range(i+1, min(i+cutoff+1, len(s)+1)):
                word = s[i:j]
                if unpaired_brackets(word):
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
    filtered_vocab = {k: v for k, v in vocab.items() if len(k) <= cutoff and is_valid_smiles(k) and v >= min_count and is_single_ring_system(k)}
    word_count = sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
    # add special_tokens
    special_tokens = {'[UNK]': 0, '<PAD>': 1, '[CLS]': 2, '[MASK]': 5, '<EOS>': 6, '[SEP]': 7, 'Cl': 8, 'B': 9, 'o': 10, 'N': 11, '[N+]': 12, '[C@H]': 13, '\\': 14, '3': 15, '(': 16, '8': 17, '[C@@H]': 18, '10': 19, 'I': 20, '[O-]': 21, '[nH]': 22, 'F': 23, 'c': 24, '12': 25, ')': 26, ']': 27, '[': 28, 'S': 29, 'P': 30, '2': 31, 'n': 32, '-': 33, '11': 34, '.': 35, '%10': 36, '#': 37, '%': 38, 'O': 39, '~': 40, '7': 41, 's': 42, '=': 43, 'C': 44, '4': 45, '>>': 46, '/': 47, 'Br': 48, '9': 49, '6': 50, '5': 51, '1': 52, 'p': 53}
    vocab_dict = {pair[0]: i+len(special_tokens) for i, pair in enumerate(word_count)}
    vocab_dict.update(special_tokens)
    vocab_list = list(vocab_dict.keys())
    vocab_dict = {vocab_list[i]:i for i in range(len(vocab_list))}
    return vocab_dict

if __name__ == "__main__":
    column_list = ['smiles1', 'smiles2']
    index_set = pd.read_csv(sys.argv[1], header=None, names=['ID', 'smiles'], dtype={'smiles':str, 'ID':str})
    expand_ratio = int(sys.argv[2])
    index_set['list'] = index_set['smiles'].apply(lambda x:gen_smiles_list(x, expand_ratio))
    dataset_pass = False
    groups = index_set.groupby('smiles')
    indexs = [x for x in groups.groups.values()]
    ## Get a tokenizer for the DeepShape compounds dataset
    while dataset_pass == False:
        train_data, val_and_test = train_test_split(indexs, test_size=0.03)
        train_data = pd.concat([index_set.iloc[idx] for idx in train_data])
        val_and_test = pd.concat([index_set.iloc[idx] for idx in val_and_test])
        overlap_smiles = [i for i in val_and_test['smiles'].to_list() if i in train_data['smiles'].to_list()]
        if len(overlap_smiles) > 0:
            continue
        else:
            print('NOTE: find a nice split for train and val+test data.')
        train_smiles = [item for sublist in train_data['list'].to_list() for item in sublist]
        other_smiles = [item for sublist in val_and_test['list'].to_list() for item in sublist]
        train_vocab_dict = extract_smiles_vocabulary(train_smiles, 18, 300)
        other_vocab_dict = extract_smiles_vocabulary(other_smiles, 18, 300)
        train_vocab_list = list(train_vocab_dict.keys())
        other_vocab_list = list(other_vocab_dict.keys())
        print('NOTE: successfully extracted vocabulary.')
        if set(other_vocab_list).issubset(set(train_vocab_list)):
            dataset_pass = True
            save_vocabulary_to_json(train_vocab_dict, 'vocabulary.json')
            val_data, test_data = train_test_split(val_and_test, test_size=0.7)
        else:
            print('Warning: the test set contained words that did not appear in the training set.')
    ## The division of the dataset cannot be based on token, otherwise it is easy to overfitting!
    index_set = index_set.set_index('smiles')
    train_data, val_and_test = train_test_split(index_set.index, test_size=0.05)
    val_data, test_data = train_test_split(val_and_test, test_size=0.7)
    train_data = index_set.loc[train_data, :]
    val_data = index_set.loc[val_data, :]
    test_data = index_set.loc[test_data, :]
    train_data['assign'] = 'train'
    val_data['assign'] = 'val'
    test_data['assign'] = 'test'
    index_set = pd.concat([train_data, val_data, test_data])
    index_set = index_set.reset_index()
    index_set['list'] = index_set['list'].apply(json.dumps)
    index_set.to_json('index.json', orient='records')



