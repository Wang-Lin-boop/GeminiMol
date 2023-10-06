import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from chem import gen_standardize_smiles, get_skeleton

def scaffold_split(dataset, smiles_column='smiles', label_column='Y', test_ratio=0.3, val_ratio=0.3):
    dataset['skeleton'] = dataset[smiles_column].apply(lambda x:get_skeleton(x))
    dataset = dataset[dataset['skeleton']!='smiles_unvaild']
    dataset_pass = False
    while dataset_pass == False:
        train_skeletons, val_and_test_skeletons = train_test_split(dataset['skeleton'].unique(), test_size=test_ratio)
        test_skeletons, val_skeletons = train_test_split(val_and_test_skeletons, test_size=val_ratio)
        train = pd.DataFrame(columns=dataset.columns)
        test = pd.DataFrame(columns=dataset.columns)
        val = pd.DataFrame(columns=dataset.columns)
        for skeleton in dataset['skeleton'].unique():
            if skeleton in train_skeletons:
                train = pd.concat([train, dataset.loc[dataset['skeleton'] == skeleton]])
            elif skeleton in val_skeletons:
                val = pd.concat([val, dataset.loc[dataset['skeleton'] == skeleton]])
            elif skeleton in test_skeletons:
                test = pd.concat([test, dataset.loc[dataset['skeleton'] == skeleton]])
        if label_column is None:
            dataset_pass = True
        elif len(set(train[label_column].to_list())) >= 2 and len(set(val[label_column].to_list())) >= 2 and len(set(test[label_column].to_list())) >= 2:
            dataset_pass = True
    return train, val, test

if __name__ == '__main__':
    dataset = pd.read_csv(sys.argv[1])
    dataset_name = sys.argv[2]
    smiles_column = sys.argv[3]
    label_column = sys.argv[4]
    dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x))
    dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
    train, val, test = scaffold_split(dataset, smiles_column=smiles_column, label_column=label_column)
    print(dataset_name, '::  Train:', len(train), '  Validation:', len(val), '  Test:', len(test))
    train[['ID', smiles_column, label_column]].to_csv(f'{dataset_name}_scaffold_train.csv', index=False)
    val[['ID', smiles_column, label_column]].to_csv(f'{dataset_name}_scaffold_valid.csv', index=False)
    test[['ID', smiles_column, label_column]].to_csv(f'{dataset_name}_scaffold_test.csv', index=False)