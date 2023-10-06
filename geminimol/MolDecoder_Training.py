import os
import sys
import time
import json
import torch
from random import sample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model.GeminiMol import GeminiMolDecoder
from utils.chem import gen_standardize_smiles

def test_decoder(predictor_geminimol, smiles_list):
    best_recovery, best_validity, best_similarity_3D, best_similarity_2D = predictor_geminimol.evaluate(smiles_list)
    model_score = best_similarity_3D + best_similarity_2D
    print(f'Validity    {round(best_validity, 3)}')
    print(f'Recovery    {round(best_recovery, 3)}')
    print(f'3D Similarity    {round(best_similarity_3D, 3)} ({round(best_similarity_3D/(best_validity+1.0e-12), 3)})')
    print(f'2D Similarity    {round(best_similarity_2D, 3)} ({round(best_similarity_2D/(best_validity+1.0e-12), 3)})')
    print(f'Model score    {round(model_score, 3)}')

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    # params
    model_name = sys.argv[1]
    # read data and build dataset
    decoder_head = int(sys.argv[2])
    decoder_layers = int(sys.argv[3])
    decoder_embedding_dim = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    vocab_dict = f"{sys.argv[6]}/stat/vocabularies.json"
    with open(vocab_dict, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    mlp_layers={ # "MLP:0:GELU,MLP:0:GELU,MLP:3:GELU"
            'css_concentrate': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0},
            'mapping': {'type': 'MLP', 'layers': 3, 'activation': 'GELU', 'dropout': 0.0},
            'prompt': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0}
        }
    mlp_layers_cmd = sys.argv[7] 
    mlp_layers={ 
            'css_concentrate': {
                'type': mlp_layers_cmd.split(',')[0].split(':')[0], 
                'layers': int(mlp_layers_cmd.split(',')[0].split(':')[1]), 
                'activation': mlp_layers_cmd.split(',')[0].split(':')[2],
                'dropout': float(mlp_layers_cmd.split(',')[0].split(':')[3])
                },
            'prompt': {
                'type': mlp_layers_cmd.split(',')[1].split(':')[0], 
                'layers': int(mlp_layers_cmd.split(',')[1].split(':')[1]), 
                'activation': mlp_layers_cmd.split(',')[1].split(':')[2],
                'dropout': float(mlp_layers_cmd.split(',')[1].split(':')[3])
                },
            'mapping': {
                'type': mlp_layers_cmd.split(',')[2].split(':')[0], 
                'layers': int(mlp_layers_cmd.split(',')[2].split(':')[1]), 
                'activation': mlp_layers_cmd.split(',')[2].split(':')[2],
                'dropout': float(mlp_layers_cmd.split(',')[2].split(':')[3])
                }
            }
    params = {
        "decoder_embedding_size": decoder_embedding_dim,
        "decoder_head": decoder_head,
        "decoder_layers": decoder_layers,
        "batch_size": batch_size,
        "mlp_layers": mlp_layers
    }
    print(params)
    # set training params
    epochs = int(sys.argv[8].split(':')[0])
    zinc_finetune_epochs = int(sys.argv[8].split(':')[1])
    temperature = float(sys.argv[9])
    # initial a GeminiMolDecoder model
    predictor_geminimol = GeminiMolDecoder(model_name, vocab_dict, **params)
    with open(f'{model_name}/tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    with open(f"{model_name}/mol_decoder_params.json", 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    # read data and build dataset
    if os.path.exists(f"{sys.argv[6]}/mol_set/MolDecoder.smi"):
        data_set = pd.read_csv(f"{sys.argv[6]}/mol_set/MolDecoder.smi")
    else:
        DS_data_set = pd.read_csv(f"{sys.argv[6]}/mol_set/DeepShapeDB.smi", header=None, usecols=[0], sep='\s|,|;|\t| ',engine='python', names=['smiles'])
        DS_data_set['smiles'] = DS_data_set['smiles'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        DS_data_set = DS_data_set[DS_data_set['smiles']!='smiles_unvaild']
        MC_data_set = pd.read_csv(f"{sys.argv[6]}/mol_set/HitLocator_MedChem.smi", header=None, usecols=[0], sep='\s|,|;|\t| ',engine='python', names=['smiles'])
        MC_data_set['smiles'] = MC_data_set['smiles'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        MC_data_set = MC_data_set[MC_data_set['smiles']!='smiles_unvaild']
        # mark
        DS_data_set['source'] = 'DeepShapeDB'
        MC_data_set['source'] = 'MedChem'
        # split dataset
        train_set, test_set = train_test_split(DS_data_set, test_size=0.1, random_state=training_random_seed)
        train_set, val_set = train_test_split(train_set, test_size=0.05, random_state=training_random_seed)
        train_set = pd.concat([train_set, MC_data_set])
        train_set['assign'] = 'train'
        val_set['assign'] = 'val'
        test_set['assign'] = 'test'
        data_set = pd.concat([train_set, val_set, test_set])
        data_set.to_csv(f"{sys.argv[6]}/mol_set/MolDecoder.smi", index=False)
    # training
    data_set.to_csv(f"{model_name}/data.csv", index=False)
    train_set = data_set[(data_set['assign'] == 'train') & (data_set['source'] == 'DeepShapeDB')]['smiles'].tolist()
    val_set = data_set[data_set['assign']=='val']['smiles'].tolist()
    test_set = data_set[data_set['assign']=='test']['smiles'].tolist()
    medchem_set = data_set[(data_set['assign'] == 'train') & (data_set['source'] == 'MedChem')]['smiles'].tolist()
    print('GeminiMol Training Set: Number=', len(train_set))
    print('MedChem Training Set: Number=', len(medchem_set))
    print('Validation Set: Number=', len(val_set))
    print('Test Set: Number=', len(test_set))
    print(f"\nNOTE: Training {model_name} GeminiMolDecoder...\n")
    if epochs > 0:
        predictor_geminimol.fit(
            train_smiles_list = train_set * 10 + medchem_set,
            similarity_metrics = 'Cosine',
            val_smiles_list = val_set, 
            epochs = epochs, 
            learning_rate = 5.0e-5, 
            optim_type='AdamW',
            cross_entropy=True,
            valid_award=0,
            weight_dict=True,
            fine_tune=False,
            flooding=None,
            temperature=temperature,
            batch_group=5,
            mini_epoch=50
        )
    # test best model
    print('======== Job Report (Before FT) ========')
    print(f'Test on training set:')
    test_decoder(predictor_geminimol, train_set)
    print(f'Test on validation set:')
    test_decoder(predictor_geminimol, val_set)
    print(f'Test on test set:')
    test_decoder(predictor_geminimol, test_set)
    print('========================================')
    if zinc_finetune_epochs > 0:
        for zinc_subset in sample(os.listdir(f"{sys.argv[6]}/mol_set/zinc/"), zinc_finetune_epochs):
            extrnal_data = pd.read_csv(f"{sys.argv[6]}/mol_set/zinc/{zinc_subset}", header=None, usecols=[0], sep='\s|,|;|\t| ',engine='python', names=['smiles'])
            extrnal_data['smiles'] = extrnal_data['smiles'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
            extrnal_data = extrnal_data[extrnal_data['smiles']!='smiles_unvaild']
            extrnal_data['assign'] = 'train'
            extrnal_data['source'] = zinc_subset
            data_set = pd.concat([data_set, extrnal_data], ignore_index=True)
            data_set.to_csv(f"{model_name}/data.csv", index=False)
            print(f'Extrnal Training Set {zinc_subset}: Number=', len(extrnal_data))
            predictor_geminimol.fit(
                train_smiles_list = extrnal_data['smiles'].tolist() + train_set,
                similarity_metrics = 'Cosine',
                val_smiles_list = val_set, 
                epochs = 2, 
                learning_rate = 1.0e-5, 
                optim_type='AdamW',
                cross_entropy=True,
                valid_award=0,
                weight_dict=True,
                fine_tune=True,
                temperature=temperature,
                flooding=0.2,
                batch_group=5,
                mini_epoch=20
            )
    # test best model
    print('======== Job Report (Before FT) ========')
    print(f'Test on training set:')
    test_decoder(predictor_geminimol, train_set)
    print(f'Test on validation set:')
    test_decoder(predictor_geminimol, val_set)
    print(f'Test on test set:')
    test_decoder(predictor_geminimol, test_set)
    print('========================================')
    # cleaning and exiting
    del predictor_geminimol
    torch.cuda.empty_cache()
    time.sleep(5)