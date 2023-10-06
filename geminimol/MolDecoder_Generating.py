import sys
import os
import re
import torch
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from model.GeminiMol import GeminiMolDecoder

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    model_path = sys.argv[1]
    model_name = model_path.split('/')[-1]
    replica_num = int(sys.argv[2])
    smiles_len_list = [64, 72, 96, 108, 112, 128, 144, 169, 181, 198]
    predictor = GeminiMolDecoder(model_path)
    output_smiles = predictor.random_walking(
        replica_num=replica_num, 
        smiles_len_list=smiles_len_list, 
        num_steps_per_replica=10,
        temperature=0.1,
        )
    benchmark_dict = {}
    jobname = f'{model_name}_{str(random.randint(100, 999))}'
    os.mkdir(jobname)
    for smiles_len in smiles_len_list:
        output_smiles_list = list(set(output_smiles[smiles_len]))
        smiles_table = {
            'SMILES': [],
            'validity': []
        }
        generated_smiles_number = len(output_smiles_list)
        validity_score = 0
        for decoded_smiles in output_smiles_list:
            mol = None
            try:
                if predictor.check_unvalid_smiles(decoded_smiles):
                    mol = Chem.MolFromSmiles(decoded_smiles)
                    if mol is None:
                        smiles_table['SMILES'] += [decoded_smiles]
                        smiles_table['validity'] += [0]
                    elif mol.GetNumAtoms() <= 2:
                        smiles_table['SMILES'] += [decoded_smiles]
                        smiles_table['validity'] += [0]
                    else:
                        Chem.SanitizeMol(mol)
                        decoded_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=False, isomericSmiles=True)
                        smiles_table['SMILES'] += [decoded_smiles]
                        smiles_table['validity'] += [1]
                        validity_score += 1
                else:
                    smiles_table['SMILES'] += [decoded_smiles]
                    smiles_table['validity'] += [0]
            except:
                smiles_table['SMILES'] += [decoded_smiles]
                smiles_table['validity'] += [0]
        gene_res = pd.DataFrame(smiles_table)
        gene_res.to_csv(f'{jobname}/{str(smiles_len)}_output.csv', index=False)
        print(f'validity score on {smiles_len}: {validity_score/generated_smiles_number}')
