import sys
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import rdkit.Chem.rdFMCS as FMCS
from model.GeminiMol import GeminiMolDecoder
from utils.chem import rule_of_five

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    ## load model
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    predictor = GeminiMolDecoder(model_name)
    predictor.batch_size = batch_size
    predictor.geminimol.batch_size = batch_size
    running_mode = sys.argv[3]
    ## setup ref smiles
    start_point = str(sys.argv[4])
    jobname = str(sys.argv[5])
    if running_mode == 'MCMC':
        output_smiles_list = predictor.MCMC(start_point, predictor.scaffold_hopping_award, replica_num=200, num_steps_per_replica=30, temperature=0.1, num_seeds_per_steps=24, iterative_mode='random', init_seeds=30)
    elif running_mode == 'CMCMC':
        output_smiles_list = predictor.MCMC(start_point, predictor.scaffold_hopping_award, replica_num=200, num_steps_per_replica=30, temperature=0.1, num_seeds_per_steps=24, iterative_mode='continous', init_seeds=1)
    elif running_mode in ['AdamW', 'SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adamax', 'Rprop']:
        output_smiles_list = predictor.directed_evolution(start_point, predictor.scaffold_hopping_award, optim_type=running_mode, replica_num=200, num_steps_per_replica=500, temperature=0.1)
    else:
        raise ValueError(f'Error: the running mode {running_mode} is not supported.')
    ## output
    output_smiles_list = list(set(output_smiles_list))
    smiles_table = {
        'SMILES': [],
        'MCS_Similarity': [],
        'validity': []
    }
    ref_mol = Chem.MolFromSmiles(start_point)
    for decoded_smiles in output_smiles_list:
        mol = None
        try:
            mol = Chem.MolFromSmiles(decoded_smiles)
            if mol is None:
                pass
            elif mol.GetNumAtoms() <= 6:
                pass
            elif rule_of_five(mol) == 0:
                pass
            else:
                res = FMCS.FindMCS([mol, ref_mol], ringMatchesRingOnly=True, atomCompare=(FMCS.AtomCompare.CompareElements))
                Chem.SanitizeMol(mol)
                decoded_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=False, isomericSmiles=True)
                smiles_table['SMILES'] += [decoded_smiles]
                smiles_table['MCS_Similarity'] += [ res.numBonds / len(ref_mol.GetBonds()) ]
                smiles_table['validity'] += [1]
        except:
            pass
    if len(smiles_table['SMILES']) == 0:
        raise ValueError(f'Error: no valid molecules are generated.')
    pred_res = predictor.geminimol.virtual_screening([start_point], pd.DataFrame(smiles_table), smiles_column='SMILES')
    pred_res.sort_values('MCS_Similarity', ascending=True, inplace=True)
    pred_res.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)
    del pred_res['validity']
    del pred_res['features']
    pred_res.to_csv(f'{jobname}_output.csv', index=False)
    print('Successfully Done!')
    print(pred_res)
    print(f'Output: {jobname}_output.csv')