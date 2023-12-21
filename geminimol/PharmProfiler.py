import sys
import torch
import pandas as pd
from model.GeminiMol import GeminiMol
from utils.chem import gen_standardize_smiles, check_smiles_validity, is_valid_smiles

class Pharm_Profiler:
    def __init__(self, 
            model_name, 
            standardize = False
        ):
        self.predictor = GeminiMol(
            model_name,
            internal_label_list = [],
            external_label_list = ['Pearson'],
        )
        self.probes_dict = {
            # name : { 
            # 'smiles': smiles_list: 
            # 'weight': weight (float) 
            # }
        }
        self.standardize = standardize

    def prepare(self, dataset, smiles_column='smiles'):
        if self.standardize == True:
            dataset[smiles_column] = dataset[smiles_column].apply(
                lambda x:gen_standardize_smiles(x, kekule=False)
            )
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(
                lambda x:check_smiles_validity(x)
            )
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset

    def update_probes(self, 
        name,
        smiles_list, 
        weight
    ):
        for smiles in smiles_list:
            assert is_valid_smiles(smiles), f'Error: the probe {smiles} is invalid.'
        self.probes_dict[name] = {
            'smiles': smiles_list,
            'weight': weight
        }

    def __call__(
        self, 
        compound_library, 
        prepare = True,
        smiles_column = 'smiles',
        target_column = 'target',
    ):
        if prepare:
            compound_library = self.prepare(compound_library, smiles_column=smiles_column)
        print(f'NOTE: the compound library contains {len(compound_library)} compounds.')
        compound_library.drop_duplicates(
            subset=[smiles_column], 
            keep='first', 
            inplace=True,
            ignore_index = True
            )
        compound_library.reset_index(drop=True, inplace=True)
        total_res = compound_library.copy()
        print(f'NOTE: non-duplicates compound library contains {len(compound_library)} compounds.')
        features_database = self.predictor.create_database(
            compound_library, smiles_column=smiles_column, worker_num=1
        )
        print('NOTE: features database was created.')
        if target_column in compound_library.columns:
            reverse_screening = True
            print(f'NOTE: the target column {target_column} was found in compound library.')
            print(f'NOTE: starting reverse screening...')
        else:
            reverse_screening = False
            print(f'NOTE: we did not find the target column {target_column} in compound library.')
            print(f'NOTE: starting forward screening...')
        for name, probe in self.probes_dict.items():
            print(f'NOTE: using {name} as the probe.')
            probe_res = self.predictor.virtual_screening(
                probe['smiles'], 
                features_database, 
                input_with_features = True,
                reverse = reverse_screening, 
                smiles_column = smiles_column, 
                similarity_metrics = 'Pearson',
                worker_num = 4
            )
            probe_res[name] = probe['weight'] * probe_res['Pearson']
            probe_res.drop(columns=[smiles_column], inplace=True)
            total_res = total_res.merge(probe_res[[smiles_column, name]], on=smiles_column, how='left')
        total_res.fillna(0, inplace=True)
        total_res['Score'] = total_res[list(self.probes_dict.keys())].sum(axis=1)
        return total_res

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    ## load model
    model_name = sys.argv[1]
    predictor = Pharm_Profiler(model_name)
    # job_name
    job_name = sys.argv[2]
    smiles_col = sys.argv[3]
    if ':' in sys.argv[4]:
        compound_library = pd.read_csv(sys.argv[4])
        target_col = 'none'
    else:
        compound_library = pd.read_csv(sys.argv[4].split(':')[0])
        target_col = sys.argv[4].split(':')[1]
    # update profiles
    if ':' in sys.argv[5]:
        ref_smiles_table = pd.read_csv(sys.argv[5].split(':')[0])
        for weight in ref_smiles_table[sys.argv[5].split(':')[1]].to_list(): # weight column 
            predictor.update_probes(
                name = f'weight_{weight}',
                smiles_list = ref_smiles_table[ref_smiles_table[weight]==weight][smiles_col].to_list(),
                weight = weight
                )
    else:
        ref_smiles_table = pd.read_csv(sys.argv[5])
        predictor.update_probes(
            name = 'active',
            smiles_list = ref_smiles_table[smiles_col].to_list(),
            weight = 1.0
            )
    keep_number = int(sys.argv[6])
    # virtual screening 
    total_res = predictor(
        compound_library,
        prepare = True,
        smiles_column = smiles_col,
        target_column = target_col,
    )
    total_res.sort_values('Score', ascending=False, inplace=True)
    total_res.head(keep_number).to_csv(f"{job_name}_results.csv", index=False, header=True, sep=',')

