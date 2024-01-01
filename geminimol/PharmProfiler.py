import os
import sys
import torch
import pandas as pd
from model.GeminiMol import GeminiMol
from utils.chem import gen_standardize_smiles, check_smiles_validity, is_valid_smiles

class Pharm_Profiler:
    def __init__(self, 
            encoder, 
            standardize = False
        ):
        self.encoder = encoder
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

    def update_library(self,
        compound_library,
        prepare = True,
        smiles_column = 'smiles',
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
        print(f'NOTE: non-duplicates compound library contains {len(compound_library)} compounds.')
        self.features_database = self.encoder.create_database(
            compound_library, 
            smiles_column = smiles_column, 
            worker_num = 2
        )
        print('NOTE: features database was created.')
        return self.features_database

    def __call__(
        self, 
        smiles_column = 'smiles',
        probe_cluster = False,
        smiliarity_metrics = 'Pearson',
    ):
        print(f'NOTE: columns of feature database: {self.features_database.columns}')
        total_res = self.features_database.copy()
        del total_res['features']
        print(f'NOTE: starting screening...')
        score_list = []
        for name, probe in self.probes_dict.items():
            print(f'NOTE: using {name} as the probe.')
            probe_list = probe['smiles']
            if probe_cluster:
                probe_res = self.encoder.virtual_screening(
                    probe_list, 
                    self.features_database, 
                    input_with_features = True,
                    reverse = True, 
                    smiles_column = smiles_column, 
                    similarity_metrics = [smiliarity_metrics],
                    worker_num = 2
                )
                probe_res[f'{name}'] = probe['weight'] * probe_res[smiliarity_metrics]
                total_res = pd.merge(
                    total_res,
                    probe_res[[smiles_column, f'{name}']], 
                    on = smiles_column
                )
                score_list.append(f'{name}')
            else:
                for i in range(len(probe_list)):
                    probe_res = self.encoder.virtual_screening(
                        [probe['smiles'][i]], 
                        self.features_database, 
                        input_with_features = True,
                        reverse = True, 
                        smiles_column = smiles_column, 
                        similarity_metrics = [smiliarity_metrics],
                        worker_num = 2
                    )
                    probe_res[f'{name}_{i}'] = probe['weight'] * probe_res[smiliarity_metrics]
                    total_res = pd.merge(
                        total_res,
                        probe_res[[smiles_column, f'{name}_{i}']], 
                        on = smiles_column
                    )
                    score_list.append(f'{name}_{i}')
        total_res.fillna(0, inplace=True)
        total_res['Score'] = total_res[score_list].sum(axis=1)
        return total_res

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    ## load model
    model_name = sys.argv[1]
    encoder = GeminiMol(
            model_name,
            internal_label_list = [],
            extrnal_label_list = ['Pearson'],
        )
    predictor = Pharm_Profiler(
        encoder,
        standardize = True
        )
    # job_name
    job_name = sys.argv[2]
    smiles_col = sys.argv[3]
    compound_library = pd.read_csv(sys.argv[4])
    library_path = sys.argv[4].split('.')[0]
    # update profiles
    if ':' in sys.argv[5]:
        ref_smiles_table = pd.read_csv(sys.argv[5].split(':')[0])
        label_col = sys.argv[5].split(':')[1]
        for weight in ref_smiles_table[label_col].to_list(): # weight column 
            predictor.update_probes(
                name = f'w_{weight}',
                smiles_list = ref_smiles_table[ref_smiles_table[label_col]==weight][smiles_col].to_list(),
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
    probe_cluster = True if sys.argv[7] in [
        'True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y'
        ] else False
    # generate features database
    if os.path.exists(f'{library_path}.pkl'):
        predictor.features_database = pd.read_pickle(f'{library_path}.pkl')
    else:
        features_database = predictor.update_library(
            compound_library,
            prepare = True,
            smiles_column = smiles_col,
        )
        # save database to pkl
        features_database.to_pickle(f'{library_path}.pkl')
    # virtual screening 
    total_res = predictor(
        smiles_column = smiles_col,
        probe_cluster = probe_cluster,
    )
    total_res.sort_values('Score', ascending=False, inplace=True)
    total_res.head(keep_number).to_csv(f"{job_name}_results.csv", index=False, header=True, sep=',')
    print(f'NOTE: job completed! check {job_name}_results.csv for results!')

