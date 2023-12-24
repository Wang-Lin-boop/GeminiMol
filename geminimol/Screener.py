import os
import sys
import time
import torch
import pandas as pd
from utils.chem import gen_standardize_smiles, check_smiles_validity

class Virtual_Screening:
    def __init__(self, predictor, metric = 'Cosine'):
        self.predictor = predictor
        self.metric = metric
    
    def prepare(self, dataset, smiles_column='smiles'):
        if self.standardize == True:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x, kekule=False))
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:check_smiles_validity(x))
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset

    def __call__(self, ref_smiles_table, compound_library, score_name=None, prepare=True, standardize=True, return_ref_id=False, reverse=False, only_best_match=True):
        ## checking the input smiles library
        assert 'SMILES' in ref_smiles_table.columns and 'Title' in ref_smiles_table.columns, "Error: the table of reference smiles must contain both SMILES and Title columns."
        assert 'SMILES' in compound_library.columns and 'Title' in compound_library.columns, "Error: the table of compound library must contain both SMILES and Title columns."
        if score_name is None:
            score_name = self.metric
        assert score_name in self.predictor.similarity_metrics_list, "Error: the specified score item isn't match to your selected model."
        self.standardize = standardize
        if prepare:
            ref_smiles_table = self.prepare(ref_smiles_table, smiles_column='SMILES')
            ref_smiles_table.drop_duplicates(
                subset=['SMILES'], 
                keep='first', 
                inplace=True,
                ignore_index = True
                )
            ref_smiles_table.reset_index(drop=True, inplace=True)
            compound_library = self.prepare(compound_library, smiles_column='SMILES')
            compound_library.drop_duplicates(
                subset=['SMILES'], 
                keep='first', 
                inplace=True,
                ignore_index = True
                )
            compound_library.reset_index(drop=True, inplace=True)
        ref_smiles_dict = dict(zip(ref_smiles_table['Title'], ref_smiles_table['SMILES']))
        if return_ref_id:
            total_res = pd.DataFrame() 
            for ref_mol, ref_smiles in ref_smiles_dict.items():
                pred_res = self.predictor.virtual_screening([ref_smiles], compound_library, reverse=reverse, smiles_column='SMILES', similarity_metrics=score_name)
                pred_res['ref_id'] = ref_mol
                total_res = pd.concat([total_res, pred_res], ignore_index=True)
        else:
            total_res = self.predictor.virtual_screening(list(ref_smiles_dict.values()), compound_library, reverse=reverse, smiles_column='SMILES', similarity_metrics=score_name)
        total_res.sort_values(score_name, ascending=False, inplace=True)
        if only_best_match:
            total_res.drop_duplicates(
                subset=['SMILES'], 
                keep='first', 
                inplace=True,
                ignore_index = True
            )
        return total_res

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    ## load model
    model_name = sys.argv[1]
    if os.path.exists(f'{model_name}/GeminiMol.pt'):
        from model.GeminiMol import GeminiMol
        predictor = GeminiMol(model_name)
        metric = 'Pearson'
    else:
        from utils.fingerprint import Fingerprint
        predictor = Fingerprint(model_name) # ECFP4 or TopologicalTorsion
        metric = 'Tversky'
    predictor = Virtual_Screening(predictor, metric = metric)
    # running
    job_name = sys.argv[2]
    keep_number = int(sys.argv[5])
    ref_smiles_table = pd.read_csv(sys.argv[3])
    compound_library = pd.read_csv(sys.argv[4])
    prepare_library = False
    smiles_col = sys.argv[6]
    id_col = sys.argv[7]
    compound_library = compound_library[[id_col, smiles_col]]
    compound_library.columns = ['Title', 'SMILES']
    if "Targets" in compound_library.columns:
        reverse_screening = True ## set reverse to True when idenifiying drug targets.
    else:
        reverse_screening = False ## set reverse to True when idenifiying drug targets.
    if "Label" in ref_smiles_table.columns:
        active_compounds = ref_smiles_table[ref_smiles_table["Label"].isin(["active", "Active", "Yes", "yes", "true", "True", 1])]
        inactive_compounds = ref_smiles_table[ref_smiles_table["Label"].isin(["inactive", "Inactive", "No", "no", "false", "False", 0])]
        active_total_res = predictor(active_compounds, compound_library, return_ref_id=True, prepare=prepare_library, standardize=prepare_library, reverse=reverse_screening) ## set reverse to True when idenifiying drug targets.
        active_total_res['Active_Probability'] = active_total_res[predictor.metric]
        total_res = predictor(inactive_compounds, active_total_res.head(keep_number*10), return_ref_id=False, prepare=prepare_library, standardize=prepare_library, reverse=reverse_screening)
        total_res['Inactive_Probability'] = total_res[predictor.metric]
        total_res['Probability'] = max(total_res['Active_Probability'] - total_res['Inactive_Probability'], 0)
        total_res.sort_values('Probability', ascending=False, inplace=True)
    else:
        total_res = predictor(ref_smiles_table, compound_library, return_ref_id=True, prepare=prepare_library, standardize=prepare_library, reverse=reverse_screening, only_best_match=False) ## set reverse to True when idenifiying drug targets.
        total_res['Probability'] = total_res[predictor.metric]
    if reverse_screening:
        total_res.drop_duplicates(
            subset=['Targets'], 
            keep='first', 
            inplace=True,
            ignore_index = True
        )
    del total_res['features']
    total_res.head(keep_number).to_csv(f"{job_name}_results.csv", index=False, header=True, sep=',')
    # cleaning and exiting
    del predictor
    torch.cuda.empty_cache()
    time.sleep(5)




