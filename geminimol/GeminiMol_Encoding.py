import os
import sys
import numpy as np
import pandas as pd
from utils.fingerprint import Fingerprint
import matplotlib.pyplot as plt
from utils.chem import check_smiles_validity, gen_standardize_smiles

class Mol_Encoder:
    def __init__(self, model_name, encoder_list=[Fingerprint(['ECFP4'])], standardize=False, smiles_column='smiles'):
        self.model_name = model_name
        self.standardize = standardize
        self.encoders = encoder_list
        self.smiles_column = smiles_column

    def prepare(self, dataset):
        dataset = dataset.dropna(subset=[self.smiles_column])
        print(f"NOTE: read the dataset size ({len(dataset)}).")
        out_dataset = dataset.copy()
        if self.standardize == True:
            out_dataset.loc[:, self.smiles_column] = out_dataset[self.smiles_column].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        else:
            out_dataset.loc[:, self.smiles_column] = out_dataset[self.smiles_column].apply(lambda x: check_smiles_validity(x))
        out_dataset = out_dataset[out_dataset[self.smiles_column]!='smiles_unvaild']
        print(f"NOTE: processed dataset size ({len(out_dataset)}).")
        return out_dataset

    def encoder(self, query_smiles_table):
        features_columns = []
        query_smiles_table = self.prepare(query_smiles_table)
        for single_encoder in self.encoders:
            features = single_encoder.extract_features(query_smiles_table, smiles_column=self.smiles_column)
            features_columns += list(features.columns)
            query_smiles_table = query_smiles_table.join(features, how='left')
        return query_smiles_table, features_columns
    
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1])
    method = sys.argv[2]
    smiles_column = sys.argv[3]
    if method == "CombineFP":
        methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
        encoders = [Fingerprint([methods_list])]
    elif os.path.exists(f'{method}/GeminiMol.pt'):
        method_list = [method]
        from model.GeminiMol import GeminiMol
        encoders = [GeminiMol(method)]
    elif method.split(":")[0] == "GMFP" and os.path.exists(f'{method.split(":")[1]}/GeminiMol.pt'):
        method_list = ["ECFP4", "AtomPairs", "GeminiMol"]
        from model.GeminiMol import GeminiMol
        encoders = [ GeminiMol(method.split(":")[1]), Fingerprint(["FCFP4", "AtomPairs"]) ]
    else:
        methods_list = [method]
        encoders = [Fingerprint([methods_list])]
    predictor = Mol_Encoder(method, encoders, smiles_column=smiles_column)
    dataset, features_columns = predictor.encoder(dataset)
    # dataset.to_csv(f"{sys.argv[1].split('/')[-1].split('.')[0]}_{method.split('/')[-1]}_encoding.csv", index=False)
    print(dataset.head())
    print(f"NOTE: {method} encoding finished.")
    mean_values = dataset[features_columns].mean()
    std_values = dataset[features_columns].std()
    max_values = dataset[features_columns].max()
    min_values = dataset[features_columns].min()
    result_df = pd.DataFrame({'Mean': mean_values, 'Std': std_values, 'Max': max_values, 'Min': min_values})
    result_df['ExtremeVariance'] = result_df['Max'] - result_df['Min']
    result_df.sort_values(by=['ExtremeVariance'], ascending=False, inplace=True)
    result_df = result_df.reset_index()
    # result_df.to_csv(f"{sys.argv[1].split('/')[-1].split('.')[0]}_{method.split('/')[-1]}_stat.csv", index=True)
    print(result_df)
    if os.path.exists(f'{method}/GeminiMol.pt'):
        result_df.to_csv(f'{method}/feat_stat.csv', index=True)        
        # plot shade range graph for Mean and lines for Max/Min
        plt.figure(figsize=(3.6*1.0, 3.2*1.0), dpi=600) # 
        plt.plot(result_df.index, result_df['Mean'], color='#588AF5', label='Mean', linewidth=0.3)
        plt.plot(result_df.index, result_df['Max'], color='#D861F5', label='Max', linewidth=0.3)
        plt.plot(result_df.index, result_df['Min'], color='#F55A46', label='Min', linewidth=0.3)
        plt.fill_between(result_df.index, result_df['Mean'] + result_df['Std'], result_df['Mean'] - result_df['Std'], color='#F5D690', linewidth=0, alpha=0.8)
        plt.fill_between(result_df.index, result_df['Mean'] + 3 * result_df['Std'], result_df['Mean'] - 3* result_df['Std'], color='#6CF5B2', linewidth=0, alpha=0.3)
        plt.ylim(-3, 3)
        plt.xlim(0, len(result_df.index)+1)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(f'{method}/feat_stat.png')
        plt.close()
    



