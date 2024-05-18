import sys
import pandas as pd
from chem import cal_MCS_score, gen_standardize_smiles

if __name__ == "__main__":
    # params
    data_table = pd.read_csv(sys.argv[1])
    # prepare the smiles
    data_table['smiles1'] = data_table['smiles1'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
    data_table = data_table[data_table['smiles1'] != 'smiles_unvaild'].reset_index(drop=True)
    data_table['smiles2'] = data_table['smiles2'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
    data_table = data_table[data_table['smiles2'] != 'smiles_unvaild'].reset_index(drop=True)
    ## MCS
    data_table['MCS_any'] = data_table.apply(
        lambda x: cal_MCS_score(x['smiles1'], x['smiles2'], atom_mode="any"), 
        axis=1
    )
    data_table['MCS_elements'] = data_table.apply(
        lambda x: cal_MCS_score(x['smiles1'], x['smiles2'], atom_mode="elements"), 
        axis=1
    )
    data_table.to_csv(sys.argv[1], index=False)
