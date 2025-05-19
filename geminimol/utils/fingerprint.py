import pandas as pd
from functools import partial
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

class Fingerprint():
    '''
    This class contains some basic molecular fingerprint feature extraction as well as similarity calculation methods, and they assume the role of baseline methods in drug discovery tasks.

    the fingerprint_type_list is looks like: 
        ["ECFP4"], single fingerprint for similarity calculation.
        ["ECFP4", "TopologicalTorsion", "MACCS"], multiple fingerprints for QSAR feature extraction.

    '''
    def __init__(self, fingerprint_type_list=["ECFP4"]):
        self.fingerprint_dict = {
            "similarity": {
                "MACCS": MACCSkeys.GenMACCSKeys,
                "RDK": Chem.RDKFingerprint,
                "Layered": Chem.rdmolops.LayeredFingerprint,
                "Pattern": Chem.rdmolops.PatternFingerprint,
                "TopologicalTorsion": partial(Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect, nBits=2048),
                "AtomPairs": partial(Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect, nBits=2048),
                "ECFP4": partial(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=2048),
                "FCFP4": partial(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=2, nBits=2048, useFeatures=True),
                "ECFP6": partial(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=3, nBits=2048),
                "FCFP6": partial(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect, radius=3, nBits=2048, useFeatures=True),
            },
            "QSAR": {
                "MACCS": MACCSkeys.GenMACCSKeys,
                "RDK": partial(Chem.RDKFingerprint, fpSize=2048),
                "Layered": Chem.rdmolops.LayeredFingerprint,
                "Pattern": Chem.rdmolops.PatternFingerprint,
                "TopologicalTorsion": partial(Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprint, nBits=2048),
                "AtomPairs": partial(Chem.rdMolDescriptors.GetHashedAtomPairFingerprint, nBits=2048),
                "ECFP4": partial(Chem.rdMolDescriptors.GetHashedMorganFingerprint, radius=2, nBits=2048),
                "FCFP4": partial(Chem.rdMolDescriptors.GetHashedMorganFingerprint, radius=2, nBits=2048, useFeatures=True),
                "ECFP6": partial(Chem.rdMolDescriptors.GetHashedMorganFingerprint, radius=3, nBits=2048),
                "FCFP6": partial(Chem.rdMolDescriptors.GetHashedMorganFingerprint, radius=3, nBits=2048, useFeatures=True),
            }
        }
        self.similarity_metrics = {
            "Tanimoto": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query), 
            "Dice": lambda ref, query: DataStructs.DiceSimilarity(ref, query), 
            "Cosine": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query, metric=DataStructs.CosineSimilarity), 
            "Sokal": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query, metric=DataStructs.SokalSimilarity), 
            "Russel": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query, metric=DataStructs.RusselSimilarity), 
            "Kulczynski": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query, metric=DataStructs.KulczynskiSimilarity), 
            "McConnaughey": lambda ref, query: DataStructs.FingerprintSimilarity(ref, query, metric=DataStructs.McConnaugheySimilarity), 
            "Tversky": lambda ref, query: DataStructs.cDataStructs.TverskySimilarity(query, ref, 0.01, 0.99)
        }
        self.similarity_metrics_list = self.similarity_metrics.keys()
        self.fingerprint_type_list = fingerprint_type_list

    def gen_fingerprints(self, smiles, fingerprint):
        mol = Chem.MolFromSmiles(smiles)
        return self.fingerprint_methods[fingerprint](mol)    

    def similarity(self, smiles1, smiles2, fingerprint_type=None, similarity_metric=None):
        if similarity_metric == None:
            similarity_metric = "Tanimoto"
        if fingerprint_type == None:
            fingerprint_type = self.fingerprint_type_list[0]
        fgp1 = self.fingerprint_dict['similarity'][fingerprint_type](Chem.MolFromSmiles(smiles1))
        fgp2 = self.fingerprint_dict['similarity'][fingerprint_type](Chem.MolFromSmiles(smiles2))
        return self.similarity_metrics[similarity_metric](fgp1, fgp2)

    def create_database(self,  query_smiles_table, smiles_column='smiles'):
        query_smiles_table['features'] = query_smiles_table[smiles_column].apply(lambda x:self.gen_fingerprints(x, self.fingerprint_type_list[0]))
        return query_smiles_table
    
    def similarity_predict(self, fingerprint_database, ref_smiles, as_pandas=True, similarity_metrics=None):
        if similarity_metrics == None:
            similarity_metrics = "Tanimoto"
        pred_values = {key:[] for key in similarity_metrics}
        ref_fgp = self.fingerprint_methods[self.fingerprint_type_list[0]](Chem.MolFromSmiles(ref_smiles))
        for similarity_metric in similarity_metrics:
            pred_values[similarity_metric] = fingerprint_database['features'].apply(lambda x: self.similarity_metrics[similarity_metric](ref_fgp, x))
        if as_pandas == True:
            res_df = pd.DataFrame(pred_values, columns=similarity_metrics)
            return res_df
        else:
            return pred_values 

    def virtual_screening(self, ref_smiles_list, query_smiles_table, reverse=True, smiles_column='smiles', similarity_metrics=["Tanimoto"]):
        self.fingerprint_methods = self.fingerprint_dict['similarity']
        assert len(self.fingerprint_type_list) == 1, f"ERROR: multiple fingerprint isn't supported for virtual screening now, give {self.fingerprint_type_list}."
        fingerprint_database = self.create_database(query_smiles_table, smiles_column=smiles_column)
        total_res = pd.DataFrame()
        for ref_smiles in ref_smiles_list:
            query_scores = self.similarity_predict(fingerprint_database, ref_smiles, similarity_metrics=similarity_metrics)
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
        return total_res

    def gen_fingerprints_list(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        combine_fgp_list = []
        for fingerprint_type in self.fingerprint_type_list:
            fgp = self.fingerprint_dict['QSAR'][fingerprint_type](mol)
            combine_fgp_list += fgp.ToList()
        return combine_fgp_list

    def extract_fingerprints(self, query_smiles_table, smiles_column='smiles'):
        return pd.DataFrame(query_smiles_table[smiles_column].apply(lambda x:self.gen_fingerprints_list(x)).to_list()).add_prefix('FP_')
    
    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        return pd.DataFrame(query_smiles_table[smiles_column].apply(lambda x:self.gen_fingerprints_list(x)).to_list()).add_prefix('FP_')
