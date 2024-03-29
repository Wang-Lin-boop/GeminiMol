import pandas as pd
import torch
import sys
import os
from model.GeminiMol import GeminiMol
from utils.fingerprint import Fingerprint

if __name__ == "__main__":
    model_path = sys.argv[1]
    model_basename = os.path.basename(model_path)
    encoder_method = sys.argv[2]
    extrnal_data = pd.read_csv(sys.argv[3])
    smiles_column = sys.argv[4]
    ## read the encoder models
    fingerprint_list = []
    encoders = {}
    for method in encoder_method.split(":"):
        if os.path.exists(f'{method}/GeminiMol.pt'):
            method_list = [method]
            from model.GeminiMol import GeminiMol
            encoders[method] = GeminiMol(
                method,
                depth = 0, 
                custom_label = None, 
                extrnal_label_list = ['Cosine', 'Pearson', 'RMSE', 'Manhattan']
            )
        elif os.path.exists(f'{method}/backbone'):
            from model.CrossEncoder import CrossEncoder
            encoders[method] = CrossEncoder(
                method,
                candidate_labels = [
                    'LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 
                    'ShapeScore', 'ShapeOverlap', 'ShapeAggregation', 'CrossSim', 'CrossAggregation', 'CrossOverlap', 
                ]
            )
        elif method == "CombineFP":
            methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
            encoders[method] = Fingerprint(methods_list)
        else:
            methods_list = [method]
            fingerprint_list += [method]
    if len(fingerprint_list) > 0:
        encoders['Fingerprints'] = Fingerprint(fingerprint_list)
    encoders_list = list(encoders.values())
    ## load the model
    print('NOTE: loading models...')
    if os.path.exists(f"{model_path}/predictor.pt"): # FineTuning
        from FineTuning import GeminiMolQSAR
        if len(encoders_list) == 1 and isinstance(encoders_list[0], GeminiMol):
            encoder = encoders_list[0]
            QSAR_model = GeminiMolQSAR(
                geminimol_encoder = encoder, 
                model_name = model_path,
                standardize = False, 
                smiles_column = smiles_column, 
            )
            predicted_res = QSAR_model.predict(extrnal_data)
        else:
            raise RuntimeError('NOTE: FineTuning only supports one GeminiMol encoder!')
    elif os.path.exists(f"{model_path}/predictor.pkl"): # AutoQSAR
        from AutoQSAR import AutoQSAR
        from utils.fingerprint import Fingerprint
        encoders_list = list(encoders.values())
        QSAR_model = AutoQSAR(
            model_name = model_path, 
            encoder_list = encoders_list,
            standardize = True, 
            smiles_column = smiles_column,
            )
        predicted_res = QSAR_model.predict(extrnal_data, model="WeightedEnsemble_L2_FULL")
    ## output the results
    predicted_res.to_csv(f"{model_basename}_prediction.csv", index=False, header=True, sep=',')
    print(f'NOTE: job completed! check {model_basename}_prediction.csv for results!')