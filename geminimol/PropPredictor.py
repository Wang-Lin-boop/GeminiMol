import pandas as pd
import torch
import sys
import os
from model.GeminiMol import GeminiMol, PropDecoder

predict_mode = sys.argv[1]
product_model_name = sys.argv[2]
extrnal_data = sys.argv[3]
    
if predict_mode == "FineTuning":
    import json
    from FineTuning import GeminiMolQSAR
    from model.GeminiMol import GeminiMol
    encoder = GeminiMol(
        model_name = "/public/home/wangshh2022/project/GeminiMol/encoders_models/GeminiMol_M5086", 
        custom_label = None, 
        extrnal_label_list = ['Cosine', 'Pearson', 'RMSE', 'Manhattan']
        )
    gemini_model = GeminiMolQSAR(geminimol_encoder= encoder, model_name=product_model_name)
    if os.path.exists(f"{product_model_name}/predictor.pt"):
        with open(f"{product_model_name}/model_params.json", 'r', encoding='utf-8') as f:
            gemini_model.params = json.load(f)
        gemini_model.load_state_dict(torch.load(f"{product_model_name}/predictor.pt"))
        gemini_model.eval()
        
    data = pd.read_csv(extrnal_data)     
    predicted_res = gemini_model.predict(data)
    predicted_res.to_csv(f"{predict_mode}_prediction_results.csv", index=False, header=True, sep=',')
    
elif predict_mode == "AutoQSAR":
    from AutoQSAR import AutoQSAR
    from autogluon.tabular import TabularPredictor
    from utils.fingerprint import Fingerprint
    encoders = {}
    method = "CombineFP"
    methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
    encoders[method] = Fingerprint(methods_list)
    encoders_list = list(encoders.values())
    auto_model = AutoQSAR(
        model_name=product_model_name, 
        encoder_list = encoders_list
        )
    if os.path.exists(f"{product_model_name}/predictor.pkl"):
        auto_model.QSAR_Model = TabularPredictor.load(path=product_model_name, verbosity=4)
        
    data = pd.read_csv(extrnal_data)    
    predicted_res = auto_model.predict(data, model="WeightedEnsemble_L2_FULL")
    predicted_res.to_csv(f"{predict_mode}_prediction_results.csv", index=False, header=True, sep=',')
        