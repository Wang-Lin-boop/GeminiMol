import os
import math
import shutil
import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from rdkit import Chem

def gen_standardize_smiles(smiles, kekule=False, random=False):
    try:
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random, isomericSmiles=True)
        return smiles
    except:
        return smiles

class CrossSimilarity():
    def __init__(self, model_name, bb_path, feature_list=['smiles1','smiles2'], label=['MCS', 'Max', 'Mean', 'Min']):
        self.model_name = model_name
        self.label_list = label
        self.feature_list = feature_list
        torch.set_float32_matmul_precision('high') 
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
            shutil.copytree(bb_path, f'{self.model_name}/backbone')
        else:
            if not os.path.exists(f'{self.model_name}/backbone'):
                shutil.copytree(bb_path, f'{self.model_name}/backbone')

    def load(self, label):
        assert label in self.label_list, f'Error: unknown model {label}'
        if os.path.exists(f'{self.model_name}/{label}/model.ckpt'):
            return MultiModalPredictor.load(path=f'{self.model_name}/{label}')

    def predict(self, df, as_pandas=True):
        res_df = pd.DataFrame()
        res_df.index = df.index
        for label_pred_model in self.label_list:
            query_scores = self.predictor.predict(df, as_pandas=True)
            query_scores = pd.DataFrame(query_scores, columns=[label_pred_model])
            res_df = res_df.join(query_scores, how='left')
        if as_pandas == True:
            return res_df[self.label_list]
        else:
            return res_df.to_dict(orient='list')

    def evaluate(self, df):
        # df = df.reset_index(drop=True)
        pred_table = self.predict(df, as_pandas=True)
        results = pd.DataFrame(index=self.label_list, columns=["RMSE","PEARSONR","SPEARMANR"])
        for label_name in self.label_list:
            pred_score = pred_table[label_name].tolist()
            true_score = df[label_name].tolist()
            results.loc[[label_name],['RMSE']] = round(math.sqrt(mean_squared_error(true_score, pred_score)), 6)
            results.loc[[label_name],['PEARSONR']] = round(pearsonr(pred_score, true_score)[0] , 6)
            results.loc[[label_name],['SPEARMANR']] = round(spearmanr(pred_score, true_score)[0] , 6)
        return results

    def fit(self, label, train_data, val_df=None, epochs=12, learning_rate=5.0e-4, batch_size=96, num_gpus=1):
        self.hyperparameters = {
            "model.hf_text.checkpoint_name": f'{self.model_name}/backbone',
            "model.hf_text.max_text_len" : 514, # 384 or 514
            'optimization.learning_rate': learning_rate, # 1.0e-3 or 5.0e-5
            "optimization.weight_decay": learning_rate, # 1.0e-3 or 1.0e-4
            "optimization.lr_schedule": "cosine_decay",
            "optimization.lr_decay": 0.9,
            "optimization.top_k_average_method": "greedy_soup",
            "optimization.top_k": 3, 
            "optimization.warmup_steps": 0.2, 
            "env.num_gpus": num_gpus,
            "optimization.val_check_interval": 0.1,
            'env.per_gpu_batch_size': batch_size,
            'optimization.max_epochs': epochs,
            "optimization.patience": 10,
            "env.eval_batch_size_ratio": 4,
            "env.num_workers_evaluation": 4,
            "env.batch_size": num_gpus*batch_size
        }
        train_data = pd.DataFrame(train_data, columns=[self.feature_list[0], self.feature_list[1], label]) 
        if val_df is None:
            tuning_data = None
        else:
            tuning_data = pd.DataFrame(val_df, columns=[self.feature_list[0], self.feature_list[1], label])
        if not os.path.exists(f'{self.model_name}/{label}/model.ckpt'):
            self.predictor = MultiModalPredictor(label=label, path=f'{self.model_name}/{label}', eval_metric="r2")
        else:
            self.predictor = self.load(label)
        self.predictor.fit(
            train_data, 
            tuning_data = tuning_data, 
            column_types = {self.feature_list[0]: "text", self.feature_list[1]: "text", label: "numerical"},
            presets = None, # best_quality, high_quality,medium_quality_faster_train
            hyperparameters = self.hyperparameters, 
            seed = 1102,
        )

class CrossEncoder(CrossSimilarity):
    def __init__(self, 
            model_name, 
            candidate_labels = [
                'ShapeScore', 'ShapeOverlap', 'ShapeDistance', 'ShapeAggregation', 
                'CrossSim', 'CrossDist', 'CrossAggregation', 'CrossOverlap', 
                'LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 
                'MCS']
            ):
        self.model_name = model_name
        label_list = []
        for label_pred_model in candidate_labels:
            if os.path.exists(f'{self.model_name}/{label_pred_model}/model.ckpt'):
                label_list += [label_pred_model]
        super().__init__(
            model_name=model_name, 
            bb_path=f'{self.model_name}/backbone', 
            feature_list=['smiles1','smiles2'], 
            label=label_list
        )
        self.similarity_metrics_list = label_list

    def similarity_predict(self, dataset, as_pandas=True, similarity_metrics=None):
        res_df = pd.DataFrame()
        res_df.index = dataset.index
        for label_pred_model in similarity_metrics:
            assert label_pred_model in self.similarity_metrics_list, f"ERROR: CrossShape model for {label_pred_model} not found."
            query_scores = self.load(label_pred_model).predict(dataset, as_pandas=True)
            query_scores = pd.DataFrame(query_scores, columns=[label_pred_model])
            res_df = res_df.join(query_scores, how='left')
        if as_pandas == True:
            return res_df[similarity_metrics]
        else:
            return res_df.to_dict(orient='list')

    def virtual_screening(self, ref_smiles_list, query_smiles_table, reverse=False, smiles_column='smiles', similarity_metrics=None):
        total_res = pd.DataFrame()
        if reverse == False:
            query_smiles_table['smiles1'] = query_smiles_table[smiles_column]
        else:
            query_smiles_table['smiles2'] = query_smiles_table[smiles_column]
        for ref_smiles in ref_smiles_list:
            if reverse == False:
                query_smiles_table['smiles2'] = ref_smiles
            else:
                query_smiles_table['smiles1'] = ref_smiles
            query_scores = self.similarity_predict(query_smiles_table[['smiles1','smiles2']], as_pandas=True, similarity_metrics=similarity_metrics)
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            res = query_smiles_table.join(query_scores, how='left')
            total_res = pd.concat([total_res, res], ignore_index=True)
        return total_res

    def encode(self, smiles_list):
        reshape_data = pd.DataFrame()
        reshape_data['smiles1'], reshape_data['smiles2'] = smiles_list, smiles_list
        features = torch.cat([
            self.load(label_pred_model).extract_embedding(reshape_data, as_tensor=True)
            for label_pred_model in self.similarity_metrics_list
            ], dim=1)
        return features

    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        shape_data = pd.DataFrame()
        shape_data['smiles1'], shape_data['smiles2'] = query_smiles_table[smiles_column].apply(lambda x:gen_standardize_smiles(x, random=False)), query_smiles_table[smiles_column].apply(lambda x:gen_standardize_smiles(x, random=True))
        shape_features = pd.DataFrame()
        shape_features.index = shape_data.index
        for label_pred_model in self.similarity_metrics_list:
            embedding_df = self.load(label_pred_model).extract_embedding(shape_data, as_pandas=True)
            embedding_df = embedding_df.add_prefix(f'CE_{label_pred_model}_')
            shape_features = shape_features.join(embedding_df, how='left')
        return shape_features
