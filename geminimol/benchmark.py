import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import oddt.metrics as vsmetrics
from functools import partial
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, average_precision_score
from scipy.stats import pearsonr, spearmanr
from utils.chem import gen_standardize_smiles, check_smiles_validity, is_valid_smiles

def load_molecular_representation(model_name):
    # load model
    if ':' in model_name:
        custom_label_list = model_name.split(':')[1].split(',')
        model_name = model_name.split(':')[0]
    else:
        custom_label_list = None
    if os.path.exists(f'{model_name}/GeminiMol.pt'):
        from model.GeminiMol import GeminiMol
        predictor = GeminiMol(
            model_name, 
            custom_label = custom_label_list, 
            extrnal_label_list = [
                'RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson'
            ]
            )
        model_name = str(model_name.split('/')[-1])
    elif os.path.exists(f'{model_name}/backbone'):
        from model.CrossEncoder import CrossEncoder
        if custom_label_list is None:
            candidate_labels = [
                    'LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 
                    'ShapeScore', 'ShapeOverlap', 'ShapeAggregation', 'CrossSim', 'CrossAggregation', 'CrossOverlap', 
                ]
        else:
            candidate_labels = custom_label_list
        predictor = CrossEncoder(
            model_name,
            candidate_labels = candidate_labels
        )
        model_name = str(model_name.split('/')[-1])
    elif model_name == "CombineFP":
        model_name = "CombineFP"
        method_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
        from utils.fingerprint import Fingerprint
        predictor = Fingerprint(method_list)
    else:
        from utils.fingerprint import Fingerprint
        predictor = Fingerprint([model_name])
    return predictor

class Benchmark():
    '''
    Benchmark for virtual screening, target identification, and QSAR.

    Parameters:
        > model_name (str): the name of the predictor.
        > record (bool): whether to record the prediction results.
        > data_record (bool): whether to record the prediction data.

    Attributes:
        > predictor (object): the predictor for virtual screening, target identification, and QSAR.
        > model_name (str): the name of the predictor.
        > record (bool): whether to record the prediction results.
        > data_record (bool): whether to record the prediction data.
        > statistics_metrics_dict (dict): the dictionary of statistics metrics for virtual screening, target identification, and QSAR.
        > metric_functions (dict): the dictionary of metric functions for virtual screening, target identification, and QSAR.
        > label_map (dict): the dictionary of label mapping for virtual screening, target identification, and QSAR.

    Methods:
        > ```prepare(dataset, smiles_column='smiles')```: prepare the dataset for virtual screening, target identification, and QSAR.
        > ```read_smi(smi_file)```: read the smiles file for virtual screening, target identification, and QSAR.
        > ```statistics(total_pred_res, statistics_metrics_list=None, score_name='score', ascending=False, state_name='state', duplicate_column='smiles')```: calculate the statistics for virtual screening, target identification, and QSAR.
        > ```vritual_screening_on_target (target, ref_smiles_list, query_smiles_state_table, statistics_metrics_list, reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles')````: virtual screening on target for virtual screening, target identification, and QSAR.
        > ```reporting_benchmark(statistic_tables)```: reporting the benchmark for virtual screening, target identification, and QSAR.

    '''
    def __init__(self, model_name, record = True, data_record = False):
        '''
        Parameters:
            > predictor (object): 
                the predictor must have two methods, ```virtual_screening(dataset, ref_smiles, reverse=False, smiles_column='smiles', similarity_metrics=None)``` and ```extract_features(query_smiles_table, smiles_column='smiles')```.
            > model_name (str): the name of the predictor.
            > record (bool): whether to record the prediction results.
            > data_record (bool): whether to record the prediction data.

        '''
        self.model_name = model_name
        self.record = record
        self.data_record = data_record
        self.statistics_metrics_dict = {
            'ranking': ['AUROC', 'BEDROC', 'EF1%', 'EF0.5%', 'EF0.1%', 'EF0.05%', 'EF0.01%', 'logAUC', 'AUPRC'], 
            'classification': ['AUROC', 'ACC', 'specificity', 'precision', 'recall', 'sensitivity', 'f1', 'AUPRC'],
            'regression': ['RMSE', 'MSE', 'MAE', 'SPEARMANR', 'PEARSONR'], 
        } 
        self.metric_functions = {
            'AUROC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
            'AUPRC': lambda y_true, y_pred: average_precision_score(y_true, y_pred),
            'BEDROC': lambda y_true, y_pred: vsmetrics.bedroc(y_true, y_pred, alpha=160.9, pos_label=1),
            'EF1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=1, pos_label=1, kind='fold'),
            'EF0.5%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.5, pos_label=1, kind='fold'),
            'EF0.1%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.1, pos_label=1, kind='fold'),
            'EF0.05%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.05, pos_label=1, kind='fold'),
            'EF0.01%': lambda y_true, y_pred: vsmetrics.enrichment_factor(y_true, y_pred, percentage=0.01, pos_label=1, kind='fold'),
            'logAUC': lambda y_true, y_pred: vsmetrics.roc_log_auc(y_true, y_pred, pos_label=1, ascending_score=False, log_min=0.001, log_max=1.0),
            'MSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            'MAE': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'SPEARMANR': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
            'PEARSONR': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
            'ACC': lambda y_true, y_pred: accuracy_score(y_true, [round(num) for num in y_pred]),
            'specificity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred], pos_label=0), 
            'precision': lambda y_true, y_pred: precision_score(y_true, [round(num) for num in y_pred]), 
            'recall': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
            'sensitivity': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
            'f1': lambda y_true, y_pred: f1_score(y_true, [round(num) for num in y_pred]), 
        }
        self.label_map = {
            'Active': 1, 
            'Inactive': 0, 
            'active': 1, 
            'inactive': 0, 
            'Yes': 1, 
            'No': 0, 
            'yes': 1, 
            'no': 0, 
            'True': 1, 
            'False': 0, 
            'true': 1, 
            'false': 0, 
            'Positive': 1, 
            'Negative': 0, 
            'positive': 1, 
            'negative': 0, 
            1: 1, 
            0: 0
        }

    def prepare(
            self, 
            dataset, 
            smiles_column='smiles'
        ):
        if self.standardize == True:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x, kekule=False))
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:check_smiles_validity(x))
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset.reset_index(drop=True)

    def read_smi(self, smi_file):
        query_smiles = pd.read_csv(smi_file,header=None,usecols=[0],sep='\s|,|;|\t| ',engine='python')
        query_smiles.columns = ['smiles']
        print(f"Note: The input raw table {smi_file} has {len(query_smiles['smiles'])} rows.")
        query_smiles = self.prepare(query_smiles)
        print(f"Note:  The processed input table {smi_file} has {len(query_smiles['smiles'])} rows.")
        return query_smiles
    
    def statistics(
            self, 
            total_pred_res, 
            statistics_metrics_list=None, 
            score_name='score', 
            ascending=False, 
            state_name='state', 
            duplicate_column='smiles'
        ):
        total_pred_res.sort_values(
            score_name, 
            ascending = ascending, 
            inplace = True,
            ignore_index = True
        )
        total_pred_res.drop_duplicates(
            subset = [duplicate_column], 
            keep = 'first', 
            inplace = True,
            ignore_index = True
        )
        y_pred = np.array(total_pred_res[score_name].to_list())
        y_true = np.array(total_pred_res[state_name].to_list())
        statistic_results = {}
        if statistics_metrics_list is None:
            statistics_metrics_list = list(self.metric_functions.keys())
        for metric in statistics_metrics_list:
            statistic_results[metric] = self.metric_functions[metric](y_true, y_pred)
        return statistic_results
    
    def vritual_screening_on_target(
            self, 
            target, 
            ref_smiles_list, 
            query_smiles_state_table, 
            statistics_metrics_list, 
            reverse=False, 
            smiles_column='smiles', 
            state_name='state', 
            duplicate_column='smiles'
        ):
        if os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}_data.csv"):
            pred_res = pd.read_csv(f"{self.model_name}/{self.benchmark_name}/{target}_data.csv")
        else:
            pred_res = self.predictor.virtual_screening(ref_smiles_list, query_smiles_state_table, reverse=reverse, smiles_column=smiles_column, similarity_metrics=self.predictor.similarity_metrics_list)
            if self.record and self.data_record:
                pred_res.to_csv(f"{self.model_name}/{self.benchmark_name}/{target}_data.csv", index=False, header=True, sep=',')
        # statistic_table (score_types/similarity_metrics, metrics)
        statistic_table = pd.DataFrame(
            columns=statistics_metrics_list, 
            index=self.predictor.similarity_metrics_list
        )
        for score_type in self.predictor.similarity_metrics_list:
            statistic_results = self.statistics(pred_res, statistics_metrics_list=statistics_metrics_list, duplicate_column=duplicate_column, score_name=score_type, state_name=state_name)
            for statistics_metric, value in statistic_results.items():
                statistic_table.loc[[score_type],[statistics_metric]] = value
        if self.record:
            statistic_table.to_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index=True, header=True, sep=',')
        return statistic_table
    
    def reporting_benchmark(self, statistic_tables):
        # benchmark_results (score_types/similarity_metrics, metrics)
        benchmark_results = pd.concat(statistic_tables.values()).groupby(level=0).mean().groupby(axis=1, level=0).mean()
        benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_final_statistics.csv", index=True, header=True, sep=',')
        return benchmark_results

    def read_DUDE(self, target):
        active_smiles = self.read_smi(str(self.data_path+"/"+target+"/actives_final.smi"))
        decoys_smiles = self.read_smi(str(self.data_path+"/"+target+"/decoys_final.smi"))
        active_smiles['state'] = 1
        decoys_smiles['state'] = 0
        query_smiles_state_table = pd.concat([active_smiles, decoys_smiles], ignore_index=True)
        return query_smiles_state_table

    def DUDE_VS(
            self, 
            benchmark_task_type="ranking", 
            standardize=False
        ):
        self.predictor = load_molecular_representation(self.model_name)
        self.standardize = standardize
        self.data_table = pd.read_csv(f"{self.data_path}/DUDE-smiles.csv")
        self.target_dict = dict(zip(self.data_table['Title'], self.data_table['SMILES']))
        # statistic_tables { target : statistic_table (score_types/similarity_metrics, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.target_dict.keys()}
        for target, ref_smiles in self.target_dict.items():
            if os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index_col=0)
            else:
                query_smiles_state_table = self.read_DUDE(target)
                self.statistic_tables[target] = self.vritual_screening_on_target(target, [ref_smiles], query_smiles_state_table, self.statistics_metrics_dict[benchmark_task_type], reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles')
        return self.reporting_benchmark(self.statistic_tables)

    def read_LITPCBA(self, target):
        ref_smiles_list = self.read_smi(str(self.data_path+"/"+target+"/"+"ref.smi"))['smiles'].to_list()
        for smiles in ref_smiles_list:
            if not is_valid_smiles(smiles):
                raise RuntimeError(f"{target}, {smiles}")
        active_smiles = self.read_smi(str(self.data_path+"/"+target+"/actives.smi"))
        decoys_smiles = self.read_smi(str(self.data_path+"/"+target+"/inactives.smi"))
        active_smiles['state'] = 1
        decoys_smiles['state'] = 0
        query_smiles_state_table = pd.concat([active_smiles, decoys_smiles], ignore_index=True)
        return query_smiles_state_table, ref_smiles_list

    def LITPCBA_VS(self, 
            target_list = [ "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",  "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"], 
            benchmark_task_type="ranking", 
            standardize=False
        ):
        self.predictor = load_molecular_representation(self.model_name)
        self.target_list = target_list
        self.standardize = standardize
        # statistic_tables { target : statistic_table (score_types/similarity_metrics, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.target_list}
        for target in self.target_list:
            if os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index_col=0)
            else:
                query_smiles_state_table, ref_smiles_list = self.read_LITPCBA(target)
                self.statistic_tables[target] = self.vritual_screening_on_target(target, ref_smiles_list, query_smiles_state_table, self.statistics_metrics_dict[benchmark_task_type], reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles')
        return self.reporting_benchmark(self.statistic_tables)

    def read_BindingDB(self, target):
        binding_data_table = pd.read_csv(f"{self.data_path}/{target}_DATA.csv", on_bad_lines='skip', dtype={'Monomer_ID':str, 'Ligand_SMILES':str, 'Binding':str, 'Targets':str, 'state_label':int})
        binding_data_table = self.prepare(binding_data_table[['Monomer_ID', 'Ligand_SMILES', 'Binding', 'Targets', 'state_label']], smiles_column='Ligand_SMILES')
        return binding_data_table

    def BindingDB_TargetID(
            self, 
            decoy_list=None, 
            index="BindingDB", 
            benchmark_task_type="ranking", 
            standardize=False
        ):
        self.predictor = load_molecular_representation(self.model_name)
        self.data_table = pd.read_csv(f"{self.data_path}/{index}_Benchmark_Decoys.csv", dtype={'SMILES':str, 'Title':str, 'Number_of_Target':int})
        self.standardize = standardize  ## if you wanna to use new decoys, please standardize the decoy smiles
        self.data_table = self.prepare(self.data_table, smiles_column='SMILES')
        if decoy_list is not None:
            self.decoy_list = decoy_list
        else:
            self.decoy_list = self.data_table['Title'].to_list()
        self.target_dict = dict(zip(self.data_table['Title'], self.data_table['SMILES']))
        self.target_number_dict = dict(zip(self.data_table['Title'], self.data_table['Number_of_Target']))
        # statistic_tables { target : statistic_table (score_types/similarity_metrics, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.decoy_list}
        for target in self.decoy_list:
            ref_smiles = self.target_dict[target]
            number_of_targets = self.target_number_dict[target]
            if os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index_col=0)
            else:
                binding_data_table = self.read_BindingDB(f"{target}_{number_of_targets}")
                self.statistic_tables[target] = self.vritual_screening_on_target(target, [ref_smiles], binding_data_table, self.statistics_metrics_dict[benchmark_task_type], reverse=True, smiles_column='Ligand_SMILES', state_name='state_label', duplicate_column='Targets')
        return self.reporting_benchmark(self.statistic_tables)

    def QSAR(self, 
            target_list=None, 
            smiles_column='SMILES', 
            label_column='Label', 
            standardize=False, 
            benchmark_task_type="classification"
        ):
        from AutoQSAR import AutoQSAR
        self.predictor = load_molecular_representation(self.model_name)
        self.target_list = target_list
        # statistic_tables { target : statistic_table (models, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.target_list}
        # benchmark_results (targets, metrics)
        benchmark_results = pd.DataFrame(columns=['model']+self.statistics_metrics_dict[benchmark_task_type], index=self.target_list)
        for target in self.target_list:
            print(f'NOTE: benchmarking on the {target}....')
            training_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_train.csv')
            training_data.dropna(subset=[smiles_column, label_column], inplace=True)
            training_data[label_column] = training_data[label_column].replace(self.label_map)
            val_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_valid.csv')
            val_data.dropna(subset=[smiles_column, label_column], inplace=True)
            val_data[label_column] = val_data[label_column].replace(self.label_map)
            test_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_test.csv')
            test_data.dropna(subset=[smiles_column, label_column], inplace=True)
            test_data[label_column] = test_data[label_column].replace(self.label_map)
            QSAR_task_type = 'binary' if benchmark_task_type in ['ranking', 'classification'] else 'regression'
            if benchmark_task_type == 'classification':
                recommended_metrics = 'AUROC'
            elif benchmark_task_type == 'ranking':
                recommended_metrics = 'BEDROC'
            else:
                recommended_metrics = 'SPEARMANR'
            self.QSAR_model = AutoQSAR(
                f"{self.model_name}/{self.benchmark_name}/{target}", 
                encoder_list = [self.predictor], 
                standardize = standardize, 
                label_column=label_column, 
                smiles_column=smiles_column, 
                task_type=QSAR_task_type
            )
            if not os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}/predictor.pkl"):
                self.QSAR_model.trianing_models(
                    training_data, 
                    val_set = val_data,
                    stack = False, 
                    num_trials = 1,
                    model_list = ['GBM', 'NN_TORCH', 'CAT', 'XGB']
                    )
            leaderborad = self.QSAR_model.evaluate(
                val_data, 
                model_list = 'all', 
                metric_list = self.statistics_metrics_dict[benchmark_task_type]
            )
            best_model = leaderborad[recommended_metrics].idxmax()
            benchmark_results.loc[[target],['model']] = best_model
            predicted_results = self.QSAR_model.predict(test_data, model=best_model) 
            statistic_results = self.statistics(predicted_results, statistics_metrics_list=self.statistics_metrics_dict[benchmark_task_type], duplicate_column=smiles_column, score_name='pred', state_name=label_column)
            for statistics_metric, value in statistic_results.items():
                benchmark_results.loc[[target],[statistics_metric]] = value
            self.statistic_tables[target] = self.QSAR_model.evaluate(
                test_data, 
                model_list = 'all', 
                metric_list = self.statistics_metrics_dict[benchmark_task_type]
            )
            if self.record:
                self.statistic_tables[target].to_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index=True, header=True, sep=',')
        benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_each_target.csv", index=True, header=True, sep=',')
        return self.reporting_benchmark(self.statistic_tables)

    def PropDecoder(
            self,
            target_list=None, 
            candidate_smiles_columns=['SMILES', 'Drug', 'smiles', 'compound', 'molecule'], 
            candidate_label_columns=['Label', 'label', 'Y', 'Target', 'target'], 
            standardize=False, 
        ):
        from PropDecoder import QSAR
        self.predictor = load_molecular_representation(self.model_name)
        self.target_list = target_list
        # benchmark_results (targets, metrics)
        benchmark_results = pd.DataFrame(columns=['model', 'model_score', 'test_metrics'], index=self.target_list)
        for target in self.target_list:
            print(f'NOTE: benchmarking on the {target}....')
            training_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_train.csv')
            for candidate_smiles_column in candidate_smiles_columns:
                if candidate_smiles_column in training_data.columns:
                    smiles_column = candidate_smiles_column
                    break
            for candidate_label_column in candidate_label_columns:
                if candidate_label_column in training_data.columns:
                    label_column = candidate_label_column
                    break
            label_set = list(set(training_data[label_column].to_list()))
            if len(label_set) == 2:
                task_type = 'binary'
                test_metrics = 'AUROC'
                benchmark_task_type = 'classification'
            else:
                task_type = 'regression'
                test_metrics = 'SPEARMANR'
                benchmark_task_type = 'regression'
            training_data.dropna(subset=[smiles_column, label_column], inplace=True)
            training_data[label_column] = training_data[label_column].replace(self.label_map)
            val_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_valid.csv')
            val_data.dropna(subset=[smiles_column, label_column], inplace=True)
            val_data[label_column] = val_data[label_column].replace(self.label_map)
            test_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_test.csv')
            test_data.dropna(subset=[smiles_column, label_column], inplace=True)
            test_data[label_column] = test_data[label_column].replace(self.label_map)
            QSAR_model = QSAR(  
                f"{self.model_name}/{self.benchmark_name}/{target}", 
                encoder_list = [self.predictor], 
                standardize = standardize, 
                label_column = label_column, 
                smiles_column = smiles_column, 
            )
            if not os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}/predictor.pt"):
                epochs = ( 300000 // len(training_data) ) + 1
                if len(training_data) > 30000:
                    batch_size, learning_rate, patience = 256, 1.0e-3, 50
                    expand_ratio, hidden_dim, num_layers = 3, 2048, 5
                elif len(training_data) > 10000:
                    batch_size, learning_rate, patience = 128, 5.0e-4, 60
                    expand_ratio, hidden_dim, num_layers = 2, 2048, 4
                elif len(training_data) > 5000:
                    batch_size, learning_rate, patience = 64, 1.0e-4, 80
                    expand_ratio, hidden_dim, num_layers = 1, 1024, 3
                elif len(training_data) > 2000:
                    batch_size, learning_rate, patience = 32, 5.0e-5, 100
                    expand_ratio, hidden_dim, num_layers = 0, 1024, 3
                else:
                    batch_size, learning_rate, patience = 24, 1.0e-5, 100
                    expand_ratio, hidden_dim, num_layers = 0, 1024, 3
                if task_type == 'binary':
                    dropout_rate = 0.3
                    dense_dropout = 0.1
                    dense_activation = 'Softplus' # GELU
                    projection_activation = 'Softplus' # GELU
                    projection_transform = 'Sigmoid'
                elif task_type == 'regression':
                    dropout_rate = 0.1
                    dense_dropout = 0.0
                    dense_activation = 'ELU' # ELU
                    projection_activation = 'Identity' # ELU
                    if training_data[label_column].max() <= 1.0 and training_data[label_column].min() >= 0.0:
                        projection_transform = 'Sigmoid'
                    else:
                        projection_transform = 'Identity'
                QSAR_model.trianing_models(
                    training_data,
                    val_set = val_data,
                    epochs = epochs,
                    learning_rate = learning_rate,
                    params = {
                        'task_type': task_type,
                        'hidden_dim': hidden_dim,
                        'expand_ratio': expand_ratio,
                        'dense_dropout': dense_dropout,
                        'dropout_rate': dropout_rate,
                        'num_layers': num_layers,
                        'rectifier_activation': 'SiLU',
                        'concentrate_activation': 'SiLU',
                        'dense_activation': dense_activation,
                        'projection_activation': projection_activation,
                        'projection_transform': projection_transform,
                        'linear_projection': False,
                        'batch_size': batch_size
                    },
                    patience = patience
                )
            result = QSAR_model.evaluate(
                test_data, 
                smiles_name = smiles_column, 
                label_name = label_column,
                metrics = self.statistics_metrics_dict[benchmark_task_type],
                as_pandas= False
                )
            benchmark_results.loc[[target],['model']] = 'PropDecoder'
            benchmark_results.loc[[target],['model_score']] = result[test_metrics]
            benchmark_results.loc[[target],['test_metrics']] = test_metrics
            target_res = pd.DataFrame(result, index=['Test'])
            target_res.to_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index=True, header=True, sep=',')
        benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_each_target.csv", index=True, header=True, sep=',')
        return benchmark_results

    def FineTuning(
            self,
            target_list=None, 
            candidate_smiles_columns=['SMILES', 'Drug', 'smiles', 'compound', 'molecule'], 
            candidate_label_columns=['Label', 'label', 'Y', 'Target', 'target'], 
            standardize=False, 
        ):
        from FineTuning import GeminiMolQSAR
        self.target_list = target_list
        # benchmark_results (targets, metrics)
        benchmark_results = pd.DataFrame(columns=['model', 'model_score', 'test_metrics'], index=self.target_list)
        for target in self.target_list:
            print(f'NOTE: benchmarking on the {target}....')
            training_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_train.csv')
            for candidate_smiles_column in candidate_smiles_columns:
                if candidate_smiles_column in training_data.columns:
                    smiles_column = candidate_smiles_column
                    break
            for candidate_label_column in candidate_label_columns:
                if candidate_label_column in training_data.columns:
                    label_column = candidate_label_column
                    break
            label_set = list(set(training_data[label_column].to_list()))
            if len(label_set) == 2:
                task_type = 'binary'
                test_metrics = 'AUROC'
                benchmark_task_type = 'classification'
            else:
                task_type = 'regression'
                test_metrics = 'SPEARMANR'
                benchmark_task_type = 'regression'
            training_data.dropna(subset=[smiles_column, label_column], inplace=True)
            training_data[label_column] = training_data[label_column].replace(self.label_map)
            val_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_valid.csv')
            val_data.dropna(subset=[smiles_column, label_column], inplace=True)
            val_data[label_column] = val_data[label_column].replace(self.label_map)
            test_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_test.csv')
            test_data.dropna(subset=[smiles_column, label_column], inplace=True)
            test_data[label_column] = test_data[label_column].replace(self.label_map)
            if not os.path.exists(f"{self.model_name}/{self.benchmark_name}/{target}/predictor.pt"):
                epochs = ( 300000 // len(training_data) ) + 1
                if len(training_data) > 30000:
                    batch_size, learning_rate, patience = 256, 1.0e-3, 50
                    expand_ratio, hidden_dim, num_layers = 3, 2048, 5
                elif len(training_data) > 10000:
                    batch_size, learning_rate, patience = 128, 5.0e-4, 60
                    expand_ratio, hidden_dim, num_layers = 2, 2048, 4
                elif len(training_data) > 5000:
                    batch_size, learning_rate, patience = 64, 1.0e-4, 80
                    expand_ratio, hidden_dim, num_layers = 1, 1024, 3
                elif len(training_data) > 2000:
                    batch_size, learning_rate, patience = 32, 5.0e-5, 100
                    expand_ratio, hidden_dim, num_layers = 0, 1024, 3
                else:
                    batch_size, learning_rate, patience = 24, 1.0e-5, 100
                    expand_ratio, hidden_dim, num_layers = 0, 1024, 3
                if task_type == 'binary':
                    dropout_rate = 0.3 
                    dense_dropout = 0.1 
                    dense_activation = 'Softplus' # GELU
                    projection_activation = 'Softplus' # GELU
                    projection_transform = 'Sigmoid'
                elif task_type == 'regression':
                    dropout_rate = 0.1
                    dense_dropout = 0.0
                    dense_activation = 'ELU' # ELU
                    projection_activation = 'Identity' # ELU
                    if training_data[label_column].max() <= 1.0 and training_data[label_column].min() >= 0.0:
                        projection_transform = 'Sigmoid'
                    else:
                        projection_transform = 'Identity'
                QSAR_model = GeminiMolQSAR(  
                    model_name = f"{self.model_name}/{self.benchmark_name}/{target}", 
                    geminimol_encoder = load_molecular_representation(self.model_name), 
                    standardize = standardize, 
                    label_column = label_column, 
                    smiles_column = smiles_column, 
                    params = {
                        'task_type': task_type,
                        'hidden_dim': hidden_dim,
                        'expand_ratio': expand_ratio,
                        'dense_dropout': dense_dropout,
                        'dropout_rate': dropout_rate,
                        'num_layers': num_layers,
                        'rectifier_activation': 'SiLU',
                        'concentrate_activation': 'SiLU',
                        'dense_activation': dense_activation,
                        'projection_activation': projection_activation,
                        'projection_transform': projection_transform,
                        'linear_projection': False,
                        'batch_size': batch_size
                    }
                )
                QSAR_model.trianing_models(
                    training_data,
                    val_set = val_data,
                    epochs = epochs,
                    learning_rate = learning_rate,
                    patience = patience
                )
            else:
                QSAR_model = GeminiMolQSAR(
                    model_name = f"{self.model_name}/{self.benchmark_name}/{target}",
                    geminimol_encoder = load_molecular_representation(self.model_name),
                    standardize = False, 
                    label_column = label_column, 
                    smiles_column = smiles_column, 
                    params = None
                )
            result = QSAR_model.evaluate(
                test_data, 
                smiles_name = smiles_column, 
                label_name = label_column,
                metrics = self.statistics_metrics_dict[benchmark_task_type],
                as_pandas= False
                )
            benchmark_results.loc[[target],['model']] = 'FineTuning'
            benchmark_results.loc[[target],['model_score']] = result[test_metrics]
            benchmark_results.loc[[target],['test_metrics']] = test_metrics
            target_res = pd.DataFrame(result, index=['Test'])
            target_res.to_csv(f"{self.model_name}/{self.benchmark_name}/{target}_statistics.csv", index=True, header=True, sep=',')
        benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_each_target.csv", index=True, header=True, sep=',')
        return benchmark_results

    def __call__(self, benchmark_name, data_path, standardize=False):
        self.benchmark_name = benchmark_name
        self.data_path = data_path
        ## please set the standardize to True if you add some new datasets
        if not os.path.exists(f"{self.model_name}"):
            os.mkdir(f"{self.model_name}")
        if not os.path.exists(f"{self.model_name}/{self.benchmark_name}/"):
            os.mkdir(f"{self.model_name}/{self.benchmark_name}/")
        benchmark_functions = {
            'DUDE': partial(
                self.DUDE_VS, 
                benchmark_task_type="ranking", 
                standardize=standardize
            ),
            'LIT-PCBA': partial(
                self.LITPCBA_VS, 
                benchmark_task_type="ranking", 
                standardize = standardize 
            ),
            'TIBD': partial(
                self.BindingDB_TargetID, 
                index="TIBD", 
                benchmark_task_type="ranking", 
                standardize=standardize
            ),
            'PropDecoder-ADMET': partial(
                self.PropDecoder,
                target_list=[
                    'Bioavailability_Ma', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'hERG_Karim', 'AMES', 'DILI', 'SkinReaction', 'Carcinogens_Lagunin', 'ClinTox', 'hERG_inhib', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'PAMPA_NCATS', 'AddictedChem',
                    'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith','Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu', 'HydrationFreeEnergy_FreeSolv', 'hERG_at_1uM', 'hERG_at_10uM'
                ], 
                standardize = standardize,
            ),
            'PropDecoder-QSAR': partial(
                self.PropDecoder,
                target_list=[
                    "NCI_786-0","NCI_BT-549","NCI_DMS114","NCI-H23", "NCI_HCT-116","NCI_HOP-92","NCI_KM12","NCI_M19-MEL","NCI_MDA-N","NCI_OVCAR-8","NCI_RXF393","NCI_SK-MEL-2","NCI_SN12K1","NCI_SW-620","NCI_UACC-62","NCI_A498","NCI_CAKI-1","NCI_DMS273","NCI-H322M","NCI_HCT-15","NCI_HS578T","NCI_KM20L2","NCI_MALME-3M","NCI_MOLT-4","NCI_P388","NCI_RXF-631","NCI_SK-MEL-28","NCI_SNB-19","NCI_T-47D","NCI_UO-31","NCI_A549-ATCC","NCI_CCRF-CEM","NCI_DU-145","NCI-H460","NCI_HL-60(TB)","NCI_HT29","NCI_LOX-IMVI","NCI_MCF7","NCI_OVCAR-3","NCI_P388-ADR","NCI_SF-268","NCI_SK-MEL-5","NCI_SNB-75","NCI_TK-10","NCI_XF498","NCI_ACHN","NCI_COLO205","NCI_EKVX","NCI-H522","NCI_HOP-18","NCI_IGROV1","NCI_LXFL529","NCI_MDA-MB-231-ATCC","NCI_OVCAR-4","NCI_PC-3","NCI_SF-295","NCI_SK-OV-3","NCI_SNB-78","NCI_U251","NCI-ADR-RES","NCI_DLD-1","NCI-H226","NCI_HCC-2998","NCI_HOP-62","NCI_K-562","NCI_M14","NCI_MDA-MB-435","NCI_OVCAR-5","NCI_RPMI-8226","NCI_SF-539","NCI_SN12C","NCI_SR","NCI_UACC-257", "ALDH1", "FEN1", "GBA", "MAPK1", "PKM2", "KAT2A", "VDR", "RGS12", "HADH2", "HSD17B4", "CRF-R2-antagonists", "CRF-R2-agonist", "CBFb-RUNX1", "Rango", "ARNT-TAC3", "Vif-APOBEC3G", "A1Apoptosis", "Peg3", "NF-kB","uPA","PINK1"
                ], 
                standardize=standardize,
            ),
            'FineTuning-ADMET': partial(
                self.FineTuning,
                target_list=[
                    'Bioavailability_Ma', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'hERG_Karim', 'AMES', 'DILI', 'SkinReaction', 'Carcinogens_Lagunin', 'ClinTox', 'hERG_inhib', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'PAMPA_NCATS', 'AddictedChem',
                    'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith','Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu', 'HydrationFreeEnergy_FreeSolv', 'hERG_at_1uM', 'hERG_at_10uM'
                ], 
                standardize = standardize,
            ),
            'FineTuning-QSAR': partial(
                self.FineTuning,
                target_list=[
                    "NCI_786-0","NCI_BT-549","NCI_DMS114","NCI-H23", "NCI_HCT-116","NCI_HOP-92","NCI_KM12","NCI_M19-MEL","NCI_MDA-N","NCI_OVCAR-8","NCI_RXF393","NCI_SK-MEL-2","NCI_SN12K1","NCI_SW-620","NCI_UACC-62","NCI_A498","NCI_CAKI-1","NCI_DMS273","NCI-H322M","NCI_HCT-15","NCI_HS578T","NCI_KM20L2","NCI_MALME-3M","NCI_MOLT-4","NCI_P388","NCI_RXF-631","NCI_SK-MEL-28","NCI_SNB-19","NCI_T-47D","NCI_UO-31","NCI_A549-ATCC","NCI_CCRF-CEM","NCI_DU-145","NCI-H460","NCI_HL-60(TB)","NCI_HT29","NCI_LOX-IMVI","NCI_MCF7","NCI_OVCAR-3","NCI_P388-ADR","NCI_SF-268","NCI_SK-MEL-5","NCI_SNB-75","NCI_TK-10","NCI_XF498","NCI_ACHN","NCI_COLO205","NCI_EKVX","NCI-H522","NCI_HOP-18","NCI_IGROV1","NCI_LXFL529","NCI_MDA-MB-231-ATCC","NCI_OVCAR-4","NCI_PC-3","NCI_SF-295","NCI_SK-OV-3","NCI_SNB-78","NCI_U251","NCI-ADR-RES","NCI_DLD-1","NCI-H226","NCI_HCC-2998","NCI_HOP-62","NCI_K-562","NCI_M14","NCI_MDA-MB-435","NCI_OVCAR-5","NCI_RPMI-8226","NCI_SF-539","NCI_SN12C","NCI_SR","NCI_UACC-257", "ALDH1", "FEN1", "GBA", "MAPK1", "PKM2", "KAT2A", "VDR", "RGS12", "HADH2", "HSD17B4", "CRF-R2-antagonists", "CRF-R2-agonist", "CBFb-RUNX1", "Rango", "ARNT-TAC3", "Vif-APOBEC3G", "A1Apoptosis", "Peg3", "NF-kB","uPA","PINK1"
                ], 
                standardize=standardize,
            ),
            'ADMET-C': partial(
                self.QSAR, 
                target_list=[
                    'Bioavailability_Ma', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI', 'SkinReaction', 'Carcinogens_Lagunin', 'ClinTox', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'PAMPA_NCATS', 'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'hERG_Karim', 'hERG_inhib', 'AddictedChem'
                ], 
                standardize=standardize, 
                smiles_column='Drug', 
                label_column='Y', 
                benchmark_task_type="classification"
            ),
            'ADMET-R': partial(
                self.QSAR, 
                target_list=[
                    'Caco2_Wang', 'Lipophilicity_AstraZeneca', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu', 'HydrationFreeEnergy_FreeSolv', 'Solubility_AqSolDB', 'hERG_at_1uM', 'hERG_at_10uM'
                ], 
                smiles_column='Drug', 
                label_column='Y', 
                standardize=standardize, 
                benchmark_task_type="regression"
            ),
            'LIT-QSAR': partial(
                self.QSAR, 
                target_list=[
                    "ALDH1", "FEN1", "GBA", "MAPK1", "PKM2", "KAT2A", "VDR"
                ], 
                smiles_column='SMILES', 
                label_column='Label', 
                standardize=standardize, 
                benchmark_task_type="ranking"
            ),
            'ST-QSAR': partial(
                self.QSAR, 
                target_list=[
                    "RGS12", "HADH2", "HSD17B4", "CRF-R2-antagonists", "CRF-R2-agonist", "CBFb-RUNX1", "Rango", "ARNT-TAC3", "Vif-APOBEC3G", "A1Apoptosis", "Peg3"
                ], 
                smiles_column='SMILES', 
                label_column='Label', 
                standardize=standardize, 
                benchmark_task_type="ranking"
            ),
            'PW-QSAR': partial(
                self.QSAR, 
                target_list=[
                    "NF-kB","uPA","PINK1"
                ], 
                smiles_column='SMILES', 
                label_column='Label', 
                standardize=standardize, 
                benchmark_task_type="ranking"
            ),
            'CELLS-QSAR': partial(
                self.QSAR, 
                target_list=[
                    "NCI_786-0","NCI_BT-549","NCI_DMS114","NCI-H23", "NCI_HCT-116","NCI_HOP-92","NCI_KM12","NCI_M19-MEL","NCI_MDA-N","NCI_OVCAR-8","NCI_RXF393","NCI_SK-MEL-2","NCI_SN12K1","NCI_SW-620","NCI_UACC-62","NCI_A498","NCI_CAKI-1","NCI_DMS273","NCI-H322M","NCI_HCT-15","NCI_HS578T","NCI_KM20L2","NCI_MALME-3M","NCI_MOLT-4","NCI_P388","NCI_RXF-631","NCI_SK-MEL-28","NCI_SNB-19","NCI_T-47D","NCI_UO-31","NCI_A549-ATCC","NCI_CCRF-CEM","NCI_DU-145","NCI-H460","NCI_HL-60(TB)","NCI_HT29","NCI_LOX-IMVI","NCI_MCF7","NCI_OVCAR-3","NCI_P388-ADR","NCI_SF-268","NCI_SK-MEL-5","NCI_SNB-75","NCI_TK-10","NCI_XF498","NCI_ACHN","NCI_COLO205","NCI_EKVX","NCI-H522","NCI_HOP-18","NCI_IGROV1","NCI_LXFL529","NCI_MDA-MB-231-ATCC","NCI_OVCAR-4","NCI_PC-3","NCI_SF-295","NCI_SK-OV-3","NCI_SNB-78","NCI_U251","NCI-ADR-RES","NCI_DLD-1","NCI-H226","NCI_HCC-2998","NCI_HOP-62","NCI_K-562","NCI_M14","NCI_MDA-MB-435","NCI_OVCAR-5","NCI_RPMI-8226","NCI_SF-539","NCI_SN12C","NCI_SR","NCI_UACC-257"
                ], 
                smiles_column='SMILES', 
                label_column='Label', 
                standardize=standardize, 
                benchmark_task_type="ranking"
            ),
        }
        self.benchmark_results = benchmark_functions[self.benchmark_name]()
        print(f'======== {self.benchmark_name} Benchmark Report ========')
        print(f'Model performance on {self.benchmark_name} benchmark: {self.model_name}')
        print(self.benchmark_results)

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # params
    model_name = sys.argv[1]
    benchmark_index_file = sys.argv[2]
    benchmark_file_basepath = os.path.dirname(benchmark_index_file)
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    benchmark_task = sys.argv[3]
    Benchmark_Protocol = Benchmark(model_name=model_name, data_record=True)
    # benchmarking
    Benchmark_Protocol(benchmark_task, f"{benchmark_file_basepath}/{benchmark_index_dict[benchmark_task]}", standardize=True)


