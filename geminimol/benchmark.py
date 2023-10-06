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

class Benchmark():
    def __init__(self, predictor, model_name, record = True, data_record = False):
        self.predictor = predictor
        self.model_name = model_name
        self.record = record
        self.data_record = data_record
        self.statistics_metrics_dict = {
            'ranking': ['AUROC', 'BEDROC', 'EF1%', 'EF0.5%', 'EF0.1%', 'EF0.05%', 'EF0.01%', 'logAUC'], 
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
        self.label_map = {'Active': 1, 'Inactive': 0, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'positive': 1, 'negative': 0, 1: 1, 0: 0}

    def prepare(self, dataset, smiles_column='smiles'):
        if self.standardize == True:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x, kekule=False))
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:check_smiles_validity(x))
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset

    def read_smi(self, smi_file):
        query_smiles = pd.read_csv(smi_file,header=None,usecols=[0],sep='\s|,|;|\t| ',engine='python')
        query_smiles.columns = ['smiles']
        print(f"Note: The input raw table {smi_file} has {len(query_smiles['smiles'])} rows.")
        query_smiles = self.prepare(query_smiles)
        print(f"Note:  The processed input table {smi_file} has {len(query_smiles['smiles'])} rows.")
        return query_smiles
    
    def statistics(self, total_pred_res, statistics_metrics_list=None, score_name='score', ascending=False, state_name='state', duplicate_column='smiles'):
        total_pred_res.sort_values(score_name, ascending=ascending, inplace=True)
        total_pred_res.drop_duplicates(subset=[duplicate_column], keep='first', inplace=True)
        y_pred = np.array(total_pred_res[score_name].to_list())
        y_true = np.array(total_pred_res[state_name].to_list())
        statistic_results = {}
        if statistics_metrics_list is None:
            statistics_metrics_list = list(self.metric_functions.keys())
        for metric in statistics_metrics_list:
            statistic_results[metric] = self.metric_functions[metric](y_true, y_pred)
        return statistic_results
    
    def vritual_screening_on_target(self, target, ref_smiles_list, query_smiles_state_table, statistics_metrics_list, reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles'):
        if os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_data.csv"):
            pred_res = pd.read_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_data.csv")
        else:
            pred_res = self.predictor.virtual_screening(ref_smiles_list, query_smiles_state_table, reverse=reverse, smiles_column=smiles_column, similarity_metrics=self.predictor.similarity_metrics_list)
            if self.record and self.data_record:
                pred_res.to_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_data.csv", index=False, header=True, sep=',')
        # statistic_table (score_types/similarity_metrics, metrics)
        statistic_table = pd.DataFrame(columns=statistics_metrics_list, index=self.predictor.similarity_metrics_list)
        for score_type in self.predictor.similarity_metrics_list:
            statistic_results = self.statistics(pred_res, statistics_metrics_list=statistics_metrics_list, duplicate_column=duplicate_column, score_name=score_type, state_name=state_name)
            for statistics_metric, value in statistic_results.items():
                statistic_table.loc[[score_type],[statistics_metric]] = value
        if self.record:
            statistic_table.to_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv", index=True, header=True, sep=',')
        return statistic_table
    
    def reporting_benchmark(self, statistic_tables):
        # benchmark_results (score_types/similarity_metrics, metrics)
        benchmark_results = pd.concat(statistic_tables.values()).groupby(level=0).mean().groupby(axis=1, level=0).mean()
        if self.record:
            benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_final_statistics.csv", index=True, header=True, sep=',')
        return benchmark_results

    def read_DUDE(self, target):
        active_smiles = self.read_smi(str(self.data_path+"/"+target+"/actives_final.smi"))
        decoys_smiles = self.read_smi(str(self.data_path+"/"+target+"/decoys_final.smi"))
        active_smiles['state'] = 1
        decoys_smiles['state'] = 0
        query_smiles_state_table = pd.concat([active_smiles, decoys_smiles], ignore_index=True)
        return query_smiles_state_table

    def DUDE_VS(self, benchmark_task_type="ranking", standardize=False):
        self.standardize = standardize
        self.data_table = pd.read_csv(f"{self.data_path}/DUDE-smiles.csv")
        self.target_dict = dict(zip(self.data_table['Title'], self.data_table['SMILES']))
        # statistic_tables { target : statistic_table (score_types/similarity_metrics, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.target_dict.keys()}
        for target, ref_smiles in self.target_dict.items():
            if os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv", index_col=0)
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
        self.target_list = target_list
        self.standardize = standardize
        # statistic_tables { target : statistic_table (score_types/similarity_metrics, metrics)}
        self.statistic_tables = {key:pd.DataFrame() for key in self.target_list}
        for target in self.target_list:
            if os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv", index_col=0)
            else:
                query_smiles_state_table, ref_smiles_list = self.read_LITPCBA(target)
                self.statistic_tables[target] = self.vritual_screening_on_target(target, ref_smiles_list, query_smiles_state_table, self.statistics_metrics_dict[benchmark_task_type], reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles')
        return self.reporting_benchmark(self.statistic_tables)

    def read_BindingDB(self, target):
        binding_data_table = pd.read_csv(f"{self.data_path}/{target}_DATA.csv", on_bad_lines='skip', dtype={'Monomer_ID':str, 'Ligand_SMILES':str, 'Binding':str, 'Targets':str, 'state_label':int})
        binding_data_table = self.prepare(binding_data_table[['Monomer_ID', 'Ligand_SMILES', 'Binding', 'Targets', 'state_label']], smiles_column='Ligand_SMILES')
        return binding_data_table

    def BindingDB_TargetID(self, decoy_list=None, index="BindingDB", benchmark_task_type="ranking", standardize=False):
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
            if os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv"):
                self.statistic_tables[target] = pd.read_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{target}_statistics.csv", index_col=0)
            else:
                binding_data_table = self.read_BindingDB(f"{target}_{number_of_targets}")
                self.statistic_tables[target] = self.vritual_screening_on_target(target, [ref_smiles], binding_data_table, self.statistics_metrics_dict[benchmark_task_type], reverse=True, smiles_column='Ligand_SMILES', state_name='state_label', duplicate_column='Targets')
        return self.reporting_benchmark(self.statistic_tables)

    def QSAR(self, target_list=None, stack=False, smiles_column='SMILES', label_column='Label', standardize=False, benchmark_task_type="classification"):
        from QSAR import QSAR
        self.target_list = target_list
        # benchmark_results (targets, metrics)
        benchmark_results = pd.DataFrame(columns=['model']+self.statistics_metrics_dict[benchmark_task_type], index=self.target_list)
        for target in self.target_list:
            training_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_train.csv')
            test_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_test.csv')
            if benchmark_task_type in ["ranking", "classification"]:
                training_data[label_column] = training_data[label_column].replace(self.label_map)
                test_data[label_column] = test_data[label_column].replace(self.label_map)
            if len(list(set(training_data[label_column].to_list()))) == 2:
                task_type = 'binary'
            elif 3 <= len(list(set(training_data[label_column].to_list()))) <=10:
                task_type = 'multiclass'
            else:
                task_type = 'regression'
            if os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{self.model_name}_{target}_results.csv"):
                self.QSAR_model = QSAR(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{self.model_name}_{target}", encoder_list = [self.predictor], standardize = standardize, label_column=label_column, smiles_column=smiles_column, task_type=task_type)
                test_leaderboard = pd.read_csv(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{self.model_name}_{target}_results.csv")
                test_leaderboard.sort_values(self.QSAR_model.recommanded_metric[task_type], ascending=False, inplace=True)
                self.QSAR_model.best_model_name = test_leaderboard['model'].to_list()[0]
            else:
                if os.path.exists(f'{self.data_path}/{target}/{target}_scaffold_valid.csv'):
                    val_data = pd.read_csv(f'{self.data_path}/{target}/{target}_scaffold_valid.csv')
                    if benchmark_task_type in ["ranking", "classification"]:
                        val_data[label_column] = val_data[label_column].replace(self.label_map)
                else:
                    val_data = None
                self.QSAR_model = QSAR(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/{self.model_name}_{target}", encoder_list = [self.predictor], standardize = standardize, label_column=label_column, smiles_column=smiles_column, task_type=task_type)
                self.QSAR_model.trianing_models(training_data, val_set=val_data, stack=stack)
                # if you need select the best model by test score, run: 
                test_leaderboard = self.QSAR_model.test(test_data) 
            benchmark_results.loc[[target],['model']] = self.QSAR_model.best_model_name
            # default, select the best model by validation score
            predicted_results = self.QSAR_model.predict(test_data) 
            statistic_results = self.statistics(predicted_results, statistics_metrics_list=self.statistics_metrics_dict[benchmark_task_type], duplicate_column=smiles_column, score_name='pred', state_name=label_column)
            for statistics_metric, value in statistic_results.items():
                benchmark_results.loc[[target],[statistics_metric]] = value
        if self.record:
            benchmark_results.to_csv(f"{self.model_name}/{self.benchmark_name}_final_statistics.csv", index=True, header=True, sep=',')
        return benchmark_results

    def __call__(self, benchmark_name, data_path, standardize=False):
        self.benchmark_name = benchmark_name
        self.data_path = data_path
        ## please set the standardize to True if you add some new datasets
        if not os.path.exists(f"{self.model_name}"):
            os.mkdir(f"{self.model_name}")
        if not os.path.exists(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/"):
            os.mkdir(f"{self.model_name}/{self.model_name}_{self.benchmark_name}/")
        benchmark_functions = {
            'DUDE': partial(self.DUDE_VS, benchmark_task_type="ranking", standardize=standardize),
            'LIT-PCBA': partial(self.LITPCBA_VS, benchmark_task_type="ranking", standardize=standardize),
            'TIBD': partial(self.BindingDB_TargetID, index="TIBD", benchmark_task_type="ranking", standardize=standardize),
            'BindingDB': partial(self.BindingDB_TargetID, index="BindingDB", benchmark_task_type="ranking", standardize=standardize),
            'ADMET-C': partial(self.QSAR, target_list=['Bioavailability_Ma', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', 'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'hERG_Karim', 'AMES', 'DILI', 'SkinReaction', 'Carcinogens_Lagunin', 'ClinTox', 'hERG_inhib', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'PAMPA_NCATS', 'AddictedChem'], standardize=standardize, smiles_column='Drug', label_column='Y', benchmark_task_type="classification", stack=False),
            'ADMET-R': partial(self.QSAR, target_list=['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu', 'HydrationFreeEnergy_FreeSolv', 'hERG_at_1uM', 'hERG_at_10uM'], smiles_column='Drug', label_column='Y', standardize=standardize,  benchmark_task_type="regression", stack=False),
            'QSAR': partial(self.QSAR, target_list=["ALDH1", "FEN1", "GBA", "MAPK1", "PKM2", "KAT2A", "VDR", "RGS12", "HADH2", "HSD17B4", "CRF-R2-antagonists", "CRF-R2-agonist", "CBFb-RUNX1", "Rango", "ARNT-TAC3", "Vif-APOBEC3G", "NF-kB","A1Apoptosis","uPA","Peg3","PINK1"], smiles_column='SMILES', label_column='Label', standardize=standardize, benchmark_task_type="ranking", stack=False),
            'CELLS': partial(self.QSAR, target_list=["NCI_786-0","NCI_BT-549","NCI_DMS114","NCI-H23", "NCI_HCT-116","NCI_HOP-92","NCI_KM12","NCI_M19-MEL","NCI_MDA-N","NCI_OVCAR-8","NCI_RXF393","NCI_SK-MEL-2","NCI_SN12K1","NCI_SW-620","NCI_UACC-62","NCI_A498","NCI_CAKI-1","NCI_DMS273","NCI-H322M","NCI_HCT-15","NCI_HS578T","NCI_KM20L2","NCI_MALME-3M","NCI_MOLT-4","NCI_P388","NCI_RXF-631","NCI_SK-MEL-28","NCI_SNB-19","NCI_T-47D","NCI_UO-31","NCI_A549-ATCC","NCI_CCRF-CEM","NCI_DU-145","NCI-H460","NCI_HL-60(TB)","NCI_HT29","NCI_LOX-IMVI","NCI_MCF7","NCI_OVCAR-3","NCI_P388-ADR","NCI_SF-268","NCI_SK-MEL-5","NCI_SNB-75","NCI_TK-10","NCI_XF498","NCI_ACHN","NCI_COLO205","NCI_EKVX","NCI-H522","NCI_HOP-18","NCI_IGROV1","NCI_LXFL529","NCI_MDA-MB-231-ATCC","NCI_OVCAR-4","NCI_PC-3","NCI_SF-295","NCI_SK-OV-3","NCI_SNB-78","NCI_U251","NCI-ADR-RES","NCI_DLD-1","NCI-H226","NCI_HCC-2998","NCI_HOP-62","NCI_K-562","NCI_M14","NCI_MDA-MB-435","NCI_OVCAR-5","NCI_RPMI-8226","NCI_SF-539","NCI_SN12C","NCI_SR","NCI_UACC-257"], smiles_column='SMILES', label_column='Label', standardize=standardize, benchmark_task_type="ranking", stack=False),
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
    # params
    model_name = sys.argv[1]
    method_list = [model_name]
    assert isinstance(method_list, list), f"ERROR: the method list of Benchmark class must be list, but get {method_list}."
    avliable_fingerprint_list = ["MACCS", "RDK", "TopologicalTorsion", "AtomPairs", "ECFP4", "FCFP4", "ECFP6", "FCFP6", "CombineFP"]
    if len(method_list) == 1 and method_list[0] not in avliable_fingerprint_list and os.path.exists(f'{method_list[0]}'):
        model_name = str(method_list[0].split('/')[-1])
        if os.path.exists(f'{method_list[0]}/GeminiMol.pt'):
            from model.GeminiMol import GeminiMol
            predictor = GeminiMol(method_list[0])    
    elif len(method_list) == 1 and method_list[0] == "CombineFP":
        model_name = "CombineFP"
        method_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
        from utils.fingerprint import Fingerprint
        predictor = Fingerprint(method_list)
    else:
        for method in method_list:
            assert method in avliable_fingerprint_list, f"ERROR: Unsupported fingerprint type {method}, note that combining the customed GeminiMol model with other Fingerprints isn't supported now! {method_list}" 
        model_name = '_'.join(method_list)
        from utils.fingerprint import Fingerprint
        predictor = Fingerprint(method_list)
    Benchmark_Protocol = Benchmark(predictor=predictor, model_name=model_name, data_record=True)
    # benchmarking
    benchmark_index_file = sys.argv[2]
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    benchmark_task = sys.argv[3]
    Benchmark_Protocol(benchmark_task, benchmark_index_dict[benchmark_task], standardize=False)


