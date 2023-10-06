import os
import sys
import time
import json
import random
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from model.GeminiMol import GeminiMol
from benchmark import Benchmark

benchmark_list_dict = {
    # 'BindingDB': [
    #     '4779', # michael receptor
    #     '17055', # macrocyclic compounds
    #     '7460', # natural product
    # ],
    'LIT-PCBA': [
        # 'TP53', # disorder protein
        "ESR1_ago", "OPRK1", "ADRB2", # agonist
        "KAT2A", "VDR", # protein-protein interaction
    ],
    'QSAR': [
        "ALDH1", # abundant active compound data
        "MAPK1", # rare active compound data
        "Peg3", # rare QSAR data
    ],
    'CELLS': [
        "NCI_SR", # abundant QSAR data
        "NCI_P388-ADR", # rare QSAR data
        "NCI_HOP-18", # normal 
    ],
    'ADMET-C': [
        'Bioavailability_Ma', 'HIA_Hou', 'Pgp_Broccatelli', 'BBB_Martins', # critical PK properties
        'hERG', 'AMES', 'DILI' # # critical Tox properties
    ],
    'ADMET-R': [
        'Caco2_Wang', 'Solubility_AqSolDB', 'PPBR_AZ', 'VDss_Lombardo', # critical PK properties
        'Half_Life_Obach', # critical PK properties
    ]
}

smiles_column_name_dict = {
    'QSAR': 'SMILES',
    'CELLS': 'SMILES',
    'ADMET-C': 'Drug',
    'ADMET-R': 'Drug'
}

label_column_name_dict = {
    'QSAR': 'Label',
    'CELLS': 'Label',
    'ADMET-C': 'Y',
    'ADMET-R': 'Y'
}

task_type_dict = {
    'QSAR': "classification",
    'CELLS': "classification",
    'ADMET-C': "classification",
    'ADMET-R': "regression"
}

metrics_dict = {
    'LIT-PCBA': "BEDROC",
    'BindingDB': "BEDROC",
    'QSAR': "AUROC",
    'CELLS': "AUROC",
    'ADMET-C': "AUROC",
    'ADMET-R': "SPEARMANR"
}

def evaluate_model(evaluater, model_info, benchmark_index_dict):
    evaluater.standardize = False
    for task, target_list in benchmark_list_dict.items():
        evaluater.benchmark_name = task
        evaluater.data_path = benchmark_index_dict[task]
        if task == 'LIT-PCBA':
            for target in benchmark_list_dict['LIT-PCBA']:
                query_smiles_state_table, ref_smiles_list = evaluater.read_LITPCBA(target)
                res_dict = evaluater.vritual_screening_on_target(target, ref_smiles_list, query_smiles_state_table, ['BEDROC'], reverse=False, smiles_column='smiles', state_name='state', duplicate_column='smiles')[metrics_dict[task]].to_dict()
                for key, value in res_dict.items():
                    model_info[f'{target}_{key}'] = value
        elif task == 'BindingDB':
            evaluater.data_table = pd.read_csv(f"{evaluater.data_path}/BindingDB_Benchmark_Decoys.csv", dtype={'SMILES':str, 'Title':str, 'Number_of_Target':int})
            evaluater.data_table = evaluater.prepare(evaluater.data_table, smiles_column='SMILES')
            evaluater.target_dict = dict(zip(evaluater.data_table['Title'], evaluater.data_table['SMILES']))
            evaluater.target_number_dict = dict(zip(evaluater.data_table['Title'], evaluater.data_table['Number_of_Target']))
            for target in benchmark_list_dict['BindingDB']:
                ref_smiles = evaluater.target_dict[target]
                number_of_targets = evaluater.target_number_dict[target]
                binding_data_table = evaluater.read_BindingDB(f"{target}_{number_of_targets}")
                res_dict = evaluater.vritual_screening_on_target(target, [ref_smiles], binding_data_table, ['BEDROC'], reverse=True, smiles_column='Ligand_SMILES', state_name='state_label', duplicate_column='Targets')[metrics_dict[task]].to_dict()
                for key, value in res_dict.items():
                    model_info[f'{target}_{key}'] = value
        else:
            res_dict = evaluater.QSAR(target_list=target_list, smiles_column=smiles_column_name_dict[task], label_column=label_column_name_dict[task], standardize=False, benchmark_task_type=task_type_dict[task])[metrics_dict[task]].to_dict()
            for key, value in res_dict.items():
                model_info[f'{key}'] = value
    return model_info

if __name__ == '__main__':
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # defalut params
    training_random_seed = 1207
    np.random.seed(training_random_seed)   
    # read data and build dataset
    data_path = sys.argv[1]
    model_name = sys.argv[2]
    output_path = sys.argv[3]
    benchmark_index_file = sys.argv[4]
    with open(benchmark_index_file, 'r', encoding='utf-8') as f:
        benchmark_index_dict = json.load(f)
    # read data and build dataset
    test_set = pd.read_csv(f'{data_path}/test.csv') 
    val_set = shuffle(test_set[test_set['assign']=="val"], random_state=training_random_seed)
    # data process
    predictor_geminimol = GeminiMol(model_name = model_name)
    if os.path.exists(f'{model_name}_strolling_benchmark.csv'):
        strolling_results = pd.read_csv(f'{model_name}_strolling_benchmark.csv')
        print(f'NOTE: {model_name}_strolling_benchmark.csv was loaded.')
        print(strolling_results)
    else:
        print(f"NOTE: Strolling {model_name} ...")
        train_set = shuffle(pd.read_csv(f'{data_path}/training_graph.csv'))
        print('Strolling Set: Number=', len(train_set))
        print('Validation Set: Number=', len(val_set))
        prefix = str(random.randint(100, 999))
        # training 
        model_list = predictor_geminimol.strolling(
            train_df = train_set,  
            val_df = val_set,
            prefix = prefix,
            output_path = output_path,
            epochs = 2, 
            learning_rate = 1.0e-4, 
            optim_type = 'AdamW',
            T_max = 10000,
            schedule_unit = 50
        )
        # record model info
        pd.DataFrame.from_records(model_list).to_csv(f'{model_name}_{prefix}_strolling.csv', index=False)
        # set up benchmark
        predictor_geminimol.similarity_metrics_list = ['Cosine']
        # evaluate models
        result_model_info_list = []
        for model_info in model_list:
            predictor_geminimol.load_state_dict(torch.load(model_info['model_path']))
            evaluater = Benchmark(predictor=predictor_geminimol, model_name=model_name, record=False)
            model_info = evaluate_model(evaluater, model_info, benchmark_index_dict)
            print(f'NOTE: {model_info["model_path"]} was evaluated.')
            result_model_info_list.append(model_info)
            strolling_results = pd.DataFrame(result_model_info_list)
            strolling_results.to_csv(f'{model_name}_strolling_benchmark.csv', index=False, header=True, sep=',')
        print(strolling_results)
        print(f'NOTE: {model_name}_strolling_benchmark.csv was saved.')
    # normalizing
    print(f'NOTE: Normalizing {model_name}_strolling_benchmark table ...')
    strolling_results = strolling_results.set_index('model_path')
    strolling_results = strolling_results.apply(lambda x: x / np.max(x))
    strolling_results = strolling_results.reset_index()
    model_ranking = strolling_results.copy()
    model_ranking['fusion_model_score'] = model_ranking.apply(lambda x: np.mean(x[1:]), axis=1)
    model_ranking = model_ranking.sort_values(by='fusion_model_score', ascending=False)
    model_ranking.to_csv(f'{model_name}_strolling_ranking.csv', index=False, header=True, sep=',')
    print(model_ranking)
    print(f'NOTE: {model_name}_model_ranking.csv was saved.')
    # greedy up model fusion
    model_list = model_ranking[['model_path', 'fusion_model_score']].to_dict(orient='records')
    cand_model_1 = GeminiMol(model_name = model_name)
    cand_model_1.load_state_dict(torch.load(model_list[0]['model_path']))
    best_model_score = model_list[0]['fusion_model_score']
    fusion_id = 0
    output_model_list = []
    for model_info in model_list[1:]:
        model_path = model_info['model_path']
        model_score = model_info['fusion_model_score']
        fusion_id += 1
        fusion_model_info = {}
        cand_model_2 = GeminiMol(model_name = model_name)
        cand_model_2.load_state_dict(torch.load(model_path))
        for param_A, param_B, param_avg in zip(cand_model_1.parameters(), cand_model_2.parameters(), predictor_geminimol.parameters()):
            param_avg.data = (param_A.data + param_B.data) / 2
        cand_model_1 = cand_model_2
        del cand_model_2
        evaluater = Benchmark(predictor=predictor_geminimol, model_name=model_name, record=False)
        torch.save(predictor_geminimol.state_dict(), f"{output_path}/GeminiMol_F{fusion_id}.pt")
        fusion_model_info['model_path'] = f"{output_path}/GeminiMol_F{fusion_id}.pt"
        fusion_model_info = evaluate_model(evaluater, fusion_model_info, benchmark_index_dict)
        print(f'NOTE: {fusion_model_info["model_path"]} was evaluated.')
        print(fusion_model_info)
        output_model_list.append(fusion_model_info)
    fusion_results = pd.DataFrame(output_model_list)
    fusion_results.to_csv(f'{model_name}_fusion_benchmark.csv', index=False, header=True, sep=',')
    print(fusion_results)
    # save the best fusion model
    fusion_results = fusion_results.set_index('model_path')
    fusion_results = fusion_results.apply(lambda x: x / np.max(x))
    fusion_results = fusion_results.reset_index()
    fusion_model_ranking = pd.concat([strolling_results, fusion_results], ignore_index=True)
    fusion_model_ranking['fusion_model_score'] = fusion_model_ranking.apply(lambda x: np.mean(x[1:]), axis=1)
    fusion_model_ranking = fusion_model_ranking.sort_values(by='fusion_model_score', ascending=False)
    fusion_model_ranking.to_csv(f'{model_name}_fusion_model_ranking.csv', index=False, header=True, sep=',')
    print(fusion_model_ranking)
    best_model_path = fusion_model_ranking.iloc[0]['model_path']
    best_model_score = fusion_model_ranking.iloc[0]['fusion_model_score']
    os.mkdir(f'{model_name}_F')
    os.system(f'cp {best_model_path} {model_name}_F/GeminiMol.pt')
    os.system(f'cp {model_name}/model_params.json {model_name}_F/')
    Benchmark_Protocol = Benchmark(predictor=GeminiMol(f'{model_name}_F'), model_name=f'{model_name}_F')
    Benchmark_Protocol('DUDE', benchmark_index_dict['DUDE'], standardize=False)
    Benchmark_Protocol('LIT-PCBA', benchmark_index_dict['LIT-PCBA'], standardize=False)
    Benchmark_Protocol('ADMET-C', benchmark_index_dict['ADMET-C'], standardize=False)
    Benchmark_Protocol('ADMET-R', benchmark_index_dict['ADMET-R'], standardize=False)
    Benchmark_Protocol('QSAR', benchmark_index_dict['QSAR'], standardize=False)
    Benchmark_Protocol('CELLS', benchmark_index_dict['CELLS'], standardize=False)
    Benchmark_Protocol('BindingDB', benchmark_index_dict['BindingDB'], standardize=False)


    