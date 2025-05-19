import sys
import numpy as np
import pandas as pd
from random import sample
import torch
from utils.fingerprint import Fingerprint
from model.GeminiMol import GeminiMol
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.spatial.distance as dist
from utils.chem import gen_standardize_smiles

if __name__ == "__main__":
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # random_seed
    # np.random.seed(1207)
    training_random_seed = 1207
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # params
    data_table = pd.read_csv(sys.argv[1])
    geminimol_path = sys.argv[2]
    jobname = sys.argv[3]
    ## read the models
    encoders = {
        'GeminiMol': GeminiMol(
                geminimol_path,
                depth = 0, 
                custom_label = ['Pearson'], 
                internal_label_list = [],
                extrnal_label_list = ['Pearson']
            ),
        'ECFP4': Fingerprint(
                fingerprint_type_list = ['ECFP4'],
            ),
        'AtomPairs': Fingerprint(
                fingerprint_type_list = ['AtomPairs'],
            ),
        'MACCS': Fingerprint(
                fingerprint_type_list = ['MACCS'],
            )
    }
    # prepare the smiles
    data_table['smiles1'] = data_table['smiles1'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
    data_table = data_table[data_table['smiles1'] != 'smiles_unvaild'].reset_index(drop=True)
    data_table['smiles2'] = data_table['smiles2'].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
    data_table = data_table[data_table['smiles2'] != 'smiles_unvaild'].reset_index(drop=True)
    if 'State' in data_table.columns:
        print("NOTE: This is pre-sampled data.")
        # check data size and ratio
        inter_group_data_size = len(data_table[data_table['State']=='inter-group'])
        intra_group_data_size = len(data_table[data_table['State']=='intra-group'])
        sample_size = min(inter_group_data_size, intra_group_data_size)
        data_table = data_table.sample(frac=1).reset_index(drop=True)
        sampled_tabele = pd.concat(
            [
                data_table[data_table['State']=='inter-group'].head(sample_size),
                data_table[data_table['State']=='intra-group'].head(sample_size)
            ],
            ignore_index=True
        )
    elif 'State' not in data_table.columns:
        print("NOTE: This is postive-pairs data.")
        # mark the state for data table
        data_table['State'] = 'intra-group'
        # sample the inter-group data
        smiles1_set = data_table[['Group','smiles1']]
        smiles1_set.columns = ['Group1','smiles1']
        smiles2_set = data_table[['Group','smiles2']]
        smiles2_set.columns = ['Group2','smiles2']
        sampled_tabele = data_table[['State','smiles1','smiles2','Group']]
        for _ in range(len(data_table)):
            smiles1_set = smiles1_set.sample(frac=1).reset_index(drop=True)
            smiles2_set = smiles2_set.sample(frac=1).reset_index(drop=True)
            round = pd.concat([smiles1_set, smiles2_set], axis=1)
            round = round[round['Group1'] != round['Group2']]
            round['State'] = 'inter-group'
            round['Group'] = round['Group1'] + ':' + round['Group2']
            sampled_tabele = pd.concat(
                [sampled_tabele, round[['State','smiles1','smiles2','Group']]], 
                axis=0, ignore_index=True
            )
    # de-duplicate the data table
    sampled_tabele = sampled_tabele.drop_duplicates(
        subset = ['smiles1', 'smiles2'],
        keep = 'first',
        ignore_index = True
    )
    # report the sample statistics
    print(f"Total {len(sampled_tabele)} pairs of compounds.")
    print(f"Total {len(sampled_tabele[sampled_tabele['State'] == 'intra-group'])} intra-group pairs.")
    print(f"Total {len(sampled_tabele[sampled_tabele['State'] == 'inter-group'])} inter-group pairs.")
    # predict the similarity
    stat_res = pd.DataFrame(columns=['Model', 'Energy', 'Manhattan', 'Minkowski', 'Wasserstein'])
    for model_name in encoders.keys():
        print(f"Predicting the similarity using {model_name} ...")
        model = encoders[model_name]
        if model_name == 'GeminiMol':
            pred_values = model.predict(sampled_tabele)
            pred_values.columns = ['GeminiMol']
            sampled_tabele = pd.concat([sampled_tabele, pred_values], axis=1)
        else:
            sampled_tabele[model_name] = sampled_tabele.apply(
                lambda x: model.similarity(x['smiles1'], x['smiles2']), 
                axis=1
            )
        inter_group_sim = sampled_tabele[sampled_tabele['State'] == 'inter-group'][model_name].to_list()
        intra_group_sim = sampled_tabele[sampled_tabele['State'] == 'intra-group'][model_name].to_list()
        # calculate the distance
        sample_rounds = 10
        sample_size = min(len(inter_group_sim), len(intra_group_sim))
        postive_samples = [
            sample(intra_group_sim, sample_size) for _ in range(sample_rounds)
        ]
        negative_samples = [
            sample(inter_group_sim, sample_size) for _ in range(sample_rounds)
        ]
        energy_distances = [
            stats.energy_distance(postive_samples[i], negative_samples[i]) for i in range(sample_rounds)
        ]
        manhattan_distances = [
            dist.cityblock(postive_samples[i], negative_samples[i]) for i in range(sample_rounds)
        ]
        minkowski_distances = [
            dist.minkowski(postive_samples[i], negative_samples[i], p=2) for i in range(sample_rounds)
        ]
        wasserstein_distances = [
            stats.wasserstein_distance(postive_samples[i], negative_samples[i]) for i in range(sample_rounds)
        ]
        stat_res = pd.concat([
            stat_res,
            pd.DataFrame(
                {
                    'Model': model_name,
                    'Energy': f"{np.mean(energy_distances):.4f}({np.std(energy_distances):.2f})",
                    'Wasserstein': f"{np.mean(wasserstein_distances):.4f}({np.std(wasserstein_distances):.2f})",
                    # 'Manhattan': f"{np.mean(manhattan_distances):.4f}({np.std(manhattan_distances):.2f})",
                    # 'Minkowski': f"{np.mean(minkowski_distances):.4f}({np.std(minkowski_distances):.2f})",
                },
                index = [0]
            )
            ],
            ignore_index=True
        )            
        # plot the density map
        plt.figure(figsize=(3.5,3.5), dpi=600)
        sns.kdeplot(inter_group_sim, fill=True, color="r", label="Inter-group")
        sns.kdeplot(intra_group_sim, fill=True, color="b", label="Intra-group")
        plt.title(f'{model_name} (Distance: {np.mean(intra_group_sim) - np.mean(inter_group_sim):.4f})')
        plt.xlabel('Similarity')
        plt.ylabel('Density')
        plt.legend()
        # set the x and y scales
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 10)
        plt.tight_layout()
        plt.savefig(f'{jobname}_{model_name}_density.png')
        plt.close()
    # save the statistics
    print(stat_res)
    print("Saving the statistics ...")
    stat_res.to_csv(f'{jobname}_stat.csv', index=False)
    # plot the ROC curves for the models
    print("Plotting the ROC curves ...")
    plt.figure(figsize=(3.5,3.5), dpi=600)
    for model_name in encoders.keys():
        fpr, tpr, thresholds = metrics.roc_curve(
            sampled_tabele['State'] == 'intra-group', 
            sampled_tabele[model_name]
        )
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})')
        print(f"{model_name} AUC: {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{jobname}_ROC.png')
    plt.close()
    # plot the PRC curves for the models
    print("Plotting the PRC curves ...")
    plt.figure(figsize=(3.5,3.5), dpi=600)
    for model_name in encoders.keys():
        precision, recall, thresholds = metrics.precision_recall_curve(
            sampled_tabele['State'] == 'intra-group', 
            sampled_tabele[model_name]
        )
        auprc = metrics.auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUPRC={auprc:.4f})')
        print(f"{model_name} AUPRC: {auprc:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{jobname}_PRC.png')
    plt.close()
    # save the results
    sampled_tabele.to_csv(f'{jobname}_data.csv', index=False)
