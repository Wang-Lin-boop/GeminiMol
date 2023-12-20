import os
import sys
import numpy as np
import pandas as pd
import math
import torch
from utils.fingerprint import Fingerprint
import matplotlib.pyplot as plt
from utils.chem import check_smiles_validity, gen_standardize_smiles
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, DictionaryLearning, KernelPCA, IncrementalPCA, FastICA, SparsePCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN, AffinityPropagation, SpectralClustering, Birch, OPTICS
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr

class Mol_Encoder:
    def __init__(self, 
            encoder_list=[Fingerprint(['ECFP4'])], 
            standardize=False, 
            smiles_column='smiles'
        ):
        self.standardize = standardize
        self.encoders = encoder_list
        self.smiles_column = smiles_column

    def prepare(self, dataset):
        dataset = dataset.dropna(subset=[self.smiles_column])
        print(f"NOTE: read the dataset size ({len(dataset)}).")
        if self.standardize == True:
            dataset[smiles_column] = dataset[self.smiles_column].apply(lambda x: gen_standardize_smiles(x, kekule=False, random=False))
        else:
            dataset[smiles_column] = dataset[self.smiles_column].apply(lambda x: check_smiles_validity(x))
        dataset = dataset[dataset[self.smiles_column]!='smiles_unvaild']
        dataset.drop_duplicates(
            subset=[self.smiles_column], 
            keep='first', 
            inplace=True,
            ignore_index = True
        )
        print(f"NOTE: processed dataset size ({len(dataset)}).")
        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def encoder(self, query_smiles_table):
        features_columns = []
        query_smiles_table = self.prepare(query_smiles_table)
        for single_encoder in self.encoders:
            features = single_encoder.extract_features(query_smiles_table, smiles_column=self.smiles_column)
            features_columns += list(features.columns)
            query_smiles_table = query_smiles_table.join(features, how='left')
        return query_smiles_table, features_columns
        
class unsupervised_clustering:
    def __init__(self, 
            cluster_algorithm_list=['AffinityPropagation'], 
            reduce_dimension_algorithm_list=['tSNE', 'PCA']
        ):
        self.cluster_models = {
            'K-Means': lambda num_clusters: KMeans(n_clusters=num_clusters),
            'hierarchical': lambda num_clusters: AgglomerativeClustering(n_clusters=num_clusters),
            'MeanShift': lambda num_clusters: MeanShift(),
            'DBSCAN': lambda num_clusters: DBSCAN(),
            'AffinityPropagation': lambda num_clusters: AffinityPropagation(),
            'Spectral': lambda num_clusters: SpectralClustering(n_clusters=num_clusters),
            'Birch': lambda num_clusters: Birch(n_clusters=num_clusters),
            'OPTICS': lambda num_clusters: OPTICS()
        }
        self.reduce_dimension = {
            'PCA': lambda features, dim_num, seed: PCA(n_components=dim_num, random_state=seed).fit_transform(features),
            'tSNE': lambda features, dim_num, seed: TSNE(n_components=dim_num, random_state=seed).fit_transform(features),
            'Dict': lambda features, dim_num, seed: DictionaryLearning(n_components=dim_num, random_state=seed).fit_transform(features),
            'KPCA': lambda features, dim_num, seed: KernelPCA(n_components=dim_num, random_state=seed).fit_transform(features),
            'IPCA': lambda features, dim_num, seed: IncrementalPCA(n_components=dim_num).fit_transform(features),
            'ICA': lambda features, dim_num, seed: FastICA(n_components=dim_num, random_state=seed).fit_transform(features),
            'SPCA': lambda features, dim_num, seed: SparsePCA(n_components=dim_num, random_state=seed).fit_transform(features),
            'FA': lambda features, dim_num, seed: FactorAnalysis(n_components=dim_num, random_state=seed).fit_transform(features),
        }
        self.cluster_algorithm_list = cluster_algorithm_list
        self.reduce_dimension_algorithm_list = reduce_dimension_algorithm_list
    
    def evaluate_clustering(self, y_true, y_pred):
        # purity and rand socre for clustering results
        cm = confusion_matrix(y_true, y_pred) # y_true, y_pred is array-like, shape (n_samples,)
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        rand_score = adjusted_rand_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        return purity, rand_score, ari # purity, rand_score, adjusted_rand_score

    def reduce_dimension_plot(self, 
            features, 
            label_series, 
            output_fn=None, 
            method='tSNE', 
            dim_num=2, 
            plot_dim=[0, 1]
        ):
        plt.figure(figsize=(4.5, 4.5), dpi=600)
        if output_fn is None:
            output_fn = f"{self.output_fn}_{self.model_name}_{method}"
        features_array = features.values 
        X_embedded = self.reduce_dimension[method](features_array, dim_num, 1207) # random seed 1207
        point_size = 6 + min(1200/len(label_series), 20)
        if all(isinstance(x, (float, int)) for x in label_series.to_list()):
            if len(label_series) >= 30:
                label_series = label_series.apply(lambda x: round(x, 1))
            label_set = sorted(list(set(label_series.to_list())))
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        else:
            label_set = [label[0] for label in sorted(label_series.value_counts().to_dict().items(), key=lambda x: x[1], reverse=True)]
            colors = plt.cm.rainbow(np.linspace(0.15, 0.85, len(label_set)))
        color_id = 0
        for label in label_set:
            idx = (label_series.to_numpy() == label).nonzero()
            plt.scatter(X_embedded[idx, plot_dim[0]], X_embedded[idx, plot_dim[1]], c=colors[color_id], label=f'label={label}', marker = '.', s=point_size)
            color_id += 1
        if len(label_set) <= 6:
            plt.legend(loc='best')
        plt.title(f"{output_fn}")
        plt.tight_layout()
        plt.savefig(f"{output_fn}.png")
        plt.close()
        X_embedded = pd.DataFrame(X_embedded, columns=[f'{method}_1', f'{method}_2'])
        return X_embedded.join(label_series, how='left')

    def cluster_features(self, 
            features, 
            algorithm, 
            num_clusters=None
        ):
        features_array = features.values 
        cluster_model = self.cluster_models[algorithm](num_clusters).fit(features_array)
        labels = cluster_model.labels_
        return pd.DataFrame(labels, columns=[f'{algorithm}_ID'])

if __name__ == "__main__":
    # check GPU
    print('CUDA available:', torch.cuda.is_available())  # Should be True
    print('CUDA capability:', torch.cuda.get_arch_list()) 
    print('GPU number:', torch.cuda.device_count())  # Should be > 0
    # random_seed
    training_random_seed = 1207
    np.random.seed(training_random_seed)
    torch.manual_seed(training_random_seed)
    torch.cuda.manual_seed(training_random_seed) 
    torch.cuda.manual_seed_all(training_random_seed)
    # params
    data_table = pd.read_csv(sys.argv[1])
    method = sys.argv[2]
    smiles_column = sys.argv[3]
    ## setup task
    output_fn = sys.argv[4]
    running_mode = sys.argv[5].split(':')[0]
    ## read the models
    encoders = {}
    for model_name in method.split(":"):
        if os.path.exists(f'{model_name}/GeminiMol.pt'):
            method_list = [model_name]
            from model.GeminiMol import GeminiMol
            encoders[model_name] = GeminiMol(
                model_name,
                depth = 0, 
                custom_label=None, 
                extrnal_label_list=['Cosine', 'Pearson', 'RMSE', 'Manhattan']
            )
        elif os.path.exists(f'{model_name}/backbone'):
            from model.CrossEncoder import CrossEncoder
            encoders[model_name] = CrossEncoder(
                model_name,
                candidate_labels = [
                    'LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 
                    'ShapeScore', 'ShapeOverlap', 'ShapeAggregation', 'CrossSim', 'CrossAggregation', 'CrossOverlap', 
                ]
            )
        elif model_name == "CombineFP":
            methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
            encoders[model_name] = Fingerprint(methods_list)
        else:
            methods_list = [model_name]
            encoders[model_name] = Fingerprint(methods_list)
    encoders_list = list(encoders.values())
    ## setup encoder
    predictor = Mol_Encoder(
        encoders_list, 
        standardize = True,
        smiles_column = smiles_column
    )
    ## setup analyzer
    analyzer = unsupervised_clustering(
            cluster_algorithm_list = [
                'Birch', 'AffinityPropagation', 
                'K-Means', 'hierarchical', 
                'DBSCAN', 'OPTICS', 
                'MeanShift', 'Spectral'
                ], 
            reduce_dimension_algorithm_list = ['tSNE']
        )
    ## setup label
    if running_mode in ['validate', 'plot']:
        label_column = sys.argv[5].split(':')[1]
        data_table.dropna(subset=[label_column], inplace=True)
        data_table.reset_index(drop=True, inplace=True)
        dataset, features_columns = predictor.encoder(data_table)
    elif running_mode == 'heatmap':
        dataset = predictor.prepare(data_table)
    elif running_mode == 'matrix':
        id_column = sys.argv[5].split(':')[1]
        if len(method.split(":")) == 1 and 'GeminiMol' in method:
            data_table.dropna(subset=[id_column], inplace=True)
            data_table.reset_index(drop=True, inplace=True)
            dataset, features_columns = predictor.encoder(data_table)
        else:
            raise RuntimeError(f"ERROR: matrix mode only support for 1 GeminiMol model!")
    else:
        ## encode
        dataset, features_columns = predictor.encoder(data_table)
    ## processing
    if running_mode == 'encode':
        dataset.to_csv(f"{output_fn}_encoding.csv", index=False)
    elif running_mode == 'matrix':
        id_column = sys.argv[5].split(':')[1]
        id_list = dataset[id_column].to_list()
        matrix = pd.DataFrame(index=id_list, columns=id_list)
        features_dict = {
            mol_id:dataset[dataset[id_column]==mol_id][features_columns].values[0] 
            for mol_id in id_list
            }
        for ref_no in range(len(id_list)):
            ref_id = id_list[ref_no]
            for query_no in range(len(id_list[ref_no:])):
                query_id = id_list[ref_no+query_no]
                pearson_similarity = pearsonr(features_dict[ref_id], features_dict[query_id])[0]
                matrix.loc[ref_id, query_id] = pearson_similarity
                matrix.loc[query_id, ref_id] = pearson_similarity
        matrix.to_csv(f"{output_fn}_data.csv", index=True, header=True, sep=',')        
    elif running_mode == 'cluster':
        cluster_num = sys.argv[5].split(':')[1]
        for cluster_algorithm in analyzer.cluster_algorithm_list:
            labels = analyzer.cluster_features(dataset, cluster_algorithm, cluster_num)
            dataset = dataset.join(labels, how='left')
        dataset.to_csv(f"{output_fn}_clusters.csv", index=False, header=True, sep=',')
    elif running_mode == 'visualise':
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
        print(result_df)
        result_df.to_csv(f'{output_fn}/feat_stat.csv', index=True)
        # plot shade range graph for Mean and lines for Max/Min
        plt.figure(figsize=(3.6*1.0, 3.2*1.0), dpi=600) # 
        plt.plot(result_df.index, result_df['Mean'], color='#588AF5', label='Mean', linewidth=0.3)
        plt.plot(result_df.index, result_df['Max'], color='#D861F5', label='Max', linewidth=0.3)
        plt.plot(result_df.index, result_df['Min'], color='#F55A46', label='Min', linewidth=0.3)
        plt.fill_between(result_df.index, result_df['Mean'] + result_df['Std'], result_df['Mean'] - result_df['Std'], color='#F5D690', linewidth=0, alpha=0.8)
        plt.fill_between(result_df.index, result_df['Mean'] + 3 * result_df['Std'], result_df['Mean'] - 3* result_df['Std'], color='#6CF5B2', linewidth=0, alpha=0.3)
        plt.ylim(result_df['Min'].min(), result_df['Max'].max())
        plt.xlim(0, len(result_df.index)+1)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(f'{output_fn}/feat_stat.png')
        plt.close()
    elif running_mode == 'validate':
        if int(sys.argv[5].split(':')[2]):
            assumed_label_ratio = int(sys.argv[5].split(':')[2])
        else:
            assumed_label_ratio = 3
            print(f'Warning: assumed_label_ratio do not set, the default ratio is 3.')
        if all(isinstance(x, (float, int)) for x in dataset[label_column].to_list()):
            if len(set(dataset[label_column].to_list())) >= 30:
                dataset[label_column] = dataset[label_column].apply(lambda x: round(x, 1))
        validated_res = pd.DataFrame(
            index=analyzer.cluster_algorithm_list, 
            columns=['purity', 'rand_score', 'adjusted_rand_score']
        )
        label_number = len(set(dataset[label_column].to_list()))
        if label_number == 2:
            dataset[label_column] = dataset[label_column].replace({'positive': 1, 'negative': 0})
            labels_list = dataset[label_column].to_list()
        else:
            labels_list = LabelEncoder().fit_transform(dataset[label_column].to_list())
        for cluster_algorithm in analyzer.cluster_algorithm_list:
            pred_labels = analyzer.cluster_features(dataset[features_columns], cluster_algorithm, num_clusters=label_number*assumed_label_ratio)
            dataset = dataset.join(pred_labels, how='left')
            purity, rand_score, ari = analyzer.evaluate_clustering(
                dataset[f'{cluster_algorithm}_ID'].to_list(), 
                labels_list
            )
            validated_res.loc[cluster_algorithm, 'purity'] = purity
            validated_res.loc[cluster_algorithm, 'rand_score'] = rand_score
            validated_res.loc[cluster_algorithm, 'adjusted_rand_score'] = ari
        validated_res.to_csv(f"{output_fn}_statistics.csv", index=True, header=True, sep=',')   
        dataset.to_csv(f"{output_fn}_clusters.csv", index=False, header=True, sep=',')
    elif running_mode == 'plot':
        if sys.argv[5].split(':')[2]:
            plot_mode = sys.argv[5].split(':')[2]
        else:
            plot_mode = 'normal'
        if plot_mode == 'concise':
            std_values = dataset[features_columns].std()
            std_top = std_values.sort_values(
                ascending=False
                )[:max(len(features_columns)//10, 100)].index.to_list()
            features_dataset = dataset[std_top]
        else:
            features_dataset = dataset[features_columns]
        for method in analyzer.reduce_dimension_algorithm_list:
            features_table = analyzer.reduce_dimension_plot(features_dataset, dataset[label_column], output_fn = f"{output_fn}_{method}", method=method, dim_num=2).join(dataset[smiles_column], how='left')
            features_table.to_csv(f"{output_fn}_{method}.csv", index=False)
    elif running_mode == 'heatmap':
        id_column = sys.argv[5].split(':')[1]
        smiles_list = dataset[smiles_column].to_list()
        id_list = dataset[id_column].to_list()
        if len(smiles_list) >= 300:
            print(f"Warning: {len(smiles_list)} molecules is too large for heatmap.")
            print(f"Warning: we need long time to calculate the similarity matrix.")
        matrix_dict = {}
        for encoder_name, encoder in encoders.items():
            encoder_basename = encoder_name.split('/')[-1]
            for metric in encoder.similarity_metrics_list:
                matrix_dict.update({f"{encoder_basename}_{metric}" : pd.DataFrame(index=id_list, columns=id_list)})
        ref_smiles_dict = dict(zip(dataset[id_column], dataset[smiles_column]))
        for encoder_name, encoder in encoders.items():
            for ref_mol, ref_smiles in ref_smiles_dict.items():
                pred_res = encoder.virtual_screening([ref_smiles], dataset, reverse=False, smiles_column=smiles_column, similarity_metrics=encoder.similarity_metrics_list)
                print(f"NOTE: {encoder_name} {ref_mol} finished.")
                pred_res = pred_res.set_index(id_column)
                encoder_basename = encoder_name.split('/')[-1]
                for metric in encoder.similarity_metrics_list:
                    matrix_dict[f"{encoder_basename}_{metric}"][ref_mol] = pred_res[metric]
        if len(smiles_list) <= 100:
            reshape_table = pd.DataFrame(columns = ['ref_id', 'query_id', 'ref_smiles', 'query_smiles'])
            for score_name, score_table in matrix_dict.items():
                score_table = score_table.stack().reset_index()
                score_table.columns = ['query_id', 'ref_id', score_name]
                score_table['ref_smiles'] = score_table['ref_id'].apply(lambda x: ref_smiles_dict[x])
                score_table['query_smiles'] = score_table['query_id'].apply(lambda x: ref_smiles_dict[x])
                reshape_table = pd.merge(reshape_table, score_table, how='outer', on=['ref_id', 'query_id', 'ref_smiles', 'query_smiles']) 
            reshape_table.to_csv(f"{output_fn}_data.csv", index=False)
        import pickle
        with open(f'{output_fn}_data.pkl', 'wb') as file:
            pickle.dump(matrix_dict, file)
        with open(f'{output_fn}_data.pkl', 'rb') as file:
            matrix_dict = pickle.load(file)
        for score_name in list(matrix_dict.keys()):
            plt.figure(figsize=(6.9, 6.2), dpi=600)
            score_array = np.array(matrix_dict[score_name], dtype=float)
            plt.imshow(score_array, vmin=0.0, vmax=1.0)
            if len(id_list) <= 20:
                plt.xticks(range(len(id_list)), id_list, rotation=45)
                plt.yticks(range(len(id_list)), id_list)
            plt.title(f"{output_fn}_{score_name}", fontsize='large', fontweight='bold')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{output_fn}_{score_name}.png")
            plt.close()
    



