import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN, AffinityPropagation, SpectralClustering, Birch, OPTICS
from sklearn.metrics import confusion_matrix, adjusted_rand_score
# from umap import UMAP
from utils.chem import gen_standardize_smiles, check_smiles_validity

class unsupervised_clustering:
    def __init__(self, model_name, cluster_algorithm_list=['AffinityPropagation'], reduce_dimension_algorithm_list=['tSNE', 'PCA']):
        if os.path.exists(f'{model_name}/GeminiMol.pt'):
            from model.GeminiMol import GeminiMol
            self.predictor = GeminiMol(model_name)
        else:
            from utils.fingerprint import Fingerprint
            if model_name == "CombineFP":
                methods_list = ["ECFP4", "FCFP6", "AtomPairs", "TopologicalTorsion"]
            else:
                methods_list = [model_name]
            self.predictor = Fingerprint(methods_list)
        self.model_name = str(os.path.split(model_name)[-1].split('.')[0])
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
            'PCA': lambda features, seed: PCA(n_components=2, random_state=seed).fit_transform(features),
            'tSNE': lambda features, seed: TSNE(n_components=2, random_state=seed).fit_transform(features),
            # 'uMap': lambda features, seed: UMAP(n_components=2, random_state=seed).fit_transform(features)
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

    def prepare(self, dataset, smiles_column='smiles', standardize=False):
        if standardize:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:gen_standardize_smiles(x, kekule=False))
        else:
            dataset[smiles_column] = dataset[smiles_column].apply(lambda x:check_smiles_validity(x))
        dataset = dataset[dataset[smiles_column]!='smiles_unvaild']
        return dataset

    def reduce_dimension_plot(self, features, label_series, method='tSNE'):
        features_array = features.values 
        label_set = set(label_series.to_list())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(label_set)))
        X_embedded = self.reduce_dimension[method](features_array, 1207) # random seed 1207
        plt.figure(figsize=(4.5, 4.5), dpi=600)
        for i, label in enumerate(label_set):
            idx = (label_series.to_numpy() == label).nonzero()
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], c=colors[i], label=f'label={label}', marker = '.')
        plt.legend(loc='best')
        plt.title(f"{self.model_name} {method}")
        plt.tight_layout()
        plt.savefig(f"{self.output_fn}_{self.model_name}_{method}"+".png")
        plt.close()

    def cluster_features(self, features, algorithm, num_clusters=None):
        features_array = features.values 
        cluster_model = self.cluster_models[algorithm](num_clusters).fit(features_array)
        labels = cluster_model.labels_
        return pd.DataFrame(labels, columns=[f'{algorithm}_ID'])

    def validate(self, dataset, output_fn, prepare=True, standardize=False, smiles_column='smiles', label_column='Y'):
        self.output_fn = output_fn
        assert smiles_column in dataset.columns, f"Error: the input table must contain {smiles_column}."
        if prepare:
            dataset = self.prepare(dataset, smiles_column=smiles_column, standardize=standardize)
        features = self.predictor.extract_features(dataset, smiles_column=smiles_column)
        for reduce_dimension in self.reduce_dimension_algorithm_list:
            self.reduce_dimension_plot(features, dataset[label_column], method=reduce_dimension)
        validated_res = pd.DataFrame(index=self.cluster_algorithm_list, columns=['purity', 'rand_score', 'adjusted_rand_score'])
        label_number = len(set(dataset[label_column].to_list()))
        if label_number == 2:
            dataset[label_column] = dataset[label_column].replace({'positive': 1, 'negative': 0})
        for cluster_algorithm in self.cluster_algorithm_list:
            labels = self.cluster_features(features, cluster_algorithm, num_clusters=label_number)
            dataset = dataset.join(labels, how='left')
            purity, rand_score, ari = self.evaluate_clustering(dataset[f'{cluster_algorithm}_ID'].to_list(), dataset[label_column].to_list())
            validated_res.loc[cluster_algorithm, 'purity'] = purity
            validated_res.loc[cluster_algorithm, 'rand_score'] = rand_score
            validated_res.loc[cluster_algorithm, 'adjusted_rand_score'] = ari
        return dataset, validated_res

    def __call__(self, dataset, output_fn, cluster_num=10, prepare=True, standardize=False, smiles_column='smiles'):
        self.output_fn = output_fn
        assert smiles_column in dataset.columns, f"Error: the input table must contain {smiles_column}."
        if prepare:
            dataset = self.prepare(dataset, smiles_column=smiles_column, standardize=standardize)
        features = self.predictor.extract_features(dataset, smiles_column=smiles_column)
        for cluster_algorithm in self.cluster_algorithm_list:
            labels = self.cluster_features(features, cluster_algorithm, cluster_num)
            dataset = dataset.join(labels, how='left')
        return dataset
        
if __name__ == '__main__':
    data_table = pd.read_csv(sys.argv[1])
    model_name = sys.argv[2]
    output_fn = sys.argv[3]
    smiles_column = sys.argv[4]
    predictor = unsupervised_clustering(model_name, cluster_algorithm_list=['Birch', 'AffinityPropagation', 'K-Means', 'hierarchical', 'DBSCAN', 'OPTICS', 'MeanShift', 'Spectral'], reduce_dimension_algorithm_list=['tSNE', 'PCA']) # 'uMap',  # 
    if len(sys.argv) > 5:
        label_column = sys.argv[5]
        dataset, validated_res = predictor.validate(data_table, output_fn, prepare=True, standardize=True, smiles_column=smiles_column, label_column=label_column)
        validated_res.to_csv(f"{output_fn}_statistics.csv", index=True, header=True, sep=',')   
        dataset.to_csv(f"{output_fn}_clusters.csv", index=False, header=True, sep=',')
    else:
        label_column = None
        dataset = predictor(data_table, output_fn, cluster_num = 10, prepare=True, standardize=True, smiles_column=smiles_column)
        dataset.to_csv(f"{output_fn}_clusters.csv", index=False, header=True, sep=',')
     
    








