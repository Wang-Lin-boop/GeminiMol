import os
import sys
import json
from random import choice, randint
import pandas as pd
import dask.dataframe as dd
import numpy as np
import random
import threading
import concurrent.futures
from rdkit import Chem
from utils.chem import is_valid_smiles
from matplotlib import colors, cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN, AffinityPropagation, SpectralClustering, Birch, OPTICS
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from scipy.cluster import hierarchy

class CEO_DATA:
    def __init__(self, index_set, features_columns=None, split_ratio={0: 0.55, 1: 0.3, 2: 0.15}, special_score_list=None, pair_score_list=None):
        if features_columns is None:
            features_columns = ['MCS', 'ShapeScore']
        index_set['list'] = index_set['list'].apply(json.loads)
        index_set.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
        self.train_smiles = list(set(index_set[index_set['assign']=='train']['smiles'].to_list()))
        self.val_smiles = list(set(index_set[index_set['assign']=='val']['smiles'].to_list()))
        self.test_smiles = list(set(index_set[index_set['assign']=='test']['smiles'].to_list()))
        self.ceo_map = pd.DataFrame(index=index_set['smiles'].to_list(), columns=index_set['smiles'].to_list())
        self.index_set = index_set.set_index('smiles')
        self.features_columns = features_columns
        self.features_dict = {self.features_columns[i]: i for i in range(len(self.features_columns))}
        self.column_list = ['smiles1', 'smiles2']
        self.split_ratio = split_ratio
        self.special_score_list = special_score_list
        self.special_score_dict = {
            "ShapeOverlap": lambda score_dict: (min(score_dict['LCMS2A1Q_MIN'], score_dict['MCMM1AM_MIN']) + score_dict['ShapeScore'])/2,
            "ShapeDistance": lambda score_dict: min(score_dict['LCMS2A1Q_MIN'], score_dict['MCMM1AM_MIN']),
            "ShapeAggregation": lambda score_dict: (score_dict['MCMM1AM_MAX'] + score_dict['LCMS2A1Q_MAX'])/2,
        }
        self.pair_score_list = pair_score_list
        self.pair_score_dict = {
            "CrossSim": lambda forward_dict, reverse_dict: (forward_dict['ShapeScore'] + reverse_dict['ShapeScore'])/2,
            "CrossDist": lambda forward_dict, reverse_dict: (forward_dict['ShapeDistance'] + reverse_dict['ShapeDistance'])/2,
            "CrossOverlap": lambda forward_dict, reverse_dict: (forward_dict['ShapeOverlap'] + reverse_dict['ShapeOverlap'])/2,
            "CrossAggregation": lambda forward_dict, reverse_dict: (forward_dict['ShapeAggregation'] + reverse_dict['ShapeAggregation'])/2,
        }

    def update(self, CEO_dataset):
        assert set(self.features_columns).issubset(CEO_dataset.columns), f"Error: the CEO dataset must be contain these columns: {self.features_columns}."
        chunks = np.array_split(CEO_dataset, 128) # split the dataset into 128 chunks
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._update_chunk, chunk)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                pass

    def _update_chunk(self, chunk):
        chunk = chunk.dropna() # delete NaN
        features_df = chunk.loc[:, self.features_columns]
        features_arr = features_df.to_numpy()
        smiles1_arr = chunk['smiles1'].values
        smiles2_arr = chunk['smiles2'].values
        self.ceo_map.loc[smiles1_arr, smiles2_arr] = features_arr # row is query mol, columns is ref mol
          
    def extract(self, feature):
        assert feature in self.features_columns, f"Error: the specified feature {feature} is not in the feature columns."
        feature_map = self.ceo_map.applymap(lambda x: x[self.features_dict[feature]], na_action='ignore')
        return feature_map

    def _calculate_score(self, score_list):
        score_dict = {self.features_columns[i]: score_list[i] for i in range(len(self.features_columns))}
        for special_score in self.special_score_list:
            score_dict[special_score] = self.special_score_dict[special_score](score_dict)
        return score_dict

    def pair_score(self, row_smiles, col_smiles):
        forward = self.ceo_map.loc[row_smiles, col_smiles] 
        reverse = self.ceo_map.loc[col_smiles, row_smiles]
        if type(forward) is list and type(reverse) is list:
            forward_dict = self._calculate_score(forward)
            forward_dict['smiles1'] = row_smiles
            forward_dict['smiles2'] = col_smiles
            reverse_dict = self._calculate_score(reverse)
            reverse_dict['smiles1'] = col_smiles
            reverse_dict['smiles2'] = row_smiles
            pair_score_dict = {}
            for pair_score in self.pair_score_list:
                pair_score_dict[pair_score] = self.pair_score_dict[pair_score](forward_dict, reverse_dict)
            return {**forward_dict, **pair_score_dict}, {**reverse_dict, **pair_score_dict}
        else:
            return None, None

    def retrieval_data(self, df, value, count, assign):
        positions = np.where(df == value)
        index_column_tuples = list(zip(positions[0], positions[1]))
        print(f"Note: Number of the data points (assign: {assign}, ellipse: {value}) is {len(index_column_tuples)}.")
        assert len(index_column_tuples) >= count, "Error: The amount of data in the data map is less than the amount required."
        random_positions = random.sample(index_column_tuples, count)
        data_list = []
        row_list = self.ceo_map.index.to_list()
        column_list = self.ceo_map.columns.to_list()
        for row, col in random_positions:
            row_smiles = row_list[row]
            col_smiles = column_list[col]
            forward_dict, reverse_dict = self.pair_score(row_smiles, col_smiles)
            if forward_dict is not None and reverse_dict is not None:
                data_list.append(forward_dict)
                data_list.append(reverse_dict)
        result_dataset = pd.DataFrame.from_records(data_list)
        result_dataset['assign'] = assign
        return result_dataset

    def lookup_smiles(self, smiles, probability=0.5):
        if random.random() < probability:  # The probability of randomization is 0.5
            try:
                smiles_list = self.index_set.at[smiles, 'list']
                return_smiles = choice(smiles_list)
                if is_valid_smiles(return_smiles):
                    return return_smiles
                else:
                    mol = Chem.MolFromSmiles(smiles) 
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=True)
            except:
                try:
                    mol = Chem.MolFromSmiles(smiles) 
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=True)
                except:
                    return 'smiles_unvaild'
        else:
            return smiles

    def process_ellipse(self, final_dataset, ellipse_Indicator, ratio):
        datasets = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for map_obj, size, assign in self.plan:
                if size <= 0:
                    pass
                else:
                    futures.append(executor.submit(self.retrieval_data, map_obj, ellipse_Indicator, int(size*ratio), assign))
            for future in concurrent.futures.as_completed(futures):
                datasets.append(future.result())
        final_dataset = pd.concat([final_dataset] + datasets, ignore_index=True)
        return final_dataset

    def get_sample_plan(self, training_size=5000, val_size=5, test_size=50, cross_size=50):
        # find the data index
        ellipse_map = self.extract('ellipse')
        training_map = ellipse_map.loc[self.train_smiles, self.train_smiles]
        val_map = ellipse_map.loc[self.val_smiles, self.val_smiles]
        test_map = ellipse_map.loc[self.test_smiles, self.test_smiles]
        cross_map = ellipse_map.loc[self.train_smiles, self.val_smiles+self.test_smiles]
        return [(training_map, int(training_size*3000), 'train'), (val_map, int(val_size*3000), 'val'), (test_map, int(test_size*3000), 'test'), (cross_map, int(cross_size*3000), 'cross')]

    def sample(self, training_size=5000, val_size=5, test_size=50, cross_size=50):
        # the final size of dataset is ( size * 3000 ) * ratio * 2 (swapping row and column)
        final_dataset = pd.DataFrame(columns=['smiles1', 'smiles2', 'assign']+self.features_columns+self.special_score_list+self.pair_score_list)
        self.plan = self.get_sample_plan(training_size=training_size, val_size=val_size, test_size=test_size, cross_size=cross_size)
        for ellipse_Indicator, ratio in self.split_ratio.items():
            final_dataset = self.process_ellipse(final_dataset, ellipse_Indicator, ratio)
        return final_dataset

    def balanced_sample(self, training_size=5000, val_size=5, test_size=50, cross_size=50):
        # the final size of dataset is ( size * 3000 ) * ratio * 2 (swapping row and column)
        final_dataset = pd.DataFrame(columns=['smiles1', 'smiles2', 'assign']+self.features_columns+self.special_score_list+self.pair_score_list)
        self.plan = self.get_sample_plan(training_size=training_size, val_size=val_size, test_size=test_size, cross_size=cross_size)
        for map_obj, size, assign in self.plan:
            random_rows = np.random.choice(map_obj.index, size=size, replace=True)
            random_cols = np.random.choice(map_obj.columns, size=size, replace=True)
            positions_set = set(zip(random_rows, random_cols))
            while len(positions_set) < size:
                row = np.random.choice(map_obj.index)
                col = np.random.choice(map_obj.columns)
                positions_set.add((row, col))
            random_positions = list(positions_set)
            data_list = []
            for row_smiles, col_smiles in random_positions:
                forward_dict, reverse_dict = self.pair_score(row_smiles, col_smiles)
                data_list.append(forward_dict)
                data_list.append(reverse_dict)
            result_dataset = pd.DataFrame.from_records(data_list)
            result_dataset['assign'] = assign
            final_dataset = pd.concat([final_dataset, result_dataset], ignore_index=True)
        return final_dataset

    def expand(self, dataset, probability=0.5, num_cpus=24):
        del self.ceo_map
        final_dataset = dd.from_pandas(dataset, npartitions=num_cpus)
        for column in self.column_list:
            final_dataset[column] = final_dataset[column].apply(lambda x: self.lookup_smiles(x, probability=probability))
            final_dataset = final_dataset[final_dataset[column]!='smiles_unvaild']
        final_dataset.drop_duplicates(subset=self.column_list, keep='first', inplace=True)
        return final_dataset.compute()

    def cluster(self, data_set, method, n_clusters=None):
        if method.lower() == 'kmeans':
            cluster_alg = KMeans(n_clusters=n_clusters)
            cluster_labels = cluster_alg.fit_predict(data_set)
        elif method.lower() == 'meanshift':
            cluster_alg = MeanShift()
            cluster_labels = cluster_alg.fit_predict(data_set)
        elif method.lower() == 'dbscan':
            cluster_alg = DBSCAN(eps=0.5, min_samples=5)
            distance_matrix = 1 - data_set.values
            cluster_labels = cluster_alg.fit_predict(distance_matrix)
        elif method.lower() == 'affinity':
            cluster_alg = AffinityPropagation()
            cluster_labels = cluster_alg.fit_predict(data_set)
        elif method.lower() == 'spectral':
            cluster_alg = SpectralClustering(n_clusters=n_clusters)
            cluster_labels = cluster_alg.fit_predict(data_set)
        elif method.lower() == 'birch':
            cluster_alg = Birch(n_clusters=n_clusters)
            cluster_labels = cluster_alg.fit_predict(data_set)
        elif method.lower() == 'optics':
            cluster_alg = OPTICS(min_samples=5)
            distance_matrix = 1 - data_set.values
            cluster_labels = cluster_alg.fit_predict(distance_matrix)
        else:
            # hierarchical_clustering
            cluster_alg = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage=method, affinity='precomputed', compute_full_tree=True) # linkage may be 'average'、'complete'、'single'、'ward'、'weighted', 'centroid'、'median' or 'ward_d'
            distance_matrix = 1 - data_set.values
            cluster_alg.fit_predict(distance_matrix)
            cluster_labels = hierarchy.cut_tree(Z=cluster_alg.children_, n_clusters=n_clusters)
            cluster_labels = cluster_labels.reshape(-1,)
        return cluster_labels

    def clean_map(self):
        # processing ceo_map
        #################################
        # the diversity of ref mol is more important than query mol
        col_threshold = 500 # delete ref mols with more than 500 unmatched query mol 
        row_threshold = 0  # delete query mols with any unmatched ref mol (after the col cleaning)
        col_nans = self.ceo_map.isnull().sum(axis=0) # the nan for per col (ref mol)
        cols_to_drop = col_nans[col_nans > col_threshold].index
        self.ceo_map = self.ceo_map.drop(index=cols_to_drop, columns=cols_to_drop)
        row_nans = self.ceo_map.isnull().sum(axis=1) # the nan for per row (query mol)
        rows_to_drop = row_nans[row_nans > row_threshold].index
        self.ceo_map = self.ceo_map.drop(index=rows_to_drop, columns=rows_to_drop)
        #################################

    def plot_method_heatmap(self, data_set, feature, method, n_clusters):
        cluster_labels = self.cluster(data_set, method, n_clusters)
        # sort the rows and columns of the similarity matrix based on the cluster labels
        sorted_indices = np.argsort(cluster_labels)
        similarity_matrix = data_set.values[sorted_indices][:, sorted_indices]
        # plot heatmap 
        norm = colors.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots(figsize=(25.6, 19.2), dpi=600)
        im = ax.imshow(similarity_matrix)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax, shrink=0.9, aspect=40, location='right', anchor=(0.1, 0.5))
        cbar.ax.tick_params(labelsize=28)
        ax.set_title(f'DeepShape {feature} {method} HeatMap', fontsize=40)
        ax.set_xlabel('Reference Molecule', fontsize=36)
        ax.set_ylabel('Query Molecule', fontsize=36)
        ax.tick_params(axis='both', which='major', labelsize=28)
        plt.savefig(f'{jobname}-{feature}-{method}-CEO-Map.png', dpi=300)
        plt.close()

    def plot_feature_heatmap(self, jobname, feature, method_list, n_clusters):
        # plot heatmap
        if os.path.exists(f'{jobname}-{feature}-CEO-raw.csv'):
            data_set = pd.read_csv(f'{jobname}-{feature}-CEO-raw.csv', index_col=0)
        else:
            num_missing = self.ceo_map.isna().sum().sum()
            if  num_missing > 0:
                self.clean_map()
            data_set = self.extract(feature)
            data_set.to_csv(f'{jobname}-{feature}-CEO-raw.csv', index=True, header=True, sep=',')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for method in method_list:
                futures.append(executor.submit(self.plot_method_heatmap, data_set, feature, method, n_clusters))
            for _ in concurrent.futures.as_completed(futures):
                pass
            
    def plot_heatmap(self, jobname, feature_list=['ShapeScore'], method_list=['affinity'], n_clusters=None):
        threads = []
        for feature in feature_list:
            t = threading.Thread(target=self.plot_feature_heatmap, args=(jobname, feature, method_list, n_clusters))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

if __name__ == "__main__":
    ##### read index from json (generated by tokenizer.py)
    index_set = pd.read_json(sys.argv[1], orient='records')
    jobname = sys.argv[2]
    dataset = CEO_DATA(index_set, features_columns=['LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 'MCS', 'ShapeScore', 'ellipse'], special_score_list=["ShapeDistance", "ShapeOverlap", 'ShapeAggregation'], pair_score_list=["CrossSim", "CrossDist", "CrossOverlap", 'CrossAggregation'])
    # read data
    if os.path.exists(f'{jobname}-Map.pkl'):
        dataset.ceo_map = pd.read_pickle(f'{jobname}-Map.pkl')
    else:
        #### read the CEO data per ref_mol
        CEO_datasets_info = pd.read_csv(sys.argv[3], header=None, names=['Number', 'Path'])
        CEO_datasets_info = CEO_datasets_info[CEO_datasets_info['Number']>=36000]
        for filepath in CEO_datasets_info['Path'].to_list():
            if os.path.exists(filepath):
                print(f"NOTE: feed {filepath} to CEO map.")
                dataset.update(pd.read_csv(filepath))
        dataset.ceo_map.to_pickle(f'{jobname}-Map.pkl')
    if sys.argv[4] == "map":
        # plot heatmap
        dataset.plot_heatmap(jobname, feature_list=['LCMS2A1Q_MAX', 'LCMS2A1Q_MIN', 'MCMM1AM_MAX', 'MCMM1AM_MIN', 'MCS', 'ShapeScore'], method_list=['average', 'single', 'ward', 'weighted', 'centroid', 'affinity', 'birch', 'optics', 'kmeans'], n_clusters=300)
    elif sys.argv[4] == "data":
        # sample data
        if not os.path.exists(f'{jobname}-ceo_training.csv'):
            # test_dataset = dataset.sample(training_size=0, val_size=2, test_size=8, cross_size=20)
            # test_dataset.to_csv(f'{jobname}-ceo_test.csv', index=False, header=True, sep=',')
            train_dataset = dataset.sample(training_size=6000, val_size=0, test_size=0, cross_size=0)
            train_dataset.to_csv(f'{jobname}-training_graph.csv', index=False, header=True, sep=',')
            train_dataset = dataset.expand(train_dataset, probability=0.9, num_cpus=24)
            train_dataset.to_csv(f'{jobname}-training_smiles.csv', index=False, header=True, sep=',')
            
            



