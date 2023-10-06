import os
import time
import json
import math
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr
import torch.nn as nn
from dgl import batch
from dgllife.utils import atomic_number, atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgllife.model.gnn.gat import GAT
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.wln import WLN
from dgllife.model.gnn.gnn_ogb import GNNOGB
from dgllife.model.gnn.pagtn import PAGTNGNN
from dgllife.model.gnn.mpnn import MPNNGNN
from dgllife.model.gnn.nf import NFGNN
from dgllife.model.gnn.graphsage import GraphSAGE
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax

class MolGraph_Encoder(nn.Module):
    def __init__(self, n_embed=128, n_heads=8, n_layers=12, dropout_rate=0.1, encoder_type='AttentiveFP', pooling_type='MLP', atom_feat_size=None, bond_feat_size=None):
        super().__init__()
        n_embed_layers = []
        n_head_layers = []
        dropout_rate_layers = []
        gnn_norm_layers = []
        for _ in range(n_layers):
            n_embed_layers.append(n_embed) 
            n_head_layers.append(n_heads)
            dropout_rate_layers.append(dropout_rate)
            gnn_norm_layers.append('right')
        self.GraphEncoder = nn.ModuleDict({
            'GAT': GAT(atom_feat_size, hidden_feats=n_embed_layers, num_heads=n_head_layers, feat_drops=dropout_rate_layers),
            'GCN': GCN(atom_feat_size, hidden_feats=n_embed_layers, gnn_norm=gnn_norm_layers, dropout=dropout_rate_layers),
            'AttentiveFP': AttentiveFPGNN(atom_feat_size, bond_feat_size, num_layers=n_layers, graph_feat_size=n_embed, dropout=dropout_rate),
            # 'vGIN': GNNOGB(in_edge_feats=bond_feat_size, num_node_types=atom_feat_size, hidden_feats=n_embed, n_layers=n_layers, dropout=dropout_rate, gnn_type='gin'),
            # 'vGCN': GNNOGB(in_edge_feats=bond_feat_size, num_node_types=atom_feat_size, hidden_feats=n_embed, n_layers=n_layers, dropout=dropout_rate, gnn_type='gcn'),
            'GraphSAGE': GraphSAGE(atom_feat_size, hidden_feats=n_embed_layers, dropout=dropout_rate_layers),
            'WLN': WLN(atom_feat_size, bond_feat_size, n_layers=n_layers, node_out_feats=n_embed),
            'MPNN': MPNNGNN(node_in_feats=atom_feat_size, node_out_feats=n_embed, edge_in_feats=bond_feat_size, edge_hidden_feats=64, num_step_message_passing=n_heads),
            # 'PAGTN': PAGTNGNN(node_in_feats=atom_feat_size, node_out_feats=n_embed, node_hid_feats =2*n_embed, edge_feats=bond_feat_size, depth=n_layers, nheads=n_heads, dropout=dropout_rate),
            'NF': NFGNN(in_feats=atom_feat_size, hidden_feats=n_embed_layers, dropout=dropout_rate_layers),
            })
        self.encoder_type = encoder_type
        self.pooling = nn.ModuleDict({
                    'AttentiveFP': AttentiveFPReadout(n_embed, dropout=0.0),
                    'MLP': MLPNodeReadout(n_embed, n_embed, n_embed, activation=nn.ReLU(), mode='mean'),
                    'WeightedPooling': WeightedSumAndMax(n_embed),
                })
        self.pooling.cuda()
        self.GraphEncoder.to('cuda')
        self.GraphEncoder_Dict = {
            'GAT': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type']),
            'GCN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type']),
            'AttentiveFP': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type']),
            # 'vGIN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'].to(torch.long), mol_graph.edata['bond_type']),
            # 'vGCN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'].to(torch.long), mol_graph.edata['bond_type']),
            'GraphSAGE': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type']),
            'WLN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type']),
            'MPNN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type']),
            # 'PAGTN': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type'])
            'NF': lambda mol_graph: self.GraphEncoder[self.encoder_type](mol_graph, mol_graph.ndata['atom_type']),
        }
        self.pooling_type = pooling_type
        self.pooling_Dict = {
            'AttentiveFP': lambda mol_graph, encoding: self.pooling[self.pooling_type](mol_graph, encoding),
            'MLP': lambda mol_graph, encoding: self.pooling[self.pooling_type](mol_graph, encoding),
            'WeightedPooling': lambda mol_graph, encoding: self.pooling[self.pooling_type](mol_graph, encoding),
        }
        
    def forward(self, mol_graph):
        encoding = self.GraphEncoder_Dict[self.encoder_type](mol_graph)
        return self.pooling_Dict[self.pooling_type](mol_graph, encoding)

class Shape2Score(nn.Module):
    def __init__(self, n_embed=256, dropout_rate=0.1):
        super(Shape2Score, self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(n_embed, n_embed*3),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*3),
                nn.Dropout(0.3),
                nn.Linear(n_embed*3, 1024),
                nn.GELU(),
                nn.Linear(1024, 128),
                nn.GELU(),
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Identity(),
            )
        self.cuda()
        
    def forward(self, features):
        score = self.decoder(features)
        prediction = torch.sigmoid(score).squeeze()
        return prediction

class BinarySimilarity(nn.Module):  
    def __init__(self, model_name, feature_list=['smiles1','smiles2'], label_dict={'MCS':0.4, 'Max':0.4, 'Mean':0.1, 'Min':0.1}, n_embed=128, n_heads=8, n_layers=12, batch_size=128, dropout_rate=0.1, attention_dropout_rate=0.1, encoder_type='AttentiveFP', pooling_type='MLP'):
        super(BinarySimilarity, self).__init__()
        torch.set_float32_matmul_precision('high') 
        self.model_name = model_name
        self.batch_size = batch_size
        self.label_list = list(label_dict.keys())
        self.feature_list = feature_list
        if encoder_type == 'AttentiveFP':
            self.pooling_type = 'AttentiveFP'
        else:
            self.pooling_type = pooling_type
        ## create MolGraph, MolecularGAT and ShapePooling module
        if encoder_type in ['vGIN', 'vGCN']:
            self.atom_featurizer = BaseAtomFeaturizer({'atom_type':atomic_number}) 
        else:
            self.atom_featurizer = BaseAtomFeaturizer({'atom_type':ConcatFeaturizer([atom_type_one_hot, atom_hybridization_one_hot, atom_formal_charge, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_type')
        self.Encoder = MolGraph_Encoder(n_embed=n_embed, n_heads=n_heads, n_layers=n_layers, dropout_rate=attention_dropout_rate, encoder_type=encoder_type, pooling_type=self.pooling_type, atom_feat_size=self.atom_featurizer.feat_size(feat_name='atom_type'), bond_feat_size=self.bond_featurizer.feat_size(feat_name='bond_type'))
        # create multiple decoders
        if self.pooling_type == 'WeightedPooling':
            encoding_size = n_embed*2
        else:
            encoding_size = n_embed
        self.Decoder = nn.ModuleDict()
        for label in self.label_list:
            self.Decoder[label] = nn.DataParallel(Shape2Score(n_embed=encoding_size*2, dropout_rate=dropout_rate))
        if os.path.exists(self.model_name):
            if os.path.exists(f'{self.model_name}/GraphShape.pt'):
                self.load_state_dict(torch.load(f'{self.model_name}/GraphShape.pt'))
        else:
            os.mkdir(self.model_name)
        ## set loss weight
        loss_weight = list(label_dict.values())
        sum_weight = np.sum(loss_weight)
        self.loss_weight = {key: value/sum_weight for key, value in zip(self.label_list, loss_weight)}

    def sents2tensor(self, input_sents):
        input_tensor = batch([smiles_to_bigraph(smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer).to('cuda') for smiles in input_sents])

        return input_tensor

    def forward(self, sent1, sent2, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # Concatenate input sentences
        input_sents = sent1 + sent2
        input_tensor = self.sents2tensor(input_sents).to('cuda')
        # Encode all sentences using the transformer
        features = self.Encoder(input_tensor)
        encoded1 = features[:batch_size]
        encoded2 = features[batch_size:]
        encoded = torch.cat((encoded1, encoded2), dim=1)
        # Calculate specific score between encoded smiles
        pred = {}
        for label in self.label_list:
            pred[label] = self.Decoder[label](encoded)
        return pred

    def encode(self, input_sents):
        # Encode all sentences using the transformer 
        input_tensor = self.sents2tensor(input_sents)
        features = self.Encoder(input_tensor)
        return features

    def decode(self, features, label_name, batch_size):
        encoded1 = features[:batch_size]
        encoded2 = features[batch_size:]
        encoded1 = features[:batch_size]
        encoded2 = features[batch_size:]
        if label_name == 'RMSE':
            return 1 - torch.sqrt(torch.mean((encoded1.view(-1, encoded1.shape[-1]) - encoded2.view(-1, encoded2.shape[-1])) ** 2, dim=-1))
        elif label_name == 'Cosine':
            encoded1_normalized = torch.nn.functional.normalize(encoded1, dim=-1)
            encoded2_normalized = torch.nn.functional.normalize(encoded2, dim=-1)
            return torch.nn.functional.cosine_similarity(encoded1_normalized, encoded2_normalized, dim=-1)
        elif label_name == 'Euclidean':
            return 1 - torch.sqrt(torch.sum((encoded1 - encoded2)**2, dim=-1))
        elif label_name == 'Manhattan':
            return 1 - torch.sum(torch.abs(encoded1 - encoded2), dim=-1)
        elif label_name == 'Minkowski':
            return 1 - torch.norm(encoded1 - encoded2, p=3, dim=-1)
        elif label_name == 'KLDiv':
            encoded1_normalized = torch.nn.functional.softmax(encoded1, dim=-1)
            encoded2_log_normalized = torch.nn.functional.log_softmax(encoded2, dim=-1)
            return 1 - (torch.sum(encoded1_normalized * (torch.log(encoded1_normalized + 1e-7) - encoded2_log_normalized), dim=-1))
        elif label_name == 'Pearson':
            mean1 = torch.mean(encoded1, dim=-1, keepdim=True)
            mean2 = torch.mean(encoded2, dim=-1, keepdim=True)
            std1 = torch.std(encoded1, dim=-1, keepdim=True)
            std2 = torch.std(encoded2, dim=-1, keepdim=True)
            encoded1_normalized = (encoded1 - mean1) / (std1 + 1e-7)
            encoded2_normalized = (encoded2 - mean2) / (std2 + 1e-7)
            return torch.mean(encoded1_normalized * encoded2_normalized, dim=-1)
        elif label_name in self.label_list:
            encoded = torch.cat((encoded1, encoded2), dim=1)
            return self.Decoder[label_name](encoded)
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)


    def extract(self, df, col_name, as_pandas=True):
        self.eval()
        data = df[col_name].tolist()
        with torch.no_grad():
            features_list = []
            for i in range(0, len(data), self.batch_size):
                input_sents = data[i:i+self.batch_size]
                input_tensor = self.sents2tensor(input_sents)
                features = self.Encoder(input_tensor)
                features_list += list(features.cpu().detach().numpy())
        if as_pandas == True:
            return pd.DataFrame(features_list)
        else:
            return features_list

    def predict(self, df, as_pandas=True):
        self.eval()
        with torch.no_grad():
            pred_values = {key:[] for key in self.label_list}
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                # Concatenate input sentences
                input_sents = sent1 + sent2
                features = self.encode(input_sents)
                for label_name in self.label_list:
                    pred = self.decode(features, label_name, len(rows))
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.label_list)
                return res_df
            else:
                return pred_values

    def evaluate(self, df):
        pred_table = self.predict(df, as_pandas=True)
        results = pd.DataFrame(index=self.label_list, columns=["RMSE","PEARSONR","SPEARMANR"])
        for label_name in self.label_list:
            pred_score = pred_table[label_name].tolist()
            true_score = df[label_name].tolist()
            results.loc[[label_name],['RMSE']] = round(math.sqrt(mean_squared_error(true_score, pred_score)), 6)
            results.loc[[label_name],['PEARSONR']] = round(pearsonr(pred_score, true_score)[0] , 6)
            results.loc[[label_name],['SPEARMANR']] = round(spearmanr(pred_score, true_score)[0] , 6)
        return results

    def fit(self, train_df, epochs, learning_rate=1.0e-3, val_df=None, optim_type='AdamW'):
        # Load the models and optimizer
        models = {
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
            'Adamax': torch.optim.Adamax,
            'Rprop': torch.optim.Rprop
        }
        if optim_type not in models:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        optimizer = models[optim_type](self.parameters(), lr=learning_rate)
        best_val_spearmanr = -1.0
        bak_val_spearmanr = -1.0
        batch_id = 0
        # setup warm up params
        num_warmup_steps = 30000
        warmup_factor = 0.1
        # Apply warm-up learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * warmup_factor 
        # setup consine LR params, lr = lr_max * 0.5 * (1 + cos( steps / T_max ))
        T_max = 10000
        # Set a counter to monitor the network reaching convergence
        counter = 0
        # triaining
        for epoch in range(epochs):
            val_spearmanr_list = []
            self.train()
            start = time.time()
            train_df = shuffle(train_df) # shuffle data set per epoch
            for i in range(0, len(train_df), self.batch_size):
                batch_id += 1
                rows = train_df.iloc[i:i+self.batch_size]
                sent1 = rows[self.feature_list[0]].to_list()
                sent2 = rows[self.feature_list[1]].to_list()
                assert len(sent1) == len(sent2), "The length of sent1 and sent2 must be the same."
                batch_size = len(sent1)
                pred = self.forward(sent1, sent2, batch_size)
                loss = torch.tensor(0.0, requires_grad=True).cuda()
                for label in self.label_list:
                    labels = torch.tensor(rows[label].to_list(), dtype=torch.float32).cuda()
                    label_loss = nn.MSELoss()(pred[label], labels) * self.loss_weight[label]
                    loss = torch.add(loss, label_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id % 500 == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {time.time()-start}, train loss: {loss.item()}")
                    if batch_id <= num_warmup_steps:
                        # Apply warm-up learning rate schedule
                        for param_group in optimizer.param_groups:
                            if warmup_factor < batch_id / num_warmup_steps:
                                param_group['lr'] = learning_rate * ( batch_id / num_warmup_steps) 
                    elif batch_id > num_warmup_steps:
                        # Apply cosine learning rate schedule
                        lr = learning_rate * ( warmup_factor + (1 - warmup_factor) * np.cos(float(batch_id % T_max)/T_max)) 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                if batch_id % 5000 == 0:
                    # Evaluation on validation set, if provided
                    if val_df is not None:
                        val_res = self.evaluate(val_df)
                        # self.train() # only for MPNN
                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                        print(val_res)
                        val_spearmanr = np.sum(val_res["SPEARMANR"].to_list())
                        val_spearmanr_list.append(val_spearmanr)
                        if val_spearmanr is None and os.path.exists(f'{self.model_name}/GraphShape.pt'):
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/GraphShape.pt'))
                        elif val_spearmanr > best_val_spearmanr:
                            best_val_spearmanr = val_spearmanr
                            torch.save(self.state_dict(), f"{self.model_name}/GraphShape.pt")
                if batch_id % T_max == 0:
                    if best_val_spearmanr <= bak_val_spearmanr:
                        counter += 1
                        if counter >= 2:
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/GraphShape.pt'))
                        elif counter >= 5:
                            break
                    else:
                        counter = 0
                        bak_val_spearmanr = best_val_spearmanr
        if val_df is not None:
            print(f"Best val SPEARMANR: {best_val_spearmanr}")
        self.load_state_dict(torch.load(f'{self.model_name}/GraphShape.pt'))

class GraphShape(BinarySimilarity):
    def __init__(self, model_name):
        self.model_name = model_name
        with open(f"{model_name}/graphshape_params.json", 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        with open(f"{model_name}/label_params.json", 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        super().__init__(model_name=model_name, label_dict=label_dict, **self.params)
        self.similarity_metrics_list = self.label_list + ['RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson'] 

    def create_database(self, query_smiles_table, smiles_column='smiles'):
        self.eval()
        data = query_smiles_table[smiles_column].tolist()
        query_smiles_table['features'] = None
        with torch.no_grad():
            for i in range(0, len(data), 2*self.batch_size):
                input_sents = data[i:i+2*self.batch_size]
                input_tensor = self.sents2tensor(input_sents)
                features = self.Encoder(input_tensor)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    query_smiles_table.at[i+j, 'features'] = features_list[j]
        return query_smiles_table

    def similarity_predict(self, shape_database, ref_smiles, ref_as_frist=False, as_pandas=True, similarity_metrics=None):
        self.eval()
        if similarity_metrics == None:
            similarity_metrics = self.similarity_metrics_list
        features_list = shape_database['features'].tolist()
        with torch.no_grad():
            pred_values = {key:[] for key in self.similarity_metrics_list}
            ref_features = self.encode([ref_smiles]).cuda()
            for i in range(0, len(features_list), self.batch_size):
                features_batch = features_list[i:i+self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                if ref_as_frist == False:
                    features = torch.cat((query_features, ref_features.repeat(len(features_batch), 1)), dim=0)
                else:
                    features = torch.cat((ref_features.repeat(len(features_batch), 1), query_features), dim=0)
                for label_name in self.similarity_metrics_list:
                    pred = self.decode(features, label_name, len(features_batch))
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=self.similarity_metrics_list)
                return res_df
            else:
                return pred_values 

    def virtual_screening(self, ref_smiles_list, query_smiles_table, reverse=False, smiles_column='smiles', similarity_metrics=None):
        shape_database = self.create_database(query_smiles_table, smiles_column=smiles_column)
        total_res = pd.DataFrame()
        for ref_smiles in ref_smiles_list:
            query_scores = self.similarity_predict(shape_database, ref_smiles, ref_as_frist=reverse, as_pandas=True, similarity_metrics=similarity_metrics)
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
        return total_res

    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        shape_features = self.extract(query_smiles_table, smiles_column, as_pandas=False)
        return pd.DataFrame(shape_features).add_prefix('GS_')
