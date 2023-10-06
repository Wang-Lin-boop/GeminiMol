# base
import os
import time
import json
import math
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
# for GraphEncoder (MolecularEncoder)
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, average_precision_score
from dgl import batch
from dgllife.utils import atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgllife.model.gnn.wln import WLN
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
# for retnet (MolDecoder)
import torch.nn.init as init
from rdkit import Chem
import rdkit.Chem.rdFMCS as FMCS
from collections import Counter

'''
Basic Modules

'''

class encoding2score(nn.Module):
    def __init__(self, 
            n_embed=1024, 
            dropout_rate=0.1, 
            expand_ratio=3,
            activation='GELU'
        ):
        super(encoding2score, self).__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'PReLU': nn.PReLU(),
        }
        self.decoder = nn.Sequential(
                nn.Linear(n_embed*2, n_embed*expand_ratio*2),
                activation_dict[activation],
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(dropout_rate*3),
                nn.Linear(n_embed*expand_ratio*2, 1024),
                activation_dict[activation],
                nn.Linear(1024, 128),
                activation_dict[activation],
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                activation_dict[activation],
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                activation_dict[activation],
                nn.Dropout(p=dropout_rate, inplace=False),
                nn.Linear(128, 128),
                activation_dict[activation],
                nn.Linear(128, 1),
                nn.Identity(),
            )
        self.cuda()
        
    def forward(self, features):
        score = self.decoder(features)
        prediction = torch.sigmoid(score).squeeze()
        return prediction

class SkipConnection(nn.Module):
    def __init__(self, feature_size, skip_layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.skip_layers = skip_layers
        if norm is None:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        elif norm == 'LayerNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.LayerNorm(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        elif norm == 'BatchNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.BatchNorm1d(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(skip_layers)
            ])
        self.ident = nn.Identity()
        self.cuda()
    
    def forward(self, x):
        ident = self.ident(x)
        for i in range(self.skip_layers):
            x = self.layers[i](x)
        return torch.cat([ident, x], dim=-1)

class MLP(nn.Module):
    def __init__(self, feature_size, layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.num_layers = layers
        if norm is None:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        elif norm == 'LayerNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.LayerNorm(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        elif norm == 'BatchNorm':
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.BatchNorm1d(feature_size),
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        self.cuda()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x

class GeminiMLP(nn.Module):
    def __init__(self, input_size, output_size, mlp_type='MLP', num_layers=3, activation='GELU', norm=None, dropout=0.0):
        super().__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'Softplus': nn.Softplus(),
            'Softsign': nn.Softsign(),
            'Hardshrink': nn.Hardshrink(),
            'Hardtanh': nn.Hardtanh(),
            'LogSigmoid': nn.LogSigmoid(),
            'Softshrink': nn.Softshrink(),
            'PReLU': nn.PReLU(),
            'Softmin': nn.Softmin(),
            'Softmax': nn.Softmax()            
        }
        try:
            activation = activation_dict[activation]
        except:
            raise ValueError(f"Error: undefined activation function {activation}.")
        if mlp_type == 'MLP':
            self.gemini_mlp = MLP(input_size, layers=num_layers, activation=activation, norm=norm, dropout=dropout)
            self.out = nn.Linear(input_size, output_size)
        elif mlp_type == 'SkipConnection':
            self.gemini_mlp = SkipConnection(input_size, skip_layers=num_layers, activation=activation, norm=norm, dropout=dropout)
            self.out = nn.Linear(2*input_size, output_size)
        self.cuda()
    
    def forward(self, x):
        x = self.gemini_mlp(x)
        return self.out(x)

class SelfAttentionPooling(nn.Module):
    def __init__(self, attention_size=16):
        super().__init__()
        self.attention_size = attention_size
        self.cuda()
    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape
        attention_scores = torch.tanh(nn.Linear(embedding_size, self.attention_size, bias=False).cuda()(x))
        attention_weights = torch.softmax(nn.Linear(self.attention_size, 1, bias=False).cuda()(attention_scores), dim=1)
        pooled = (x * attention_weights).sum(dim=1)
        return pooled

'''

MolecularEncoder

This is a graph encoder model consisting of a graph neural network from 
DGL and the MLP architecture in pytorch, which builds an understanding 
of a compound's conformational space by supervised learning of CSS data.

'''

class MolecularEncoder(nn.Module):
    def __init__(self,
        atom_feat_size = None, 
        bond_feat_size = None,
        num_features = 512,
        num_layers = 12,
        num_out_features = 1024,
        activation = 'GELU',
        readout_type = 'Weighted',
        integrate_layer_type = 'MLP',
        integrate_layer_num = 3,
        ):
        super().__init__()
        activation_dict = {
            'GELU': nn.GELU(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'ELU': nn.ELU(),
            'SELU': nn.SELU(),
            'CELU': nn.CELU(),
            'PReLU': nn.PReLU(),
        }
        # init the GeminiEncoder
        self.GeminiEncoder = WLN(
            atom_feat_size, 
            bond_feat_size, 
            n_layers=num_layers, 
            node_out_feats=num_features
        )
        # init the readout and output layers
        self.readout_type = readout_type
        if readout_type == 'Weighted':
            assert num_features*2 == num_out_features, "Error: num_features*2 must equal num_out_features for Weighted readout."
            self.readout = WeightedSumAndMax(num_features)
        elif readout_type == 'MLP':
            assert num_features == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict[activation],
                mode='mean'
            )
        elif readout_type == 'Mixed':
            self.readout = nn.ModuleDict({
                'Weighted': WeightedSumAndMax(num_features),
                'MLP': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
            self.output = GeminiMLP(
                num_features*3, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation 
            )
            self.output.cuda()
        elif readout_type == 'CombineMLP':
            assert num_features*3 == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = nn.ModuleDict({
                'Max': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='max'
                ),
                'Sum': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='sum'
                ),
                'Mean': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
        elif readout_type == 'MixedMLP':
            self.readout = nn.ModuleDict({
                'Max': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='max'
                ),
                'Sum': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='sum'
                ),
                'Mean': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
            self.output = GeminiMLP(
                num_features*3, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation 
            )
            self.output.cuda()
        elif readout_type == 'CombineWeighted':
            assert num_features*3 == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = nn.ModuleDict({
                'Weighted': WeightedSumAndMax(num_features),
                'MLP': MLPNodeReadout(
                    num_features, num_features, num_features, 
                    activation=activation_dict[activation], 
                    mode='mean'
                )
            })
        else:
            raise ValueError(f"Error: undefined readout type {readout_type}.")
        self.readout.cuda()
        self.GeminiEncoder.to('cuda')

    def forward(self, mol_graph):
        WLN_encoding = self.GeminiEncoder(mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type'])
        if self.readout_type == 'Mixed':
            mixed_readout = (self.readout['Weighted'](mol_graph, WLN_encoding), self.readout['MLP'](mol_graph, WLN_encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type == 'MixedMLP':
            mixed_readout = (self.readout['Max'](mol_graph, WLN_encoding), self.readout['Sum'](mol_graph, WLN_encoding), self.readout['Mean'](mol_graph, WLN_encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type == 'CombineMLP':
            return torch.cat([self.readout['Max'](mol_graph, WLN_encoding), self.readout['Sum'](mol_graph, WLN_encoding), self.readout['Mean'](mol_graph, WLN_encoding)], dim=1)
        elif self.readout_type == 'CombineWeighted':
            return torch.cat([self.readout['Weighted'](mol_graph, WLN_encoding), self.readout['MLP'](mol_graph, WLN_encoding)], dim=1)
        else:
            return self.readout(mol_graph, WLN_encoding)

class BinarySimilarity(nn.Module):  
    def __init__(self, 
        model_name, 
        feature_list = ['smiles1','smiles2'], 
        label_dict = {
            'ShapeScore':0.4, 
            'ShapeAggregation':0.2,
            'CrossSim':0.2, 
            'CrossAggregation':0.2
        }, 
        batch_size = 128, 
        num_layers = 12,
        num_features = 512,
        encoder_activation = 'GELU',
        readout_type = 'Mixed',
        encoding_features = 1024,
        decoder_activation = 'GELU',
        decoder_expand_ratio = 3,
        decoder_dropout_rate = 0.1,
        integrate_layer_type = 'MLP',
        integrate_layer_num = 0,
        ):
        super(BinarySimilarity, self).__init__()
        torch.set_float32_matmul_precision('high') 
        self.model_name = model_name
        self.batch_size = batch_size
        self.label_list = list(label_dict.keys())
        self.feature_list = feature_list
        ## set up featurizer
        self.atom_featurizer = BaseAtomFeaturizer({'atom_type':ConcatFeaturizer([atom_type_one_hot, atom_hybridization_one_hot, atom_formal_charge, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic])}) 
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_type')
        ## create MolecularEncoder and ShapePooling module
        self.Encoder = MolecularEncoder(
            atom_feat_size=self.atom_featurizer.feat_size(feat_name='atom_type'), 
            bond_feat_size=self.bond_featurizer.feat_size(feat_name='bond_type'), 
            num_layers=num_layers,
            num_features=num_features,
            activation=encoder_activation,
            readout_type=readout_type,
            integrate_layer_type=integrate_layer_type,
            integrate_layer_num=integrate_layer_num,
            num_out_features = encoding_features
        )
        # create multiple decoders
        self.Decoder = nn.ModuleDict()
        for label in self.label_list:
            self.Decoder[label] = encoding2score(
                n_embed=encoding_features, 
                dropout_rate=decoder_dropout_rate,
                expand_ratio=decoder_expand_ratio,
                activation=decoder_activation
                )
        if os.path.exists(self.model_name):
            if os.path.exists(f'{self.model_name}/GeminiMol.pt'):
                self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
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
        # Encode all sentences using the encoder
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
        # Encode all sentences using the encoder 
        input_tensor = self.sents2tensor(input_sents)
        features = self.Encoder(input_tensor)
        return features

    def decode(self, features, label_name, batch_size):
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

    def fit(self, 
        train_df, 
        epochs, 
        learning_rate=1.0e-4, 
        val_df=None, 
        optim_type='AdamW',
        num_warmup_steps=30000,
        T_max=10000,
        fine_tune=False
        ):
        # Load the models and optimizer
        optimizers_dict = {
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
            'Adamax': torch.optim.Adamax,
            'Rprop': torch.optim.Rprop
        }
        if optim_type not in optimizers_dict:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # Set up the optimizer
        optimizer = optimizers_dict[optim_type](self.parameters(), lr=learning_rate)
        # set up the early stop params
        if fine_tune == True:
            bacth_group = 10
            mini_epoch = 200
            best_val_spearmanr = self.evaluate(val_df)["SPEARMANR"].sum()
            bak_val_spearmanr = best_val_spearmanr
        else:
            bacth_group = 100
            mini_epoch = 1000
            best_val_spearmanr = -1.0
            bak_val_spearmanr = -1.0
        batch_id = 0
        # setup warm up params
        num_warmup_steps = num_warmup_steps
        warmup_factor = 0.1
        # Apply warm-up learning rate
        if num_warmup_steps > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor 
        # setup consine LR params, lr = lr_max * 0.5 * (1 + cos( steps / T_max ))
        T_max = T_max
        # Set a counter to monitor the network reaching convergence
        counter = 0
        # training
        for epoch in range(epochs):
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
                if batch_id % bacth_group == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {time.time()-start}, train loss: {loss.item()}")
                    # Set up the learning rate scheduler
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
                if batch_id % mini_epoch == 0:
                    # Evaluation on validation set, if provided
                    if val_df is not None:
                        val_res = self.evaluate(val_df)
                        self.train() # only for MPNN and the final GeminiMol training 
                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                        print(val_res)
                        val_spearmanr = np.sum(val_res["SPEARMANR"].to_list())
                        if val_spearmanr is None and os.path.exists(f'{self.model_name}/GeminiMol.pt'):
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
                        elif val_spearmanr > best_val_spearmanr:
                            best_val_spearmanr = val_spearmanr
                            torch.save(self.state_dict(), f"{self.model_name}/GeminiMol.pt")
                if batch_id % T_max == 0:
                    if best_val_spearmanr <= bak_val_spearmanr:
                        counter += 1
                        if counter >= 2:
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
                        elif counter >= 5:
                            break
                    else:
                        counter = 0
                        bak_val_spearmanr = best_val_spearmanr
        if val_df is not None:
            val_res = self.evaluate(val_df)
            print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
            print(val_res)
            val_spearmanr = np.sum(val_res["SPEARMANR"].to_list())
            if val_spearmanr > best_val_spearmanr:
                best_val_spearmanr = val_spearmanr
                torch.save(self.state_dict(), f"{self.model_name}/GeminiMol.pt")
            print(f"Best val SPEARMANR: {best_val_spearmanr}")
        self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
    
    def strolling(self, 
        train_df, 
        val_df,
        prefix = random.randint(1000, 9999),
        epochs = 1, 
        keep_origin = True,
        output_path = None,
        learning_rate = 1.0e-4, 
        optim_type = 'AdamW',
        T_max = 10000,
        schedule_unit = 50
        ):
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
        if output_path is None:
            output_path = f"{self.model_name}"
        # Set up the optimizer
        optimizer = models[optim_type](self.parameters(), lr=learning_rate)
        batch_id = 0
        best_score = -1.0
        model_list = []
        if keep_origin:
            model_info = {}
            model_info['model_path'] = f"{self.model_name}/GeminiMol.pt"
            test_score = self.evaluate(val_df)
            model_info['score'] = np.mean(test_score["SPEARMANR"].to_list())
            for index, row in test_score.iterrows():
                for column in test_score.columns:
                    key = f"{index}_{column}"
                    value = row[column]
                    model_info[key] = value
            model_list.append(model_info)
        # training
        for _ in range(epochs):
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
                if batch_id % schedule_unit == 0:
                    val_res = self.evaluate(val_df)
                    val_score = np.mean(val_res["SPEARMANR"].to_list())
                    print(f'Epoch: {_+1}, batch id: {batch_id+1}, score: {val_score}')
                    if val_score > best_score+0.001:
                        best_score = val_score
                        best_results = val_res
                        best_parameters = self.state_dict()
                    # Apply cosine learning rate schedule
                    lr = learning_rate * ( 0.05 + (1 - 0.05) * np.cos(float(batch_id % T_max)/T_max)) 
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                if batch_id % T_max == 0:
                    model_info = {}
                    print(f'Save the checkpoint GeminiMol_R{prefix}: {best_score}')
                    torch.save(best_parameters, f"{output_path}/GeminiMol_R{prefix}_{batch_id}.pt")
                    model_info['model_path'] = f"{output_path}/GeminiMol_R{prefix}_{batch_id}.pt"
                    model_info['score'] = best_score
                    for index, row in best_results.iterrows():
                        for column in best_results.columns:
                            key = f"{index}_{column}"
                            value = row[column]
                            model_info[key] = value
                    best_score = -1.0
                    best_parameters = None
                    model_list.append(model_info)
        model_info = {}
        print(f'Save the checkpoint GeminiMol_R{prefix}:')
        torch.save(best_parameters, f"{output_path}/GeminiMol_R{prefix}_{batch_id}.pt")
        model_info['model_path'] = f"{output_path}/GeminiMol_R{prefix}_{batch_id}.pt"
        model_info['score'] = best_score
        for index, row in best_results.iterrows():
            for column in best_results.columns:
                key = f"{index}_{column}"
                value = row[column]
                model_info[key] = value
        model_list.append(model_info)
        return model_list

class GeminiMol(BinarySimilarity):
    def __init__(self, model_name, batch_size=None):
        self.model_name = model_name
        with open(f"{model_name}/model_params.json", 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        if not batch_size is None:
            self.params['batch_size'] = batch_size
        super().__init__(model_name=model_name, **self.params)
        self.eval()
        if os.path.exists(f'{self.model_name}/feat_stat.csv'):
            self.noise_stat=pd.read_csv(f"{model_name}/feat_stat.csv")
        else:
            self.noise_stat=None
        self.similarity_metrics_list = self.label_list + ['RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson'] 
        if os.path.exists(f'{self.model_name}/predictors'):
            self.predictors = {}
            for predictor_file in os.listdir(f'{self.model_name}/predictors'):
                if predictor_file.split('.')[-1] != 'pt':
                    continue
                predictor_name = predictor_file.split('.')[0]
                self.predictors[predictor_name] = encoding2score(n_embed=self.params['encoding_features'], dropout_rate=0.15, expand_ratio=3, activation='LeakyReLU')
                self.predictors[predictor_name].load_state_dict(torch.load(f'{self.model_name}/predictors/{predictor_file}'))
            self.recommanded_metric = {
                'binary' : 'AUROC',
                'regression' : 'SPEARMANR', 
            }
            self.all_metric = {
                'binary' : ['ACC', 'AUROC', 'AUPRC', 'recall', 'precision', 'f1'],
                'regression' : ['RMSE', 'MAE', 'MSE', 'PEARSONR', 'SPEARMANR'],
            }
            self.metric_functions = {
                'AUROC': lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
                'AUPRC': lambda y_true, y_pred: average_precision_score(y_true, y_pred),
                'MSE': lambda y_true, y_pred: -1*mean_squared_error(y_true, y_pred),
                'MAE': lambda y_true, y_pred: -1*mean_absolute_error(y_true, y_pred),
                'RMSE': lambda y_true, y_pred: -1*np.sqrt(mean_squared_error(y_true, y_pred)),
                'SPEARMANR': lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
                'PEARSONR': lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
                'ACC': lambda y_true, y_pred: accuracy_score(y_true, [round(num) for num in y_pred]),
                'precision': lambda y_true, y_pred: precision_score(y_true, [round(num) for num in y_pred]), 
                'recall': lambda y_true, y_pred: recall_score(y_true, [round(num) for num in y_pred]), 
                'f1': lambda y_true, y_pred: f1_score(y_true, [round(num) for num in y_pred]), 
            }

    def QSAR_predictor(self, 
        df, 
        predictor_name, 
        smiles_name = 'smiles',
        as_pandas = True
        ):
        self.predictors[predictor_name].eval()
        with torch.no_grad():
            pred_values = []
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                smiles = rows[smiles_name].to_list()
                features = self.encode(smiles)
                pred = self.predictors[predictor_name](features).cuda()
                pred_values += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame({predictor_name: pred_values}, columns=predictor_name)
                return res_df
            else:
                return {predictor_name: pred_values}

    def evaluate_predictor(self, 
        df, 
        predictor_name, 
        smiles_name = 'smiles', 
        label_name = 'label',
        metrics = ['SPEARMANR']
        ):
        self.predictors[predictor_name].eval()
        prediction = self.QSAR_predictor(df, predictor_name, smiles_name=smiles_name)
        y_ture = df[label_name].to_list()
        y_pred = prediction[predictor_name].to_list()
        results = []
        for metric in metrics:
            results.append(self.metric_functions[metric](y_ture, y_pred))
        return results

    def fit_predictor(self, 
        predictor_name, 
        train_df, 
        val_df, 
        smiles_name = 'smiles',
        label_name = 'label',
        task_type = 'binary',
        epochs = 20,
        batch_size = 32,
        learning_rate = 1.0e-4, 
        optim_type = 'AdamW',
        ):
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
        if not os.path.exists(f'{self.model_name}/predictors'):
            os.mkdir(f'{self.model_name}/predictors')
            self.predictors[predictor_name] = encoding2score(n_embed=self.params['encoding_features'], dropout_rate=0.15, expand_ratio=3, activation='LeakyReLU')
        else:
            if predictor_name not in list(self.predictors.keys()):
                self.predictors[predictor_name] = encoding2score(n_embed=self.params['encoding_features'], dropout_rate=0.15, expand_ratio=3, activation='LeakyReLU')
         # Set up the optimizer
        optimizer = models[optim_type](self.predictors[predictor_name].parameters(), lr=learning_rate)
        batch_id = 0
        best_score = 0.0
        assert np.min(train_df[label_name].to_list()) >= 0 and np.max(train_df[label_name].to_list()) <= 1, f"Error: \
            Please start by scaling the labels to between 0 and 1. (now, max: {np.max(train_df[label_name].to_list())}, min: {np.min(train_df[label_name].to_list())})\
            For concentration-dependent activity values, such as IC50, a log function should be considered; \
            for floating-point labels within a specific range, polar-variance normalization should be considered; \
            for dichotomous problems, labels with better drug-forming properties should be set to 1, and those with poorer properties to 0; \
            for multi-classification problems, the labels should be set to in-between 0-1."
        if task_type == 'binary':
            loss_function = nn.CrossEntropyLoss(reduction='mean')
        elif task_type == 'regression':
            loss_function = nn.MSELoss()
        for _ in range(epochs):
            self.predictors[predictor_name].train()
            for i in range(0, len(train_df), batch_size):
                batch_id += 1
                rows = train_df.iloc[i:i+batch_size]
                smiles = rows[smiles_name].to_list()
                labels = rows[label_name].to_list()
                features = self.encode(smiles)
                pred = self.predictors[predictor_name](features).cuda()
                label_tensor = torch.tensor(labels, dtype=torch.float32).cuda()
                loss = loss_function(pred, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id % 50 == 0:
                    print(f'Epoch: {_+1}, batch id: {batch_id+1}, loss: {loss.item()}')
                if batch_id % 1000 == 0:
                    val_res = self.evaluate_predictor(
                        val_df, 
                        predictor_name, 
                        smiles_name = smiles_name,
                        metrics = [self.recommanded_metric[task_type]],
                        )
                    print(f"Epoch {_+1}, evaluate {self.recommanded_metric[task_type]} on the validation set: {val_res[self.recommanded_metric[task_type]]}")
                    if best_score < val_res[self.recommanded_metric[task_type]]:
                        best_score = val_res[self.recommanded_metric[task_type]]
                        torch.save(self.predictors[predictor_name].state_dict(), f'{self.model_name}/predictors/{predictor_name}.pt')
            val_res = self.evaluate_predictor(
                val_df, 
                predictor_name, 
                smiles_name = smiles_name,
                metrics = [self.recommanded_metric[task_type]],
                )
            print(f"Epoch {_+1}, evaluate {self.recommanded_metric[task_type]} on the validation set: {val_res[self.recommanded_metric[task_type]]}")
            if best_score < val_res[self.recommanded_metric[task_type]]:
                best_score = val_res[self.recommanded_metric[task_type]]
                torch.save(self.predictors[predictor_name].state_dict(), f'{self.model_name}/predictors/{predictor_name}.pt')
        self.predictors[predictor_name].load_state_dict(torch.load(f'{self.model_name}/predictors/{predictor_name}.pt'))

    def create_database(self, query_smiles_table, smiles_column='smiles'):
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
        return pd.DataFrame(shape_features).add_prefix('GM_')


'''

MolDecoder

This is a RetNet framework that accepts inputs of coded vectors 
(generated by molecular encoders) and transforms the coded vectors 
into compound SMILES. It gives the GeminiMol model the ability 
to generate novel molecules. Note that the implementation of the 
RetNet framework and tokenizer used here are not exactly the same 
as in the DeepShape molecular encoder.

'''

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )
        
        self.cuda()

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:].cuda()
            sin = sin[-length:].cuda()
            cos = cos[-length:].cuda()
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:].cuda()
            sin = sin[-length:].cuda()
            cos = cos[-length:].cuda()
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x

class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        
        self.xpos = XPOS(head_size)
        
        self.cuda()

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).cuda()

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return ret @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = self.gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size).cuda()

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V
        
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        
        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)
        
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        
        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D

class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim) for gamma in self.gammas
        ])
        self.cuda()

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """
    
        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)
        
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.ReLU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.cuda()
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

class MolDecoder(nn.Module):
    def __init__(self, 
            encoding_size, 
            token_dict_size,
            decoder_features_size=256, 
            decoder_head=16, 
            decoder_layers=8,
            decoder_type='RetNet',
            layers={
                'css_concentrate': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0},
                'mapping': {'type': 'MLP', 'layers': 3, 'activation': 'GELU', 'dropout': 0.0},
                'prompt': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0}
            },
            mapping_first=False, # False: RetNet -> mapping; True: mapping -> RetNet
            retnet_layer=True # True: use RetNet; False: not use RetNet
        ):
        super(MolDecoder, self).__init__()
        self.token_dict_size = token_dict_size
        self.features_size = decoder_features_size
        self.layers = decoder_layers
        self.heads = decoder_head
        self.decoder_type = decoder_type
        # initialize the css_concentrate module
        self.css_concentrate = GeminiMLP(
            encoding_size, 
            self.features_size, 
            mlp_type=layers['css_concentrate']['type'], 
            num_layers=layers['css_concentrate']['layers'], 
            activation=layers['css_concentrate']['activation'],
            dropout=layers['mapping']['dropout']
            )
        self.css_concentrate.apply(initialize_weights)
        # initialize the attention pooling module and prompt module
        if layers['prompt']['type'] == 'None':
            self.prompt_layer = False
        else:
            self.prompt_layer = True
            self.attention_pooling = SelfAttentionPooling(attention_size=32)
            self.prompt = GeminiMLP(
                self.features_size+encoding_size,
                self.features_size, 
                mlp_type=layers['prompt']['type'], 
                num_layers=layers['prompt']['layers'], 
                activation=layers['prompt']['activation'],
                norm='LayerNorm',
                dropout=layers['mapping']['dropout']
                )
            self.prompt.apply(initialize_weights)
        # initialize the mapping function 
        self.mapping = GeminiMLP(
            self.features_size, 
            self.token_dict_size, 
            mlp_type=layers['mapping']['type'], 
            num_layers=layers['mapping']['layers'], 
            activation=layers['mapping']['activation'],
            norm='LayerNorm',
            dropout=layers['mapping']['dropout']
            )
        self.mapping.apply(initialize_weights)
        if mapping_first == True:
            self.mapping_first = True
            self.retnet_feature_size = self.token_dict_size
        else:
            self.mapping_first = False
            self.retnet_feature_size = self.features_size
        # initialize the decoder
        if retnet_layer == True:
            self.retnet_layer = True
            self.decoder = RetNet(self.layers, self.retnet_feature_size, self.retnet_feature_size*2, decoder_head)
            self.decoder.apply(initialize_weights)
        else:
            self.retnet_layer = False
        self.cuda()

    def forward_parallel(self, gemini_encoding, smiles_len=128):
        batch_size = gemini_encoding.shape[0]
        # Concentrate the css encoding
        css_indicator = self.css_concentrate(gemini_encoding) # css_indicator: (batch_size, features_size)
        memory = torch.zeros((batch_size, smiles_len, self.features_size)).cuda()
        memory[:, 0, :] = css_indicator
        # Prompt
        if self.prompt_layer == True:
            for i in range(smiles_len - 1):
                memory_indicator = self.attention_pooling(memory[:, :i, :].clone())
                memory[:, i+1, :] = self.prompt(torch.cat((gemini_encoding, memory_indicator), dim=1))
        else:
            for i in range(smiles_len - 1):
                memory[:, i+1, :] = memory[:, 0, :].clone()
        # RetNet and mapping
        if self.mapping_first == True:
            memory = self.mapping(memory)
        if self.retnet_layer == True:
            memory = self.decoder(memory)
        if self.mapping_first == False:
            memory = self.mapping(memory)
        return memory
    
    def forward_recurrent(self, gemini_encoding, states, smiles_len=128):
        batch_size = gemini_encoding.shape[0]
        # Concentrate the css encoding
        css_indicator = self.css_concentrate(gemini_encoding)
        memory = torch.zeros((batch_size, smiles_len, self.features_size)).cuda()
        memory[:, 0, :] = css_indicator
        # Prompt
        if self.prompt_layer == True:
            for i in range(smiles_len - 1):
                memory_indicator = self.attention_pooling(memory[:, :i, :].clone())
                memory[:, i+1, :] = self.prompt(torch.cat((gemini_encoding, memory_indicator), dim=1))
        else:
            for i in range(smiles_len - 1):
                memory[:, i+1, :] = memory[:, 0, :].clone()
        # RetNet and mapping
        if self.mapping_first == True:
            memory = self.mapping(memory)
        states = [
            [
                torch.zeros(self.features_size // self.heads, self.decoder.v_dim // self.heads).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
                for _ in range(self.heads)
            ]
            for _ in range(self.layers)
        ]
        if self.retnet_layer == True:
            for i in range(smiles_len - 1):
                memory[:, i+1, :], states = self.decoder.forward_recurrent(memory[:, i:i+1, :].clone(), states, i)
        if self.mapping_first == False:
            memory = self.mapping(memory)
        return memory

    def forward(self, gemini_encoding, smiles_len=128, decoder_mode='parallel'):
        if decoder_mode == 'parallel':
            return self.forward_parallel(gemini_encoding, smiles_len)
        elif decoder_mode == 'recurrent': # under development
            return self.forward_recurrent(gemini_encoding, smiles_len)

class GeminiMolDecoder(nn.Module):
    def __init__(self, 
        model_name, 
        vocab_dict=None, 
        decoder_embedding_size=256, 
        decoder_head=16, 
        decoder_layers=8, 
        batch_size=None,
        mlp_layers={
                'css_concentrate': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0},
                'mapping': {'type': 'MLP', 'layers': 3, 'activation': 'GELU', 'dropout': 0.0},
                'prompt': {'type': 'MLP', 'layers': 0, 'activation': 'GELU', 'dropout': 0.0}
            }
        ):
        super(GeminiMolDecoder, self).__init__()
        self.model_name = model_name
        self.geminimol = GeminiMol(model_name)
        self.geminimol.eval()
        if os.path.exists(f'{self.model_name}/MolDecoder.pt'):
            with open(f"{model_name}/tokenizer.json", 'r', encoding='utf-8') as f:
                self.vocab_dict = json.load(f) # {token: index}
            self.token_dict = {v: k for k, v in self.vocab_dict.items()} # {index: token}
            with open(f"{model_name}/mol_decoder_params.json", 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.smiles_decoder = nn.DataParallel(
                MolDecoder(
                    encoding_size=self.geminimol.params['encoding_features'], 
                    token_dict_size=len(self.vocab_dict), 
                    decoder_features_size=params['decoder_embedding_size'], 
                    decoder_head=params['decoder_head'],
                    decoder_layers=params['decoder_layers'],
                    layers=params['mlp_layers']
                ))
            self.smiles_decoder.load_state_dict(torch.load(f'{self.model_name}/MolDecoder.pt'))
            self.batch_size = params['batch_size'] 
            self.geminimol.batch_size = batch_size
        else:
            self.vocab_dict = vocab_dict # {token: index}
            self.token_dict = {v: k for k, v in vocab_dict.items()} # {index: token}
            self.smiles_decoder = nn.DataParallel(
                MolDecoder(
                    encoding_size=self.geminimol.params['encoding_features'], 
                    token_dict_size=len(self.vocab_dict), 
                    decoder_features_size=decoder_embedding_size, 
                    decoder_head=decoder_head,
                    decoder_layers=decoder_layers,
                    layers=mlp_layers
                ))      
            if batch_size == None:
                self.batch_size = self.geminimol.params['batch_size']
            else:
                self.batch_size = batch_size
                self.geminimol.batch_size = batch_size

    def token_embedding(self, input_string, max_seq_len):
        # Convert input sentence to a tensor of numerical indices
        indices = [] # [CLS] token
        i = 0
        while i < len(input_string):
            # Find longest matching word in vocabulary
            best_match = None
            for word, index in self.vocab_dict.items():
                if input_string.startswith(word, i):
                    if not best_match or len(word) > len(best_match[0]):
                        best_match = (word, index)
            if best_match:
                indices.append(best_match[1])
                i += len(best_match[0])
            else:
                indices.append(self.vocab_dict['[UNK]']) # No matching word found, use character index ([UNK])
                i += 1
            if len(indices) == max_seq_len:
                break
        pad_len = max_seq_len - len(indices)
        indices += [self.vocab_dict['<EOS>']] # <EOS> token
        indices += [self.vocab_dict['<PAD>']] * pad_len # <PAD>
        # Reshape indices batch to a rectangular shape, with shape (batch_size, seq_len)
        return indices

    def generate_weights(self, indices_list):
        counter = Counter()
        for sentence in indices_list:
            counter.update(sentence)
        weights = {}
        total_tokens = sum(counter.values())
        for token, index in self.vocab_dict.items():
            token_count = counter.get(index, 0)
            token_weight = total_tokens / token_count if token_count > 0 else 1.0
            weights[index] = token_weight
        for token in weights:
            if weights[token] > len(weights):
                weights[token] = len(weights)
        total_weight = sum(weights.values()) / (len(weights) * 10)
        for token in weights:
            weights[token] /= total_weight
        return weights

    def smiles2tensor(self, input_sents, return_max_seq_len=False):
        # Get the max sequence length in current batch
        max_seq_len = max([len(s) for s in input_sents])
        # Create input tensor of shape (batch_size, seq_len)
        indices = [self.token_embedding(s, max_seq_len) for s in input_sents] 
        weights = self.generate_weights(indices)
        indices = [torch.tensor(indice).unsqueeze(0).cuda() for indice in indices] 
        input_tensor = torch.cat(indices, dim=0) 
        if return_max_seq_len:
            return input_tensor, weights, max_seq_len+1
        else:
            return input_tensor, weights

    def tensor2smiles(self, tensor):
        # tensor: (batch_size, seq_len)
        # vocab: a dictionary that maps token ids to tokens
        # Convert the tensor to a numpy array
        tensor = tensor.detach().cpu().numpy()
        # Iterate over each sequence in the batch
        texts = []
        for sequence in tensor:
            # Convert each token id in the sequence to its corresponding token
            text = []
            for token_id in sequence:
                if token_id not in [self.vocab_dict['[UNK]'], self.vocab_dict['<PAD>'], self.vocab_dict['<EOS>']]:
                    text.append(self.token_dict[token_id]) # Remove padding tokens
                elif token_id == self.vocab_dict['<EOS>']:
                    break # Stop decoding when the end of sequence token is encountered
            # Concatenate the tokens into a single string and add it to the list of texts
            texts.append(''.join(text))
        return texts # [batch_size]

    def forward(self, gemini_encoding, max_seq_len):
        # Decode the encoded features to smiles
        decoded_tensor = self.smiles_decoder(gemini_encoding, smiles_len=max_seq_len) # (batch_size, seq_len, token_dict_size)
        return decoded_tensor

    def decoder(self, gemini_encoding, smiles_len=128):
        # gemini_encoding: (batch_size, features_size)
        self.smiles_decoder.eval()
        with torch.no_grad():
            decoded_tensor = self.smiles_decoder(gemini_encoding, smiles_len=smiles_len) # (batch_size, seq_len, token_dict_size)
            # Find the index of the maximum value along the last dimension
            decoded_text = self.tensor2smiles(torch.argmax(decoded_tensor, dim=-1)) 
        return decoded_text
    
    def check_unvalid_smiles(self, smiles):
        for unvalid_frag in [
            'BrBr', 'ClCl', 'FF', 'II', 'ClBr', 'BrCl', 'PP',
            'CCCCCCCCCCCCCCC', 'NNNN', 'OOO', 'ssss', '((((', 'SSS', 
            'ccccccccccccccc', 'nnnnnn', 'oo', 
            '[NH+][NH+]', '[NH2+][NH2+]', '[N@H+][N@H+]', '[N@@H+][N@@H+]', 
            '[CH][CH]', '[C@H][C@H]', '[O-][O-]', '[NH3+][NH3+]', '==', '##'
            ]:
            if unvalid_frag in smiles:
                return False
        return True

    def predict(self, smiles_list, similarity_metrics='Cosine', batch_size=None):
        self.smiles_decoder.eval()
        results = {
            'input_smiles': [], 
            'decoded_smiles': [], 
            'css_similarity': [],
            '2d_similarity': [],
            'validity': []
            }
        if batch_size == None:
            batch_size = self.batch_size
        with torch.no_grad():
            decoded_smiles_list = []
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                max_seq_len = max([len(s) for s in batch])
                # Encode all sentences using the gemini encoder
                gemini_encoding = self.geminimol.encode(batch)
                # Decode the encoded features to smiles
                decoded_text = self.decoder(gemini_encoding, smiles_len=max_seq_len) # (batch_size, seq_len, token_dict_size)
                decoded_smiles_list += decoded_text
            for decoded_smiles, input_smiles in zip(decoded_smiles_list, smiles_list):
                results['input_smiles'] += [input_smiles]
                results['decoded_smiles'] += [decoded_smiles]
                mol = None
                try:
                    if self.check_unvalid_smiles(decoded_smiles):
                        mol = Chem.MolFromSmiles(decoded_smiles)
                        if mol is None:
                            results['css_similarity'] += [0]
                            results['2d_similarity'] += [0]
                            results['validity'] += [0]
                        else:
                            ref_mol = Chem.MolFromSmiles(input_smiles)
                            res = FMCS.FindMCS([mol, ref_mol], ringMatchesRingOnly=True, atomCompare=(FMCS.AtomCompare.CompareElements))
                            Chem.SanitizeMol(mol)
                            decoded_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=False, isomericSmiles=True)
                            encoded_features = self.geminimol.encode([decoded_smiles, input_smiles])
                            pred_css = self.geminimol.decode(encoded_features, label_name=similarity_metrics, batch_size=1)
                            results['css_similarity'] += list(pred_css.cpu().detach().numpy()) # 
                            results['2d_similarity'] += [ res.numBonds / len(ref_mol.GetBonds()) ]
                            results['validity'] += [1]
                    else:
                        results['css_similarity'] += [0]
                        results['2d_similarity'] += [0]
                        results['validity'] += [0]
                except:
                    results['css_similarity'] += [0]
                    results['2d_similarity'] += [0]
                    results['validity'] += [0]
        return pd.DataFrame(results)

    def evaluate(self, smiles_list, similarity_metrics='Cosine', print_table=False):
        smiles_list = list(set(smiles_list))
        prediction = self.predict(smiles_list, similarity_metrics=similarity_metrics, batch_size=self.batch_size)
        recovery = sum(1 for input_smiles, output_smiles in zip(prediction['input_smiles'].to_list(), prediction['decoded_smiles'].to_list()) if input_smiles == output_smiles) / len(smiles_list)
        if print_table:
            print(f"The valid predicted results as follows:")
            print(prediction[prediction['validity']==1][['input_smiles', 'css_similarity', 'decoded_smiles', '2d_similarity']])
        return recovery, np.mean(prediction['validity'].to_list()), np.mean(prediction['css_similarity'].to_list()), np.mean(prediction['2d_similarity'].to_list())

    def fit(self, 
        train_smiles_list, 
        epochs, 
        similarity_metrics, 
        learning_rate=1.0e-5, 
        val_smiles_list=None, 
        optim_type='AdamW', 
        cross_entropy=False,
        valid_award=0.4,
        weight_dict=True,
        fine_tune=False,
        flooding=None,
        temperature=0.05,
        batch_group=5,
        mini_epoch=20
        ):
        # Load the models and optimizer
        optim_types = {
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'NAdam': torch.optim.NAdam,
            'RMSprop': torch.optim.RMSprop,
            'Adamax': torch.optim.Adamax,
            'Rprop': torch.optim.Rprop
        }
        if optim_type not in optim_types:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # set up pandas printing
        pd.set_option('display.max_columns', None)
        # Set up the optimizer
        if optim_type in ['AdamW', 'Adamax', 'NAdam']:
            optimizer = optim_types[optim_type](self.smiles_decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=0.1)
        elif optim_type in ['SGD']:
            optimizer = optim_types[optim_type](self.smiles_decoder.parameters(), lr=learning_rate, momentum=0.3, weight_decay=0.1)
        else:
            optimizer = optim_types[optim_type](self.smiles_decoder.parameters(), lr=learning_rate, weight_decay=0.1)
        # triaining
        if fine_tune:
            best_recovery, best_validity, best_similarity_3D, best_similarity_2D = self.evaluate(val_smiles_list, similarity_metrics=similarity_metrics)
        else:
            best_recovery = 0.0
            best_similarity_3D = 0.0
            best_similarity_2D = 0.0
            best_validity = 0.0
        best_model_score = best_similarity_3D + best_similarity_2D
        batch_id = 0
        loss_dict = {
            'log': lambda loss: torch.mean(torch.log(loss+0.01)/-2.0, dim = 0), # log
            'x': lambda loss: ( 1 - torch.mean(loss, dim = 0)), # x
            'sigmoid': lambda loss: torch.mean(( 1 / (1 + 10000 ** (loss - 0.5)))), # sigmoid
            'x3': lambda loss: torch.mean( -1 * ( loss - 1) ** 3, dim = 0), # x3
            '1-x3': lambda loss: torch.mean( 1 - ( loss) ** 3, dim = 0), # x3
        }
        if cross_entropy:
            if weight_dict:
                CrossEntropy = nn.CrossEntropyLoss(reduction='none')
            else:
                CrossEntropy = nn.CrossEntropyLoss(reduction='mean')
        std_tensor = torch.tensor(self.geminimol.noise_stat['Std'].values, dtype=torch.float).unsqueeze(0).cuda()
        for epoch in range(epochs):
            self.smiles_decoder.train()
            start = time.time()
            train_smiles_list = shuffle(train_smiles_list)
            for i in range(0, len(train_smiles_list)//self.batch_size):
                batch = random.sample(train_smiles_list, self.batch_size) # train_smiles_list[i:i+self.batch_size]
                batch_id += 1
                # input_sents: a list of smiles
                # Convert input sentences to a tensor of numerical indices
                input_tensor, weights_dict, max_seq_len = self.smiles2tensor(batch, return_max_seq_len=True) # input_tensor: (batch_size, seq_len)
                # Encode all sentences using the transformer
                with torch.no_grad():
                    gemini_encoding = self.geminimol.encode(batch)
                # decoded_tensor: (batch_size, seq_len, token_dict_size)
                if temperature > 0:
                    batch_size, features_size = gemini_encoding.shape
                    gemini_encoding = gemini_encoding + torch.mul((torch.randn(batch_size, features_size).cuda() - 0.5) * 2, std_tensor * temperature)
                decoded_tensor = self.forward(gemini_encoding, max_seq_len) 
                decoded_text = self.tensor2smiles(torch.argmax(decoded_tensor, dim=-1)) 
                # calculate the validity and similarity
                similarities = []
                validities = []
                for input_smiles, decoded_smiles in zip(batch, decoded_text):
                    try:
                        if self.check_unvalid_smiles(decoded_smiles):
                            mol = Chem.MolFromSmiles(decoded_smiles)
                            ref_mol = Chem.MolFromSmiles(input_smiles)
                            res = FMCS.FindMCS([mol, ref_mol], ringMatchesRingOnly=True, atomCompare=(FMCS.AtomCompare.CompareElements))
                            similarity_2D = res.numBonds / len(ref_mol.GetBonds())
                            similarities.append(similarity_2D + valid_award + 0.001)
                            validities.append(1.0)
                        else:
                            similarities.append(0.001)
                            validities.append(0.001)
                    except:
                        similarities.append(0.001)
                        validities.append(0.001)
                # calculate the similarity loss
                similarity_loss = torch.tensor(similarities, dtype=torch.float32, requires_grad=True).cuda()
                validity_loss = torch.tensor(validities, dtype=torch.float32, requires_grad=True).cuda()
                if cross_entropy:
                    cross_entropy_loss = CrossEntropy(decoded_tensor.transpose(1, 2), input_tensor)
                    if weight_dict:
                        weights = []
                        for batch in range(input_tensor.size(0)):
                            batch_weights = []
                            for seq in range(input_tensor.size(1)):
                                idx = input_tensor[batch, seq].item()
                                weight = weights_dict[idx]
                                batch_weights.append(weight)
                            weights.append(batch_weights)
                        weights = torch.Tensor(weights).cuda()
                        cross_entropy_loss = torch.mean(cross_entropy_loss * weights)
                    loss = (loss_dict['x3'](validity_loss) + loss_dict['sigmoid'](similarity_loss)) * cross_entropy_loss 
                else:
                    loss = loss_dict['x3'](validity_loss) + loss_dict['sigmoid'](similarity_loss)
                ## with flooding 
                if flooding is not None:
                    loss = (loss - flooding).abs() + flooding
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.smiles_decoder.parameters(), 1.0) # gradient clipping, avoid gradient explosion
                optimizer.step()
                # print the loss and validation results
                if batch_id % batch_group == 0:
                    if cross_entropy:
                        print(f"Epoch {epoch+1}, batch {batch_id}, time {round(time.time()-start, 2)}, train loss: {round(loss.item(), 3)} (CE: {round(cross_entropy_loss.item(), 2)}, Sim: {round(torch.mean(similarity_loss).item(), 3)})") 
                    else:
                        print(f"Epoch {epoch+1}, batch {batch_id}, time {round(time.time()-start, 2)}, train loss: {round(loss.item(), 3)} (Sim: {round(torch.mean(similarity_loss).item(), 3)})")
                if batch_id % mini_epoch == 0 and val_smiles_list is not None:
                    val_recovery, val_validity, val_similarity_3D, val_similarity_2D = self.evaluate(val_smiles_list, similarity_metrics=similarity_metrics, print_table=True)
                    self.smiles_decoder.train()
                    model_score = val_similarity_3D + val_similarity_2D # x = 0.36, y = 1
                    print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                    print(f'Validity    {round(val_validity, 3)}')
                    print(f'Recovery    {round(val_recovery, 3)}')
                    print(f'3D Similarity    {round(val_similarity_3D, 3)} ({round(val_similarity_3D/(val_validity+1.0e-12), 3)})')
                    print(f'2D Similarity    {round(val_similarity_2D, 3)} ({round(val_similarity_2D/(val_validity+1.0e-12), 3)})')
                    print(f'Model score    {round(model_score, 3)}')
                    if model_score > best_model_score:
                        best_model_score = model_score
                        best_recovery = val_recovery
                        best_validity = val_validity
                        best_similarity_3D = val_similarity_3D
                        best_similarity_2D = val_similarity_2D
                        torch.save(self.smiles_decoder.state_dict(), f"{self.model_name}/MolDecoder.pt")
            if val_smiles_list is not None:
                val_recovery, val_validity, val_similarity_3D, val_similarity_2D = self.evaluate(val_smiles_list, similarity_metrics=similarity_metrics, print_table=True)
                self.smiles_decoder.train()
                model_score = val_similarity_3D + val_similarity_2D # x = 0.36, y = 1
                print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                print(f'Validity    {round(val_validity, 3)}')
                print(f'Recovery    {round(val_recovery, 3)}')
                print(f'3D Similarity    {round(val_similarity_3D, 3)} ({round(val_similarity_3D/(val_validity+1.0e-12), 3)})')
                print(f'2D Similarity    {round(val_similarity_2D, 3)} ({round(val_similarity_2D/(val_validity+1.0e-12), 3)})')
                print(f'Model score    {round(model_score, 3)}')
                if model_score > best_model_score:
                    best_model_score = model_score
                    best_recovery = val_recovery
                    best_validity = val_validity
                    best_similarity_3D = val_similarity_3D
                    best_similarity_2D = val_similarity_2D
                    torch.save(self.smiles_decoder.state_dict(), f"{self.model_name}/MolDecoder.pt")
                print(f"Epoch {epoch+1}, Best val validity: {round(best_validity, 3)}")
                print(f"Epoch {epoch+1}, Best val recovery: {round(best_recovery, 3)}")
                print(f"Epoch {epoch+1}, Best val 3D similarity: {round(best_similarity_3D, 3)} ({round(best_similarity_3D/(best_validity+1.0e-12), 3)})")
                print(f"Epoch {epoch+1}, Best val 2D similarity: {round(best_similarity_2D, 3)} ({round(best_similarity_2D/(best_validity+1.0e-12), 3)})")
                print(f"Epoch {epoch+1}, Best val model score: {round(best_model_score, 3)}")
        self.smiles_decoder.load_state_dict(torch.load(f'{self.model_name}/MolDecoder.pt'))

    def scaffold_hopping_award(self, 
            ref_features, 
            sampling_tensor, 
            batch_size, 
            label_name='ShapeScore'
        ):
        features = torch.cat((sampling_tensor, ref_features), dim=0)
        pred_MCS = self.geminimol.decode(features, label_name='MCS', batch_size=batch_size)
        pred_CSS = self.geminimol.decode(features, label_name=label_name, batch_size=batch_size)
        loss = ( pred_MCS - pred_CSS ) * pred_CSS
        return loss

    def decode_molecules(
            self,
            ref_smiles,
            sampling_tensor
    ):
        self.eval()
        with torch.no_grad():
            output_list = []
            for smiles_len in [
                    len(ref_smiles), 
                    len(ref_smiles) + len(ref_smiles) // 8,
                    128
                ]:
                for i in self.decoder(sampling_tensor, smiles_len=smiles_len):
                    try:
                        if self.check_unvalid_smiles(i):
                            mol = Chem.MolFromSmiles(i)
                            Chem.SanitizeMol(mol)
                            output_list += [Chem.MolToSmiles(mol, kekuleSmiles=False, doRandom=False, isomericSmiles=True)]
                    except:
                        pass
        return output_list

    def random_walking(self, replica_num=1000, smiles_len_list=128, num_steps_per_replica=10, temperature=0.1):
        self.eval()
        output_smiles = {}
        mean_tensor = torch.tensor(self.geminimol.noise_stat['Mean'].values, dtype=torch.float).unsqueeze(0).cuda()
        std_tensor = torch.tensor(self.geminimol.noise_stat['Std'].values, dtype=torch.float).unsqueeze(0).cuda()
        for smiles_len in smiles_len_list:
            output_smiles[smiles_len] = []
        with torch.no_grad():
            for _ in range(replica_num):
                sampling_tensor = ( torch.randn(self.geminimol.params['batch_size'], self.geminimol.params['encoding_features']).cuda() - 0.5 ) * 2
                for step in range(num_steps_per_replica):
                    if self.geminimol.noise_stat is not None:
                        sampling_tensor = torch.mul(sampling_tensor, std_tensor * temperature) + mean_tensor
                    for smiles_len in smiles_len_list:
                        output_smiles[smiles_len] += self.decoder(sampling_tensor, smiles_len=smiles_len)
        return output_smiles

    def MCMC(self, ref_smiles, task_function, replica_num=10, num_steps_per_replica=3, num_seeds_per_steps=10, temperature=0.5, iterative_mode='continous', init_seeds=30):
        self.eval()
        std_tensor = torch.tensor(self.geminimol.noise_stat['Std'].values, dtype=torch.float).unsqueeze(0).cuda()
        with torch.no_grad():
            ref_features = self.geminimol.encode([ref_smiles]*self.geminimol.batch_size)
            batch_size, features_size = ref_features.shape
            output_smiles_list = []
            for replica in range(replica_num):
                replica_smiles_list = []
                for step in range(init_seeds):
                    sampling_tensor = ref_features + torch.mul((torch.randn(batch_size, features_size).cuda() - 0.5) * 2, std_tensor * (temperature ** step))
                    loss = task_function(ref_features, sampling_tensor, batch_size, label_name='Cosine')
                    replica_smiles_list += self.decode_molecules(ref_smiles, sampling_tensor)
                    print(f'replica {replica+1}, warm step {step+1}, loss {torch.mean(loss)}')
                for step in range(num_steps_per_replica):
                    sampling_tensor = sampling_tensor + torch.mul((torch.randn(batch_size, features_size).cuda() - 0.5) * 2, std_tensor * (temperature ** step))
                    loss = task_function(ref_features, sampling_tensor, batch_size, label_name='Cosine')
                    replica_smiles_list += self.decode_molecules(ref_smiles, sampling_tensor)
                    print(f'replica {replica+1}, step {step+1}, loss {torch.mean(loss)}')
                    if iterative_mode == 'random':
                        seeds_pool = [ref_smiles] * batch_size + replica_smiles_list * 9 * batch_size 
                        seeds_list = random.sample(seeds_pool, batch_size)
                        sampling_tensor = self.geminimol.encode(seeds_list)
                    elif iterative_mode == 'continous':
                        _, topk_indices = torch.topk(loss, k=num_seeds_per_steps, largest=False)
                        selected_indices = topk_indices.repeat(batch_size // num_seeds_per_steps)
                        if (batch_size-(batch_size // num_seeds_per_steps)*num_seeds_per_steps) != 0:
                            selected_indices = torch.cat([selected_indices, topk_indices[:batch_size-(batch_size // num_seeds_per_steps)*num_seeds_per_steps]])
                        sampling_tensor = torch.index_select(sampling_tensor, dim=0, index=selected_indices)
                output_smiles_list += replica_smiles_list
        return output_smiles_list

    def directed_evolution(self, ref_smiles, task_function, optim_type='AdamW', replica_num=5, num_steps_per_replica=300, temperature=0.1):
        self.eval()
        std_tensor = torch.tensor(self.geminimol.noise_stat['Std'].values, dtype=torch.float).unsqueeze(0).cuda()
        # Load the models and optimizer
        optim_types = {
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
            'Adamax': torch.optim.Adamax,
            'Rprop': torch.optim.Rprop
        }
        if optim_type not in optim_types:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        ref_features = self.geminimol.encode([ref_smiles]*self.geminimol.batch_size)
        batch_size, features_size = ref_features.shape
        output_smiles_list = []
        for replica in range(replica_num):
            if replica == 0:
                sampling_tensor = torch.tensor(ref_features.clone().detach(), requires_grad=True)
            else:
                sampling_tensor = torch.tensor(ref_features.clone().detach() + torch.mul((torch.randn(batch_size, features_size).cuda() - 0.5) * 2, std_tensor * temperature), requires_grad=True)
            optimizer = optim_types[optim_type]([sampling_tensor], lr=1.0e-4)
            best_score = 0.01
            patience = 40
            for step in range(num_steps_per_replica):
                loss = torch.mean(task_function(ref_features, sampling_tensor, batch_size, label_name='Cosine'))
                print(f'replica {replica+1}, step {step+1}, loss {loss.item()}')
                if loss.item() < best_score-0.01:
                    best_score = loss.item()
                    patience = 30
                else:
                    patience += -1
                val_smiles_num = len(output_smiles_list)
                output_smiles_list += self.decode_molecules(ref_smiles, sampling_tensor)
                if len(output_smiles_list) <= val_smiles_num:
                    patience += -1
                if patience <= 0:
                    break
                loss.backward(retain_graph=True)
                optimizer.step()
        return output_smiles_list




