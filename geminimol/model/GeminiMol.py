# base
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from functools import partial
# for GraphEncoder (MolecularEncoder)
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.stats import pearsonr, spearmanr
from dgl import batch
from dgllife.utils import atom_type_one_hot, atom_formal_charge, atom_hybridization_one_hot, atom_chiral_tag_one_hot, atom_is_in_ring, atom_is_aromatic, ConcatFeaturizer, BaseAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
from dgllife.model.gnn.wln import WLN
from dgllife.model.gnn.gat import GAT
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.graphsage import GraphSAGE
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import torch.nn.init as init

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

class encoding2score(nn.Module):
    def __init__(self, 
            n_embed=1024, 
            dropout_rate=0.1, 
            expand_ratio=3,
            activation='GELU',
            initialize=False
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
        if expand_ratio == 0:
            self.decoder = nn.Sequential(
                    nn.Linear(n_embed*2, 256, bias=True), # 1
                    nn.BatchNorm1d(256), # 2
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(256, 1, bias=True),
                    nn.Identity()
                )
        else:
            self.decoder = nn.Sequential(
                    nn.Linear(n_embed*2, n_embed*expand_ratio*2), # 1
                    activation_dict[activation],
                    nn.BatchNorm1d(n_embed*expand_ratio*2), # 3
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(n_embed*expand_ratio*2, 1024), # 5
                    activation_dict[activation],
                    nn.Linear(1024, 128), # 7
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 10
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 13
                    activation_dict[activation],
                    nn.Dropout(p=dropout_rate, inplace=False),
                    nn.Linear(128, 128), # 16
                    activation_dict[activation],
                    nn.Linear(128, 1),
                    nn.Identity(),
                )
        if initialize:
            self.decoder.apply(initialize_weights)
        self.cuda()
        
    def forward(self, features):
        score = self.decoder(features)
        prediction = torch.sigmoid(score).squeeze(-1)
        return prediction

class SkipConnection(nn.Module):
    def __init__(self, feature_size, skip_layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.skip_layers = skip_layers
        if norm == 'LayerNorm':
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
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
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
        if norm == 'LayerNorm':
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
        else:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                )
                for _ in range(layers)
            ])
        self.cuda()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x

class RugbyMLP(nn.Module):
    def __init__(self, feature_size, layers=2, activation=nn.GELU(), norm=None, dropout=0.0):
        super().__init__()
        self.num_layers = layers
        if norm == 'LayerNorm':
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.LayerNorm(feature_size*self.num_layers),
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                    nn.LayerNorm(feature_size*self.num_layers),
                )
        elif norm == 'BatchNorm':
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.BatchNorm1d(feature_size*self.num_layers),
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                    nn.BatchNorm1d(feature_size*self.num_layers)
                )
        else:
            self.layers = nn.Sequential(
                    nn.Linear(feature_size, feature_size*self.num_layers),
                    activation,
                    nn.Dropout(p=dropout, inplace=False),
                    nn.Linear(feature_size*self.num_layers, feature_size*self.num_layers),
                    activation,
                )
        self.cuda()

    def forward(self, x):
        return self.layers(x)

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
            self.gemini_mlp = MLP(
                input_size, 
                layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(input_size, output_size)
        elif mlp_type == 'SkipConnection':
            self.gemini_mlp = SkipConnection(
                input_size, 
                skip_layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(2*input_size, output_size)
        elif mlp_type == 'RugbyMLP':
            self.gemini_mlp = RugbyMLP(
                input_size, 
                layers=num_layers, 
                activation=activation, 
                norm=norm, 
                dropout=dropout
            )
            self.out = nn.Linear(input_size*num_layers, output_size)
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

def params2layerlist(params, layer_num=6):
    params_list = []
    for _ in range(layer_num):
        params_list.append(params) 
    return params_list

class MolecularEncoder(nn.Module):
    '''
    MolecularEncoder

    This is a graph encoder model consisting of a graph neural network from 
    DGL and the MLP architecture in pytorch, which builds an understanding 
    of a compound's conformational space by supervised learning of CSS data.

    '''
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
        gnn_type = 'WLN'
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
        self.gnn_type = gnn_type
        if gnn_type == 'WLN':
            self.GeminiEncoder = WLN(
                atom_feat_size, 
                bond_feat_size, 
                n_layers=num_layers, 
                node_out_feats=num_features
            )
        elif gnn_type == 'GAT':
            self.GeminiEncoder = GAT(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                num_heads=params2layerlist(16, layer_num=num_layers), 
                feat_drops=params2layerlist(0, layer_num=num_layers), 
                activations=params2layerlist(nn.LeakyReLU(), layer_num=num_layers), 
                allow_zero_in_degree=True
            )
        elif gnn_type == 'GCN':
            self.GeminiEncoder = GCN(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                gnn_norm=params2layerlist('both', layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers), 
                allow_zero_in_degree=True
            )
        elif gnn_type == 'GraphSAGE':
            self.GeminiEncoder = GraphSAGE(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers)
            )
        elif gnn_type == 'GCNSAGE':
            self.GeminiEncoder = GraphSAGE(
                atom_feat_size, 
                hidden_feats=params2layerlist(num_features, layer_num=num_layers), 
                dropout=params2layerlist(0, layer_num=num_layers), 
                aggregator_type=params2layerlist('gcn', layer_num=num_layers)
            )
        elif gnn_type == 'AttentiveFP':
            self.GeminiEncoder = AttentiveFPGNN(
                atom_feat_size, 
                bond_feat_size, 
                num_layers=num_layers, 
                graph_feat_size=num_features, dropout=0.0
            )
        # init the readout and output layers
        self.readout_type = readout_type
        if readout_type == 'Weighted':
            assert num_features*2 == num_out_features, "Error: num_features*2 must equal num_out_features for Weighted readout."
            self.readout = WeightedSumAndMax(num_features)
        elif readout_type == 'MeanMLP':
            assert num_features == num_out_features, "Error: num_features must equal num_out_features for MLP readout."
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict[activation],
                mode='mean'
            )
        elif readout_type == 'ExtendMLP':
            self.readout = MLPNodeReadout(
                num_features, num_out_features, num_out_features,
                activation=activation_dict[activation],
                mode='mean'
            )
        elif readout_type == 'MMLP':
            self.readout = MLPNodeReadout(
                num_features, num_features, num_features,
                activation=activation_dict[activation],
                mode='mean'
            )
            self.output = GeminiMLP(
                num_features, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'AttentiveFP':
            assert num_features == num_out_features, "Error: num_features must equal num_out_features for AttentiveFP readout."
            self.readout = AttentiveFPReadout(
                    num_features, 
                    dropout=0.0
                    )
        elif readout_type == 'AttentiveMLP':
            self.readout = AttentiveFPReadout(
                    num_features, 
                    dropout=0.0
                    )
            self.output = GeminiMLP(
                num_features, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
        elif readout_type == 'WeightedMLP':
            self.readout = WeightedSumAndMax(num_features)
            self.output = GeminiMLP(
                num_features*2, 
                num_out_features, 
                mlp_type=integrate_layer_type, 
                num_layers=integrate_layer_num, 
                activation=activation,
                norm='BatchNorm'
            )
            self.output.cuda()
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
        elif readout_type == "MixedBN":
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
                activation=activation,
                norm='BatchNorm'
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
        if self.gnn_type in ['WLN', 'AttentiveFP']:
            encoding = self.GeminiEncoder(mol_graph, mol_graph.ndata['atom_type'], mol_graph.edata['bond_type'])
        elif self.gnn_type in ['GAT', 'GCN', 'GraphSAGE', 'GCNSAGE']:
            encoding = self.GeminiEncoder(mol_graph, mol_graph.ndata['atom_type'])
        if self.readout_type in ['MixedBN', 'Mixed']:
            mixed_readout = (self.readout['Weighted'](mol_graph, encoding), self.readout['MLP'](mol_graph, encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type in ['AttentiveMLP', 'WeightedMLP', 'MMLP']:
            return self.output(self.readout(mol_graph, encoding))
        elif self.readout_type == 'MixedMLP':
            mixed_readout = (self.readout['Max'](mol_graph, encoding), self.readout['Sum'](mol_graph, encoding), self.readout['Mean'](mol_graph, encoding))
            return self.output(torch.cat(mixed_readout, dim=1))
        elif self.readout_type == 'CombineMLP':
            return torch.cat([self.readout['Max'](mol_graph, encoding), self.readout['Sum'](mol_graph, encoding), self.readout['Mean'](mol_graph, encoding)], dim=1)
        elif self.readout_type == 'CombineWeighted':
            return torch.cat([self.readout['Weighted'](mol_graph, encoding), self.readout['MLP'](mol_graph, encoding)], dim=1)
        else:
            return self.readout(mol_graph, encoding)

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
        gnn_type = 'WLN',
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
        ## create MolecularEncoder
        self.Encoder = MolecularEncoder(
            atom_feat_size=self.atom_featurizer.feat_size(feat_name='atom_type'), 
            bond_feat_size=self.bond_featurizer.feat_size(feat_name='bond_type'), 
            gnn_type=gnn_type,
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

    def extract_hidden_decoder_features(self, features, label_name, num_layers=3):
        if label_name in self.label_list:
            binrary_features = torch.cat((features.clone(), features.clone()), dim=1)
        else:
            binrary_features = features.clone()
        for i in range(num_layers):
            layer = self.Decoder[label_name].decoder[i] 
            binrary_features = layer(binrary_features)
        return binrary_features

    def decode(self, features, label_name, batch_size, depth=0):
        # CSS similarity
        if label_name in self.label_list:
            encoded = torch.cat((features[:batch_size], features[batch_size:]), dim=1)
            return self.Decoder[label_name](encoded)
        # re-shape encoded features
        if depth == 0:
            encoded1 = features[:batch_size]
            encoded2 = features[batch_size:]
        else:
            encoded1 = torch.cat([self.extract_hidden_decoder_features(features[:batch_size], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
            encoded2 = torch.cat([self.extract_hidden_decoder_features(features[batch_size:], label_name, num_layers=depth) for label_name in self.label_list], dim=1)
        # decode vector similarity
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
        else:
            raise ValueError("Error: unkown label name: %s" % label_name)

    def extract(self, df, col_name, as_pandas=True, depth=0, labels=None):
        self.eval()
        data = df[col_name].tolist()
        if labels is None:
            labels = self.label_list
        with torch.no_grad():
            features_list = []
            for i in range(0, len(data), self.batch_size):
                input_sents = data[i:i+self.batch_size]
                input_tensor = self.sents2tensor(input_sents)
                features = self.Encoder(input_tensor)
                if depth == 0:
                    pass
                elif depth >= 1:
                    features = torch.cat([self.extract_hidden_decoder_features(features, label_name, num_layers=depth) for label_name in labels], dim=1)
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
        train_set, 
        val_set,
        calibration_set,
        epochs, 
        learning_rate=1.0e-4, 
        optim_type='AdamW',
        num_warmup_steps=30000,
        T_max=10000, # setup consine LR params, lr = lr_max * 0.5 * (1 + cos( steps / T_max ))
        weight_decay=0.01,
        patience=30
        ):
        # Load the models and optimizer
        optimizers_dict = {
            'AdamW': partial(
                torch.optim.AdamW, 
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'Adam': partial(
                torch.optim.Adam,
                lr = learning_rate, 
                betas = (0.9, 0.98),
                weight_decay = weight_decay
            ),
            'SGD': partial(
                torch.optim.SGD,
                lr=learning_rate, 
                momentum=0.8, 
                weight_decay=weight_decay
            ),
            'Adagrad': partial(
                torch.optim.Adagrad,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adadelta': partial(
                torch.optim.Adadelta,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'RMSprop': partial(
                torch.optim.RMSprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Adamax': partial(
                torch.optim.Adamax,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
            'Rprop': partial(
                torch.optim.Rprop,
                lr = learning_rate, 
                weight_decay = weight_decay
            ),
        }
        if optim_type not in optimizers_dict:
            print(f"Invalid optimizer type {optim_type}. Defaulting to AdamW.")
            optim_type = 'AdamW'
        # Set up the optimizer
        optimizer = optimizers_dict[optim_type](self.parameters())
        # set up the early stop params
        best_score = self.evaluate(val_set)["SPEARMANR"].mean() * self.evaluate(calibration_set)["SPEARMANR"].mean()
        print(f"NOTE: The initial model score is {round(best_score, 4)}.")
        bacth_group = 50
        mini_epoch = 500
        batch_id = 0
        # setup warm up params
        warmup_factor = 0.1
        # Apply warm-up learning rate
        if num_warmup_steps > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor 
        # Set a counter to monitor the network reaching convergence
        counter = 0
        # training
        positive_set = train_set[train_set['ShapeScore']>0.6].reset_index(drop=True)
        negative_set = train_set[train_set['ShapeScore']<=0.6].reset_index(drop=True)
        dataset_size = len(positive_set)
        print(f"NOTE: The positive set (ShapeScore > 0.6) size is {dataset_size}.")
        print(f"NOTE: 2 * {dataset_size} data points per epoch.")
        for epoch in range(epochs):
            self.train()
            pos_subset = positive_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            neg_subset = negative_set.sample(frac=1).head(dataset_size).reset_index(drop=True)
            start = time.time()
            for i in range(0, dataset_size, self.batch_size//2):
                batch_id += 1
                rows = pd.concat(
                    [
                        pos_subset.iloc[i:i+self.batch_size//2], 
                        neg_subset.iloc[i:i+self.batch_size//2]
                    ], 
                    ignore_index=True
                )
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
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {round(time.time()-start, 2)}, train loss: {loss.item()}")
                    # Set up the learning rate scheduler
                    if batch_id <= num_warmup_steps:
                        # Apply warm-up learning rate schedule
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate * ( warmup_factor + (1 - warmup_factor)* ( batch_id / num_warmup_steps) )
                    elif batch_id > num_warmup_steps:
                        # Apply cosine learning rate schedule
                        lr = learning_rate * ( warmup_factor + (1 - warmup_factor) * np.cos(float(batch_id % T_max)/T_max)) 
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                if batch_id % mini_epoch == 0:
                    # Evaluation on validation set, if provided
                    val_res = self.evaluate(val_set)
                    calibration_res = self.evaluate(calibration_set)
                    self.train()
                    print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                    print(val_res)
                    print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the calibration set:")
                    print(calibration_res)
                    val_spearmanr = np.mean(val_res["SPEARMANR"].to_list())
                    cal_spearmanr = np.mean(calibration_res["SPEARMANR"].to_list())
                    print(f"NOTE: Epoch {epoch+1}, batch {batch_id}, Val: {round(val_spearmanr, 4)}, Cal: {round(cal_spearmanr, 4)}, patience: {patience-counter}")
                    model_score = val_spearmanr * cal_spearmanr
                    if model_score is None and os.path.exists(f'{self.model_name}/GeminiMol.pt'):
                        print("NOTE: The parameters don't converge, back to previous optimal model.")
                        self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
                    elif model_score > best_score:
                        if counter > 0:
                            counter -= 1
                        best_score = model_score
                        torch.save(self.state_dict(), f"{self.model_name}/GeminiMol.pt")
                    else:
                        counter += 1
                    if counter >= patience:
                        print("NOTE: The parameters was converged, stop training!")
                        break
            if counter >= patience:
                break
        val_res = self.evaluate(val_set)
        print(f"Training over! Evaluating on the validation set:")
        print(val_res)
        print(f"Evaluating on the calibration set:")
        print(calibration_res)
        model_score = np.mean(val_res["SPEARMANR"].to_list()) * np.mean(calibration_res["SPEARMANR"].to_list())
        if model_score > best_score:
            best_score = model_score
            torch.save(self.state_dict(), f"{self.model_name}/GeminiMol.pt")
        print(f"Best Model Score: {best_score}")
        self.load_state_dict(torch.load(f'{self.model_name}/GeminiMol.pt'))
    
class PropDecoder(nn.Module):
    def __init__(self, 
        feature_dim = 1024,
        expand_ratio = 0,
        hidden_dim = 1024,
        num_layers = 3,
        dense_dropout = 0.0,
        dropout_rate = 0.1, 
        rectifier_activation = 'LeakyReLU',
        concentrate_activation = 'ELU',
        dense_activation = 'ELU',
        projection_activation = 'ELU',
        projection_transform = 'Sigmoid',
        linear_projection = True
        ):
        super(PropDecoder, self).__init__()
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
            'Softmax': nn.Softmax(),
            'Sigmoid': nn.Sigmoid(),
            'Identity': nn.Identity()
        }
        if expand_ratio > 0:
            self.features_rectifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * expand_ratio, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(feature_dim * expand_ratio),
                nn.Dropout(p=dropout_rate, inplace=False), 
                nn.Linear(feature_dim * expand_ratio, hidden_dim, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(hidden_dim),
            )
        else:
            self.features_rectifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim, bias=True),
                activation_dict[rectifier_activation],
                nn.BatchNorm1d(hidden_dim),
            )
        self.concentrate = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(hidden_dim, 128, bias=True), 
            activation_dict[concentrate_activation],
            nn.BatchNorm1d(128)
        )
        self.dense_layers =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128, bias=True),
                activation_dict[dense_activation],
                nn.BatchNorm1d(128),
                nn.Dropout(p=dense_dropout, inplace=False)
            )
            for _ in range(num_layers)
        ])
        self.projection = nn.Sequential(
            nn.Linear(128, 128, bias=True), 
            activation_dict[projection_activation],
            nn.Linear(128, 1, bias=True),
            activation_dict[projection_transform],
            nn.Linear(1, 1, bias=True) if linear_projection else nn.Identity(),
        )
        self.features_rectifier.cuda()
        self.concentrate.cuda()
        self.dense_layers.cuda()
        self.projection.cuda()
    
    def forward(self, features):
        features = self.features_rectifier(features)
        features = self.concentrate(features)
        for layer in self.dense_layers:
            features = layer(features)
        return self.projection(features).squeeze(-1)

class GeminiMol(BinarySimilarity):
    def __init__(self, 
            model_name, 
            batch_size = None, 
            depth = 0, 
            custom_label = None, 
            internal_label_list = None,
            extrnal_label_list = ['Cosine', 'Pearson', 'RMSE', 'Manhattan'] # 'Minkowski', 'Euclidean', 'KLDiv'
            ):
        self.model_name = model_name
        self.features_depth = depth
        with open(f"{model_name}/model_params.json", 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        if not batch_size is None:
            self.params['batch_size'] = batch_size
        if os.path.exists(f'{self.model_name}/feat_stat.csv'):
            self.noise_stat=pd.read_csv(f"{model_name}/feat_stat.csv")
        else:
            self.noise_stat=None
        super().__init__(model_name=model_name, **self.params)
        if custom_label is not None:
            self.label_list = custom_label
        self.eval()
        if internal_label_list is None:
            self.internal_label_list = ['ShapeScore', 'ShapeAggregation', 'CrossSim', 'CrossAggregation']
        else:
            self.internal_label_list = internal_label_list
        self.similarity_metrics_list = self.internal_label_list + extrnal_label_list
        if os.path.exists(f'{self.model_name}/propdecoders'):
            for propdecoder_mode in os.listdir(f'{self.model_name}/propdecoders'):
                if os.path.exists(f"{propdecoder_mode}/predictor.pt"):
                    with open(f"{propdecoder_mode}/model_params.json", 'r', encoding='utf-8') as f:
                        params = json.load(f)
                self.Decoder[propdecoder_mode] = PropDecoder(
                        feature_dim = params['feature_dim'],
                        rectifier_type = params['rectifier_type'],
                        rectifier_layers = params['rectifier_layers'],
                        hidden_dim = params['hidden_dim'],
                        dropout_rate = params['dropout_rate'], 
                        expand_ratio = params['expand_ratio'], 
                        activation = params['activation']
                    )
                self.Decoder[propdecoder_mode].load_state_dict(
                    torch.load(f'{propdecoder_mode}/predictor.pt')
                )
    
    def prop_decode(self, 
        df, 
        predictor_name, 
        smiles_name = 'smiles',
        as_pandas = True
        ):
        self.Decoder[predictor_name].eval()
        with torch.no_grad():
            pred_values = []
            for i in range(0, len(df), self.batch_size):
                rows = df.iloc[i:i+self.batch_size]
                smiles = rows[smiles_name].to_list()
                features = self.encode(smiles)
                pred = self.Decoder[predictor_name](features).cuda()
                pred_values += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame({predictor_name: pred_values}, columns=predictor_name)
                return res_df
            else:
                return {predictor_name: pred_values}

    def create_database(self, query_smiles_table, smiles_column='smiles', worker_num=1):
        data = query_smiles_table[smiles_column].tolist()
        query_smiles_table['features'] = None
        with torch.no_grad():
            for i in range(0, len(data), 2*worker_num*self.batch_size):
                input_sents = data[i:i+2*worker_num*self.batch_size]
                input_tensor = self.sents2tensor(input_sents)
                features = self.Encoder(input_tensor)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    query_smiles_table.at[i+j, 'features'] = features_list[j]
        return query_smiles_table

    def similarity_predict(self, shape_database, ref_smiles, ref_as_frist=False, as_pandas=True, similarity_metrics=None, worker_num=1):
        if similarity_metrics == None:
            similarity_metrics = self.similarity_metrics_list
        features_list = shape_database['features'].tolist()
        with torch.no_grad():
            pred_values = {key:[] for key in similarity_metrics}
            ref_features = self.encode([ref_smiles]).cuda()
            for i in range(0, len(features_list), worker_num*self.batch_size):
                features_batch = features_list[i:i+worker_num*self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                if ref_as_frist == False:
                    features = torch.cat((query_features, ref_features.repeat(len(features_batch), 1)), dim=0)
                else:
                    features = torch.cat((ref_features.repeat(len(features_batch), 1), query_features), dim=0)
                for label_name in similarity_metrics:
                    pred = self.decode(features, label_name, len(features_batch), depth=self.features_depth)
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=similarity_metrics)
                return res_df
            else:
                return pred_values 

    def virtual_screening(self, 
            ref_smiles_list, 
            query_smiles_table, 
            input_with_features = False,
            reverse = False, 
            smiles_column = 'smiles', 
            return_all_col = True,
            similarity_metrics = None, 
            worker_num = 1
        ):
        if input_with_features:
            features_database = query_smiles_table
        else:
            features_database = self.create_database(
                query_smiles_table, 
                smiles_column = smiles_column, 
                worker_num = worker_num
            )
        total_res = pd.DataFrame()
        for ref_smiles in ref_smiles_list:
            query_scores = self.similarity_predict(
                features_database, 
                ref_smiles, 
                ref_as_frist=reverse, 
                as_pandas=True, 
                similarity_metrics=similarity_metrics
            )
            assert len(query_scores) == len(query_smiles_table), f"Error: different length between original dataframe with predicted scores! {ref_smiles}"
            if return_all_col:
                total_res = pd.concat([total_res, query_smiles_table.join(query_scores, how='left')], ignore_index=True)
            else:
                total_res = pd.concat([total_res, query_smiles_table[[smiles_column]].join(query_scores, how='left')], ignore_index=True)
        return total_res

    '''
    This method is employed to extract pre-trained GeminiMol encodings from raw molecular SMILES representations.
    query_smiles_table: pd.DataFrame
    '''
    def extract_features(self, query_smiles_table, smiles_column='smiles'):
        shape_features = self.extract(
            query_smiles_table, 
            smiles_column, 
            as_pandas=False, 
            depth=self.features_depth, 
            labels=self.label_list
        )
        return pd.DataFrame(shape_features).add_prefix('GM_')
    

