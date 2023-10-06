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
        # s_n = gamma * s_n_1 + K^T @ V

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
    def __init__(self, layers, hidden_dim, ffn_size, heads):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(), # we use GELU instead of SiLU in GitHub's implementation (https://github.com/Jamie-Stirling/RetNet)
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
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        n: int
        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n) # o_n: (batch_size, 1, hidden_size), s_n: (batch_size, hidden_size // heads, hidden_size // heads), n: int
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x * math.sqrt(self.pe.shape[1]) # scaling
        seq_len = x.shape[1]
        pe = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pe
        return self.dropout(x)

class DeepShapeEncoder(nn.Module):
    def __init__(self, n_tokens, encoder_type='BERT', n_embed=128, n_heads=4, n_layers=12, fft_ratio=2, positional_dropout_rate=0.1, attention_dropout_rate=0.1, act_function='gelu', layer_norm_eps=1e-12):
        super(DeepShapeEncoder, self).__init__()
        self.n_embed = n_embed
        self.embed = nn.Embedding(num_embeddings=n_tokens, embedding_dim=n_embed)
        self.pos_enc = PositionalEncoding(n_embed, dropout=positional_dropout_rate)
        self.encoder_type = encoder_type
        if encoder_type == 'RetNet':
            self.encoder = RetNet(n_layers, n_embed, n_embed*fft_ratio, n_heads)
        elif encoder_type == 'BERT':
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_heads, dropout=attention_dropout_rate, activation=act_function, layer_norm_eps=layer_norm_eps), num_layers=n_layers)
        self.cuda()
        
    def forward(self, input_tensor):
        # Embedded tensor shape: [batch_size, seq_len, n_embed]
        embedded = self.embed(input_tensor) * math.sqrt(self.n_embed)
        if self.encoder_type == 'RetNet':
            encoding = self.encoder(embedded)
        elif self.encoder_type == 'BERT':
            pos_enc = self.pos_enc(embedded)
            encoding = self.encoder(pos_enc)
        return encoding

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

class ShapePooling(nn.Module):
    def __init__(self, attention_size=32):
        super().__init__()
        self.pooling = nn.ModuleDict({
                    'mean': nn.AdaptiveAvgPool1d(1),
                    'max': nn.AdaptiveMaxPool1d(1),
                    'attention': SelfAttentionPooling(attention_size=attention_size),
                })
        self.cuda()
        
    def forward(self, pos_enc):
        return torch.cat((
            pos_enc[:, 0, :,].squeeze(1), 
            self.pooling['mean'](pos_enc.transpose(1, 2)).squeeze(-1), 
            self.pooling['max'](pos_enc.transpose(1, 2)).squeeze(-1), 
            self.pooling['attention'](pos_enc).squeeze(1)
            ), dim=1)

class Shape2Score(nn.Module):
    def __init__(self, n_embed=256, expand_ratio=4, dropout_rate=0.1, decoder_type='GELU'):
        super(Shape2Score, self).__init__()
        self.decoder = nn.ModuleDict({
            'GELU': nn.Sequential(
                nn.Linear(n_embed*expand_ratio, n_embed*expand_ratio*2),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*expand_ratio*2, n_embed*3),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*3),
                nn.Dropout(dropout_rate),
                nn.Linear(n_embed*3, n_embed),
                nn.GELU(),
                nn.Linear(n_embed, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Identity(),
            ),
            'AdvanceGELU': nn.Sequential(
                nn.Linear(n_embed*expand_ratio, n_embed*expand_ratio*2),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(0.3),
                nn.Linear(n_embed*expand_ratio*2, 1024),
                nn.GELU(),
                nn.Linear(1024, 128),
                nn.GELU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Identity(),
            ),
            'Advance': nn.Sequential(
                nn.Linear(n_embed*expand_ratio, n_embed*expand_ratio*2),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(0.3),
                nn.Linear(n_embed*expand_ratio*2, 1024),
                nn.ReLU(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Identity(),
            ),
            'Extend': nn.Sequential(
                nn.Linear(n_embed*expand_ratio, n_embed*expand_ratio*2),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*expand_ratio*2, n_embed*6),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*6),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*6, n_embed*6),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*6),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*6, n_embed*6),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*6),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*6, n_embed*3),
                nn.GELU(),
                nn.BatchNorm1d(n_embed*3),
                nn.Dropout(dropout_rate),
                nn.Linear(n_embed*3, n_embed),
                nn.GELU(),
                nn.Linear(n_embed, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Identity(),
            ),
            'RELU': nn.Sequential(
                nn.Linear(n_embed*expand_ratio, n_embed*expand_ratio*2),
                nn.ReLU(),
                nn.BatchNorm1d(n_embed*expand_ratio*2),
                nn.Dropout(dropout_rate*2),
                nn.Linear(n_embed*expand_ratio*2, n_embed*3),
                nn.ReLU(),
                nn.BatchNorm1d(n_embed*3),
                nn.Dropout(dropout_rate),
                nn.Linear(n_embed*3, n_embed),
                nn.ReLU(),
                nn.Linear(n_embed, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Identity(),
            )})
        self.decoder_type = decoder_type
        self.cuda()
        
    def forward(self, features):
        score = self.decoder[self.decoder_type](features)
        prediction = torch.sigmoid(score).squeeze()
        return prediction

class BinarySimilarity(nn.Module):  
    def __init__(self, model_name, vocab_dict, feature_list=['smiles1','smiles2'], label_dict={'MCS':0.4, 'Max':0.4, 'Mean':0.1, 'Min':0.1}, n_embed=512, n_heads=8, n_layers=12, batch_size=128, dropout_rate=0.1, positional_dropout_rate=0.1, fft_ratio=2, attention_dropout_rate=0.1, act_function='gelu', layer_norm_eps=1e-12, attention_size=32, encoder_type='RetNet', decoder_type='GELU'):
        super(BinarySimilarity, self).__init__()
        torch.set_float32_matmul_precision('high') 
        self.model_name = model_name # model name
        self.vocab_dict = vocab_dict # {token: index}
        self.batch_size = batch_size # batch size
        self.feature_list = feature_list # feature list
        self.label_list = list(label_dict.keys()) # convert {label:weight} dict to label list
        ## create DeepShapeEncoder and ShapePooling module
        self.transformer = nn.DataParallel(DeepShapeEncoder(n_tokens=len(self.vocab_dict), encoder_type=encoder_type, n_embed=n_embed, n_heads=n_heads, n_layers=n_layers, fft_ratio=fft_ratio, positional_dropout_rate=positional_dropout_rate, attention_dropout_rate=attention_dropout_rate, act_function=act_function, layer_norm_eps=layer_norm_eps))
        self.pooling = nn.DataParallel(ShapePooling(attention_size=attention_size))
        # create multiple decoders
        self.Decoder = nn.ModuleDict()
        for label in self.label_list:
            self.Decoder[label] = nn.DataParallel(Shape2Score(n_embed=n_embed*2, expand_ratio=4, dropout_rate=dropout_rate, decoder_type=decoder_type))
        ## load model
        if os.path.exists(f'{self.model_name}/DeepShape.pt'):
            self.load_state_dict(torch.load(f'{self.model_name}/DeepShape.pt'))
        else:
            os.mkdir(self.model_name)
        ## set loss weight
        loss_weight = list(label_dict.values())
        sum_weight = np.sum(loss_weight)
        self.loss_weight = {key: value/sum_weight for key, value in zip(self.label_list, loss_weight)}

    def token_embedding(self, input_string, max_seq_len):
        # Convert input sentence to a tensor of numerical indices
        indices = [self.vocab_dict['[CLS]']] # [CLS]: 542
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
                indices.append(self.vocab_dict['[UNK]']) # No matching word found, use character index 540 ([UNK])
                i += 1
            if len(indices) == max_seq_len:
                break
        pad_len = max_seq_len - len(indices)
        indices += [self.vocab_dict['<EOS>']] # <EOS>
        indices += [self.vocab_dict['<PAD>']] * pad_len # <PAD>
        # Reshape indices batch to a rectangular shape, with shape (batch_size, seq_len)
        indices = torch.tensor(indices).unsqueeze(0).cuda()
        # Return indices batch
        return indices

    def sents2tensor(self, input_sents):
        # Get the max sequence length in current batch
        max_seq_len = max([len(s) for s in input_sents])
        # Create input tensor of shape (batch_size, seq_len)
        input_tensor = torch.cat([self.token_embedding(s, max_seq_len) for s in input_sents], dim=0)
        return input_tensor        

    def forward(self, sent1, sent2, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # Concatenate input sentences
        input_sents = sent1 + sent2
        input_tensor = self.sents2tensor(input_sents).cuda()
        # Encode all sentences using the transformer
        features = self.transformer(input_tensor)
        features = self.pooling(features)
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
        input_tensor = self.sents2tensor(input_sents).cuda()
        features = self.transformer(input_tensor)
        features = self.pooling(features)
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
                input_tensor = self.sents2tensor(input_sents).cuda()
                features = self.transformer(input_tensor)
                features = self.pooling(features)
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
                    labels = torch.tensor(rows[label].to_list(), dtype=torch.float32).cuda() #.unsqueeze(-1)
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
                        print(f"Epoch {epoch+1}, batch {batch_id}, evaluate on the validation set:")
                        print(val_res)
                        val_spearmanr = np.sum(val_res["SPEARMANR"].to_list())
                        val_spearmanr_list.append(val_spearmanr)
                        if val_spearmanr is None and os.path.exists(f'{self.model_name}/DeepShape.pt'):
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/DeepShape.pt'))
                        elif val_spearmanr > best_val_spearmanr:
                            best_val_spearmanr = val_spearmanr
                            torch.save(self.state_dict(), f"{self.model_name}/DeepShape.pt")
                if batch_id % T_max == 0:
                    if best_val_spearmanr <= bak_val_spearmanr:
                        counter += 1
                        if counter >= 2:
                            print("NOTE: The parameters don't converge, back to previous optimal model.")
                            self.load_state_dict(torch.load(f'{self.model_name}/DeepShape.pt'))
                        elif counter >= 5:
                            break
                    else:
                        counter = 0
                        bak_val_spearmanr = best_val_spearmanr
        if val_df is not None:
            print(f"Best val SPEARMANR: {best_val_spearmanr}")
        self.load_state_dict(torch.load(f'{self.model_name}/DeepShape.pt'))

class DeepShape(BinarySimilarity):
    def __init__(self, model_name):
        self.model_name = model_name
        with open(f"{model_name}/deepshape_params.json", 'r', encoding='utf-8') as f:
            params = json.load(f)
        with open(f"{model_name}/tokenizer.json", 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        with open(f"{model_name}/label_params.json", 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        super().__init__(model_name=model_name, vocab_dict=vocab_dict, label_dict=label_dict, **params)
        self.similarity_metrics_list = self.label_list + ['RMSE', 'Cosine', 'Manhattan', 'Minkowski', 'Euclidean', 'KLDiv', 'Pearson']

    def create_database(self, query_smiles_table, smiles_column='smiles'):
        self.eval()
        data = query_smiles_table[smiles_column].tolist()
        query_smiles_table['features'] = None
        with torch.no_grad():
            for i in range(0, len(data), 2*self.batch_size):
                input_sents = data[i:i+2*self.batch_size]
                input_tensor = self.sents2tensor(input_sents).cuda()
                features = self.transformer(input_tensor)
                features = self.pooling(features)
                features_list = list(features.cpu().detach().numpy())
                for j in range(len(features_list)):
                    query_smiles_table.at[i+j, 'features'] = features_list[j]
        return query_smiles_table

    def similarity_predict(self, shape_database, ref_smiles, ref_as_frist=False, as_pandas=True, similarity_metrics=None):
        self.eval()
        if similarity_metrics == None:
            similarity_metrics = self.similarity_metrics_list
        features_list = shape_database['features'].tolist()
        pred_values = {key:[] for key in similarity_metrics}
        with torch.no_grad():
            ref_features = self.encode([ref_smiles]).cuda()
            for i in range(0, len(features_list), self.batch_size):
                features_batch = features_list[i:i+self.batch_size]
                query_features = torch.from_numpy(np.array(features_batch)).cuda()
                if ref_as_frist == False:
                    features = torch.cat((query_features, ref_features.repeat(len(features_batch), 1)), dim=0)
                else:
                    features = torch.cat((ref_features.repeat(len(features_batch), 1), query_features), dim=0)
                for label_name in similarity_metrics:
                    pred = self.decode(features, label_name, len(features_batch))
                    pred_values[label_name] += list(pred.cpu().detach().numpy())
            if as_pandas == True:
                res_df = pd.DataFrame(pred_values, columns=similarity_metrics)
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
        return pd.DataFrame(shape_features).add_prefix('DS_')

class DeShapeDecoder(nn.Module):
    def __init__(self, n_embed=128, n_heads=4, n_layers=12, dropout_rate=0.1, act_function='gelu', layer_norm_eps=1e-12, n_tokens=512):
        super(DeShapeDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=n_embed, nhead=n_heads, dropout=dropout_rate, activation=act_function, layer_norm_eps=layer_norm_eps, batch_first=True), num_layers=n_layers)
        self.mapping = nn.Linear(n_embed, n_tokens)
        self.cuda()
        
    def forward(self, features):
        # Create a tensor of ones with the same shape as `features`
        memory = torch.ones(features.shape).cuda()
        # Embedded tensor shape: [batch_size, seq_len, n_embed]
        decoded = self.decoder(memory, features)
        return self.mapping(decoded)

class DeShape(nn.Module):
    def __init__(self, model_name, batch_size=None):
        super(DeShape, self).__init__()
        self.model_name = model_name
        self.deepshape = DeepShape(model_name)
        self.deepshape.eval()
        self.vocab_dict = {v: k for k, v in self.deepshape.vocab_dict.items()}
        with open(f"{model_name}/deshape_params.json", 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self.smiles_decoder = nn.DataParallel(DeShapeDecoder(n_embed=self.deepshape.params['n_embed'], n_heads=self.deepshape.params['n_heads'], n_layers=self.params['num_decoder_layers'], dropout_rate=self.deepshape.params['attention_dropout_rate'], act_function=self.deepshape.params['act_function'], layer_norm_eps=self.deepshape.params['layer_norm_eps'], n_tokens=len(self.deepshape.vocab_dict)))
        if os.path.exists(f'{self.model_name}/DeShape.pt'):
            self.smiles_decoder.load_state_dict(torch.load(f'{self.model_name}/DeShape.pt'))
        if batch_size == None:
            self.batch_size = self.params['batch_size']
        else:
            self.batch_size = batch_size  

    def forward(self, features):
        # features (batch size，seq_len, embed_dim)
        return self.smiles_decoder(features)
        
    def tensor_to_text(self, tensor):
        # tensor: (batch_size, seq_len)
        # vocab: a dictionary that maps token ids to tokens
        # Convert the tensor to a numpy array
        tensor = tensor.detach().cpu().numpy()
        # Iterate over each sequence in the batch
        texts = []
        for sequence in tensor:
            # Convert each token id in the sequence to its corresponding token
            text = [self.vocab_dict[token_id] for token_id in sequence if token_id not in [self.deepshape.vocab_dict['[CLS]'], self.deepshape.vocab_dict['[UNK]'], self.deepshape.vocab_dict['<PAD>'], self.deepshape.vocab_dict['<EOS>']]] # Remove padding tokens
            # Concatenate the tokens into a single string and add it to the list of texts
            texts.append(''.join(text))
        return texts

    def decoder(self, features):
        self.eval()
        with torch.no_grad():
            decoded_tensor = self.smiles_decoder(features)
            # Find the index of the maximum value along the last dimension
            decoded_text = self.tensor_to_text(torch.argmax(decoded_tensor, dim=-1)) 
        return decoded_text

    def evaluate(self, input_smiles_list):
        self.eval()
        output_smiles_list = []
        with torch.no_grad():
            output_smiles_list = [ self.decoder(self.deepshape.transformer(self.deepshape.sents2tensor(input_smiles_list[i:i+self.batch_size]).cuda())) for i in range(0, len(input_smiles_list), self.batch_size)]
        num_correct = sum(1 for input_smiles, output_smiles in zip(input_smiles_list, output_smiles_list) if input_smiles == output_smiles)
        recovery = num_correct / len(input_smiles_list)
        return recovery

    def fit(self, train_smiles_list, epochs, learning_rate=1.0e-3, val_smiles_list=None, optim_type='AdamW'):
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
        optimizer = optim_types[optim_type](self.parameters(), lr=learning_rate, weight_decay=1.0e-3)
        train_loss = []
        batch_id = 0
        # loss fn
        loss_fn = nn.CrossEntropyLoss()
        train_loss = []
        best_val_recovery = -1
        for epoch in range(epochs):
            total_loss = 0.0
            self.train()
            train_df = shuffle(train_df) # shuffle data set per epoch
            start = time.time()
            for i in range(0, len(train_smiles_list), self.batch_size):
                batch_id += 1
                input_smiles_list = train_smiles_list[i:i+self.batch_size]
                input_tensor = self.deepshape.sents2tensor(input_smiles_list).cuda()
                with torch.no_grad():
                    features = self.deepshape.transformer(input_tensor)
                decoded_tensor = self.forward(features)
                # If labels are provided, calculate loss and return it along with predicted scores
                loss = loss_fn(decoded_tensor.transpose(1, 2), input_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id % 50 == 0:
                    print(f"Epoch {epoch+1}, batch {batch_id}, time {time.time()-start}, train loss: {loss.item():.4f}")
                total_loss += loss.item() * self.batch_size / len(train_smiles_list)
                if batch_id % 300 == 0:
                    if val_smiles_list is not None:
                        val_recovery = self.evaluate(val_smiles_list)
                        print(f"Epoch {epoch+1}, batch {batch_id}, recovery on the validation set: {val_recovery:.4f}")
                        if val_recovery > best_val_recovery:
                            best_val_recovery = val_recovery
                            torch.save(self.smiles_decoder.state_dict(), f"{self.model_name}/DeShape.pt")
            train_loss.append(total_loss)
            # Evaluation on validation set, if provided
            if val_smiles_list is not None:
                val_recovery = self.evaluate(val_smiles_list)
                print(f"Epoch {epoch+1}, train loss: {total_loss:.4f}, recovery on the validation set: {val_recovery:.4f}")
                if val_recovery > best_val_recovery:
                    best_val_recovery = val_recovery
                    torch.save(self.smiles_decoder.state_dict(), f"{self.model_name}/DeShape.pt")
            else:
                print(f"Epoch {epoch+1}, train loss: {total_loss:.4f}")
        if val_smiles_list is not None:
            print(f"Best val SPEARMANR: {best_val_recovery}")
        self.smiles_decoder.weight.data = torch.load(f'{self.model_name}/DeShape.pth')['smiles_decoder.weight']

    def MCMC(self, ref_smiles, replica_num=10, num_steps_per_replica=3, num_seeds_per_steps=10, temperature=0.1, scaling_factor=0.5):
        self.eval()
        with torch.no_grad():
            ref_features = self.deepshape.encode([ref_smiles]).cuda()
            ref_features.repeat(self.batch_size, 1)
            batch_size, sequence_length, embedding_size = ref_features.shape
            output_smiles_list = []
            for _ in range(replica_num):
                sampling_tensor = ref_features
                for step in range(num_steps_per_replica):
                    sampling_tensor = sampling_tensor * (1 + temperature * (scaling_factor ** (step)) * torch.randn(batch_size, sequence_length, embedding_size))
                    loss = self.target_function[self.task_type](ref_features, sampling_tensor, batch_size)
                    _, topk_indices = torch.topk(loss, k=num_seeds_per_steps, largest=False)
                    selected_indices = topk_indices.repeat(batch_size // num_seeds_per_steps)
                    if (batch_size-(batch_size // num_seeds_per_steps)*num_seeds_per_steps) != 0:
                        selected_indices = torch.cat([selected_indices, topk_indices[:batch_size-(batch_size // num_seeds_per_steps)*num_seeds_per_steps]])
                    sampling_tensor = torch.index_select(sampling_tensor, dim=0, index=selected_indices)
                output_smiles_list += self.decoder(sampling_tensor)
        return output_smiles_list

    def directed_evolution(self, ref_smiles, optim_type='SGD', replica_num=5, num_steps_per_replica=30, scaling_factor=0.5):
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
        optimizer = optim_types[optim_type](self.parameters(), lr=0.01, lr_decay=0.9)
        ref_features = self.deepshape.encode([ref_smiles]).cuda()
        ref_features.repeat(self.batch_size, 1)
        batch_size, sequence_length, embedding_size = ref_features.shape
        output_smiles_list = []
        for _ in range(replica_num):
            sampling_tensor = torch.tensor(sampling_tensor * (1 + scaling_factor * torch.randn(batch_size, sequence_length, embedding_size)), requires_grad=True)
            for _ in range(num_steps_per_replica):
                loss = self.target_function[self.task_type](ref_features, sampling_tensor, batch_size)
                loss.backward()
                optimizer.step()
            output_smiles_list += self.decoder(sampling_tensor)
        return output_smiles_list

    def scaffold_hopping(self, ref_features, sampling_tensor, batch_size):
        features = torch.cat((sampling_tensor, ref_features), dim=0)
        pred_MCS = self.deepshape.decode(features, 'MCS', batch_size)
        pred_CEO = self.deepshape.decode(features, 'Max', batch_size)
        loss = ( pred_MCS - pred_CEO ) * pred_CEO
        return loss

    def __call__(self, ref_smiles, task_type):
        self.target_function = {
            'scaffold_hopping' : lambda ref_features, sampling_tensor, batch_size: self.scaffold_hopping(ref_features, sampling_tensor, batch_size), 
        }
        self.task_type = task_type
        output_smiles_list = self.MCMC(ref_smiles, replica_num=10, num_steps_per_replica=3, num_seeds_per_steps=10, temperature=0.1, scaling_factor=0.5)
        output_smiles_list += self.directed_evolution(ref_smiles, optim_type='SGD', replica_num=5, num_steps_per_replica=30, scaling_factor=0.5)
        smiles_table = pd.DataFrame({'smiles':output_smiles_list}, columns=['smiles'])
        shape_database = self.deepshape.create_database(smiles_table, smiles_column='smiles')
        query_scores = self.deepshape.similarity_predict(shape_database, ref_smiles)
        return smiles_table.join(query_scores, how='left')



