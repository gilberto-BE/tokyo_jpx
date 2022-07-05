import torch
import torch.nn as nn
torch.manual_seed(0)

# import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
# pl.utilities.seed.seed_everything(seed=42)
from torch import nn, Tensor
import math
from metrics import metrics
import matplotlib.pyplot as plt
import numpy as np


class EmbeddingNetwork(nn.Module):
    def __init__(
        self, 
        units, 
        no_embedding, 
        emb_dim
    ):
        super(EmbeddingNetwork, self).__init__()
        self.units=units
        self.no_embedding = no_embedding
        self.emb_dim = emb_dim
        self.embedding_layer = nn.Embedding(self.no_embedding, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.units)
        self.out = nn.Linear(self.units, 1)
        
    def forward(self, x):
        x = F.relu(self.embedding_layer(x))
        print('x.shape after F.relu(embedding_layer(k)):', x.shape)
        x = F.relu(self.linear(x))
        print('x.shape after linear + relu:', x.shape)
        x = self.out(x)
        print('x.shape after self.out(x):', x.shape)
        print()
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[ :x.size(0)]
        return self.dropout(x)


class GatedLinearUnit(nn.Module):
    """
    Need to check all dimensions
    """
    def __init__(self, units=10):
        super(GatedLinearUnit, self).__init__()
        self.units = units
        self.layer = nn.Linear(self.units, self.units)
        self.output = nn.Linear(self.units, self.units)

    def forward(self, x):
        return self.layer(x) @ F.sigmoid(self.output(x))


class GatedResidualNetwork(nn.Module):
    """
    Need to check all dimensions
    """
    def __init__(self, in_features, out_features, dropout=0.1, units=50):
        super(GatedResidualNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.layer1 = nn.Linear(self.in_features, self.units)
        self.layer2 = nn.Linear(self.in_features, self.out_features)
        self.dropout = nn.Dropout(dropout)
        self.gated_linear_unit = GatedLinearUnit(self.in_features)
        self.layer_norm = nn.LayerNorm(self.in_features)
        self.layer2 = nn.Linear(self.units, self.units)

    def forward(self, inputs):
        x = F.elu(self.layer1(x))
        x = self.layer2(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.layer2(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x


class FFResNetBlock(nn.Module):
    """
    TODO:
    Replace, ModuleList with regular list and sequential class!
    """
    def __init__(self, in_features, out_features, n_layers=3):
        super(FFResNetBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.layers = nn.ModuleList(nn.Linear(self.in_features, self.out_features) for _ in range(self.n_layers))
        self.layer_norm = nn.LayerNorm(self.out_features)

    def forward(self, x):
        res = x
        for L in self.layers:
            x = F.relu(self.layer_norm(L(x)))
        return x + res


class NeuralBlock(nn.Module):

    def __init__(self, units, categorical_dim, res_layers=3, res_blocks=2):
        super(NeuralBlock, self).__init__()

        self.in_features = units + categorical_dim
        self.out_features = units + categorical_dim
        self.res_layers = res_layers
        self.res_blocks = res_blocks
        self.resnet_block = nn.ModuleList(
            FFResNetBlock(self.in_features, self.out_features, self.res_layers) for _ in range(self.res_blocks)
            )
        self.fwr_layer = nn.Linear(self.in_features, self.in_features)
        self.output = nn.Linear(self.in_features, self.out_features)
        self.backcast_layer = nn.Linear(self.in_features, self.in_features)
        self.back_cast = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        res = x
        x = torch.real(torch.fft.fft2(x))
        for r_block in self.resnet_block:
            x = r_block(x)
        x_res = res - x

        x = torch.real(torch.fft.ifft2(x))
        x = F.relu(self.fwr_layer(x))
        x = self.output(x)

        x_res = torch.real(torch.fft.ifft2(x_res))
        x_res = F.relu(self.backcast_layer(x_res))
        x_res = self.back_cast(x_res)
        return x_res, x


class NeuralStack(nn.Module):
    def __init__(
        self, 
        n_blocks, 
        units, 
        categorical_dim, 
        output_dim=1,
        res_layers=3,
        res_blocks=2
        ):
        super(NeuralStack, self).__init__()
        self.n_blocks = n_blocks
        self.units = units
        self.categorical_dim = categorical_dim
        self.output_dim = output_dim
        self.res_layers = res_layers
        self.res_blocks = res_blocks

        self.blocks = nn.ModuleList(
            [
                NeuralBlock(self.units, self.categorical_dim, self.res_layers, self.res_blocks) for _ in range(self.n_blocks)
            ])
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(self.units + self.categorical_dim, self.output_dim) for _ in range(self.n_blocks)
            ])

    def forward(self, x):
        predictions = 0
        x_back = x
        for block_nr, block in enumerate(self.blocks):
            x_back, pred = block(x_back)
            predictions += self.output_layers[block_nr](pred)        
        return x_back, predictions


class NeuralNetwork(nn.Module):
    """
    TODO:
    1) IMPLEMENT PREPROCESSING AS A LAYER IN THE NETWORK.
    """
    def __init__(
        self, 
        in_features, 
        out_features,
        categorical_dim=3, 
        units=512, 
        no_embedding=None, 
        emb_dim=None,
        dropout=0.1,
        n_blocks=10,
        n_stacks=8,
        pooling_sizes=3,
        res_layers=3,
        res_blocks=2,
        apply_pooling=True
        ):

        """
        TODO:
        * Add normalization layers
        * Add regularization
        * Test other optimizers
        * Add positional encoding
        """
        super(NeuralNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.units = units
        self.no_embedding = no_embedding
        self.emb_dim = emb_dim
        self.categorical_dim = categorical_dim
        self.n_blocks = n_blocks
        self.n_stacks = n_stacks
        self.pooling_sizes = pooling_sizes
        self.res_layers = res_layers
        self.res_blocks = res_blocks
        self.apply_pooling = apply_pooling

        if no_embedding and emb_dim:
            self.embedding_layer = nn.Embedding(self.no_embedding, self.emb_dim)
            self.embedding_to_hidden = nn.Linear(self.emb_dim, self.units)
            self.embedding_output = nn.Linear(self.units, self.out_features)  
            
        # if adding cont vars subtract from in-features
        self.cont_input = nn.Linear(self.in_features, self.units)
        self.dropout = nn.Dropout(dropout)
        if self.apply_pooling:
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        self.stacks = nn.ModuleList([
            NeuralStack(
                    self.n_blocks, 
                    self.units, 
                    self.categorical_dim, 
                    self.out_features, 
                    self.res_layers, 
                    self.res_blocks) for _ in range(self.n_stacks)
                    ])

    def forward(self, x, x_cat=None):
        if x_cat is not None:
            x_cat = x_cat.to(torch.int64)
            x_cat = self.embedding_layer(x_cat)
            x_cat = F.relu(self.embedding_to_hidden(x_cat))
            x_cat = F.relu(self.embedding_output(x_cat))
        x = torch.real(torch.fft.fft2(x))
        # print('x.shape after fft2:', x.shape)
        if self.apply_pooling:
            x = self.pooling_layer(x)
            # print('x.shape after pooling:', x.shape)
        x = F.relu(self.cont_input(x))
        x = torch.cat((x, x_cat.view((x_cat.size(0), -1))), dim=1)
        x = self.dropout(x)

        tot_preds = 0
        block_input = x
        for S in self.stacks:
            block_input, pred = S(block_input)
            tot_preds += pred
        return tot_preds



