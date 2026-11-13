import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
import networkx as nx

import torch
from torch_geometric.data import Data, Batch
import random

import learners.common as LC

class BipartiteGraphConvolution(MessagePassing):
    def __init__(self, 
                 emb_size, 
                 activation_name="relu", 
                 aggregation='mean', 
                 right_to_left=False, 
                 bias=True, 
                 dev=False, name=None):
        super(BipartiteGraphConvolution, self).__init__(aggr=aggregation, flow="source_to_target")
        self.emb_size = emb_size
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.right_to_left = right_to_left
        self.bias = bias
        self.dev = dev
        self.name = name
        self._build_layers()


    def _build_layers(self):
        print("---> WARN: creating feature_module_right with bias=True !!! THE ACTUAL PARAMETER IS IGNORED")
        self.feature_module_left = nn.Sequential(
            # bias set to true as in https://github.com/liujl11git/GNN-MILP/blob/main/models.py
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            LC.activations[self.activation_name],
            # below bias is True, as in Keras when not speciefied
            nn.Linear(self.emb_size, self.emb_size)
        )
        
        self.feature_module_edge = nn.Identity()

        print("---> WARN: creating feature_module_right with bias=False !!! THE ACTUAL PARAMETER IS IGNORED")
        self.feature_module_right = nn.Sequential(
            # bias set to false as in https://github.com/liujl11git/GNN-MILP/blob/main/models.py
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            LC.activations[self.activation_name],
            # below bias is True, as in Keras when not speciefied
            nn.Linear(self.emb_size, self.emb_size)  # here bias is True, as in Keras when not spec
        )

        # Output module
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            LC.activations[self.activation_name],
            nn.Linear(self.emb_size, self.emb_size)
        )

        self._reset_parameters()


    def _reset_parameters(self):
        for module in [self.feature_module_left, self.feature_module_right, self.output_module]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    
    def forward(self, x_left, x_right, batch):
        left_features = self.feature_module_left(x_left)
        right_features = self.feature_module_right(x_right)
        
        edge_index = batch["left", "to", "right"].edge_index
        # XXX: check here: embedding of edge_attr?
        edge_attr = batch["left", "to", "right"].edge_attr if hasattr(batch["left", "to", "right"], "edge_attr") else None
        
        if self.right_to_left:
            if self.dev:
                print()
                print(f"--{self.name}-- in forward(): RIGHT -> LEFT, edge_index shape: {edge_index.shape}")
            
            edge_index = edge_index[[1, 0]]
            src_features = right_features
            tgt_features = left_features
        else: # left to right
            if self.dev:
                print()
                print(f"--{self.name}-- in forward(): LEFT -> RIGHT, edge_index shape: {edge_index.shape}")
            src_features = left_features
            tgt_features = right_features
        if self.dev:
            print(f"--{self.name}-- in forward(): before propagate()")
            print(f"--{self.name}-- in forward(): src_features shape = {src_features.shape}")
            print(f"--{self.name}-- in forward(): tgt_features shape = {tgt_features.shape}")
            print(f"--{self.name}-- in forward(): edge_attr shape = {edge_attr.shape if edge_attr is not None else None}")
            print(f"--{self.name}-- in forward(): left_features shape = {left_features.shape}")
            print(f"--{self.name}-- in forward(): right_features shape = {right_features.shape}")
            print()

            


        out = self.propagate(x=(src_features, tgt_features),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            size=(src_features.size(0), tgt_features.size(0)), 
                            )
        
        out = self.output_module(torch.cat([out, tgt_features], dim=1))
        if self.dev:
            print(f"BIP_CONV_GCN {self.name}, output shape:", out.shape)
        return out

    
    # message to node i:
    #   (j, i) in E and flow=source_to_target
    #   (i, j) in E and flow=target_to_source
    def message(self, x_j, edge_attr=None):
        
        return edge_attr.view(-1, 1) * x_j




# ------------------------------------------------------

class BipartiteLayer(nn.Module):
    def __init__(self, emb_size, activation_name="relu", aggregation="sum", bias=True, dev=False):
        super(BipartiteLayer, self).__init__()
        print("---> BipartiteLayer: actual parameter 'bias' is ignored")
        self.emb_size = emb_size
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.bias = bias
        self.dev = dev
        self._build_layers()


    def _build_layers(self):
        # (self, emb_size, activation_name="relu", aggregation='mean', right_to_left=False, bias=True, dev=False)
        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, 
                                                     activation_name=self.activation_name, 
                                                     aggregation=self.aggregation, 
                                                     right_to_left=True, 
                                                     bias=self.bias, 
                                                     dev=self.dev, name="var-to-cons") 
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, 
                                                     activation_name=self.activation_name, 
                                                     aggregation=self.aggregation, 
                                                     right_to_left=False, 
                                                     bias=self.bias, 
                                                     dev=self.dev, name="cons-to-var")

    def _reset_parameters(self):
        self.conv_v_to_c._reset_parameters()
        self.conv_c_to_v._reset_parameters()

    def forward(self, prev_cons_features, prev_var_features, batch):
        if self.dev:
            print("---> BipartiteLayer.forward()")
            print("\t prev_cons_features.shape", prev_cons_features.shape)
            print("\t prev_var_features.shape", prev_var_features.shape)
            print("\t batch.edge_index.shape", batch["left", "to", "right"].edge_index.shape)
            print(batch)
        constraint_features = self.conv_v_to_c(prev_cons_features, prev_var_features, batch)
        if self.dev:
            print("after conv_v_to_c: constraint_features.shape", constraint_features.shape)

        variable_features = self.conv_c_to_v(prev_cons_features, prev_var_features, batch)
        if self.dev:
            print("after conv_c_to_v: variable_features.shape", variable_features.shape)
        return constraint_features, variable_features
