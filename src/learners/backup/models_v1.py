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
import learners.modules as MOD


#
# MODEL 1: GCNPolicy from paper
#

class GCNPolicy(nn.Module):
    def __init__(self, 
                 emb_size, 
                 n_cons_f, 
                 n_edge_f, 
                 n_var_f, 
                 n_layers=2, 
                 activation_name="relu", 
                 aggregation="sum",  # XXX: check aggregation
                 bias=True, 
                 is_graph_level=True, 
                 dev=False, 
                 name=None, 
                 graph_op=pyg.nn.global_add_pool):
        super(GCNPolicy, self).__init__()
        self.emb_size = emb_size
        self.n_cons_f = n_cons_f
        self.n_edge_f = n_edge_f
        self.n_var_f = n_var_f
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.bias = bias
        self.n_layers = n_layers
        self.is_graph_level = is_graph_level
        self.dev = dev
        if self.dev:
            print("----> DEV MODE SET <----")
        self.name = name
        self.graph_level_op = graph_op
        self._build_layers()


    def _build_layers(self):
        print("GCNPolicy, dimensions")
        print("\t - n_cons_f:", self.n_cons_f)
        print("\t - n_edge_f:", self.n_edge_f)
        print("\t - n_var_f:", self.n_var_f)
        print("\t - emb_size:", self.emb_size)
        print("\t ---")
        
        self.cons_embedding = nn.Sequential(
            nn.Linear(self.n_cons_f, self.emb_size),
            LC.activations[self.activation_name]
        )

        self.edge_embedding = nn.Identity()
        self.var_embedding = nn.Sequential(
            nn.Linear(self.n_var_f, self.emb_size),
            LC.activations[self.activation_name]
        )

        self.bp_layers = nn.ModuleList([
            MOD.BipartiteLayer(self.emb_size, activation_name=self.activation_name, 
                           aggregation=self.aggregation, bias=self.bias, dev=self.dev)
            for _ in range(self.n_layers)
        ])

        if self.is_graph_level:
            self.output_module = nn.Sequential(
                nn.Linear(2 * self.emb_size, self.emb_size),
                LC.activations[self.activation_name],
                nn.Linear(self.emb_size, 1)
            )
        else:
            assert False, f"Only graph level is supported for now"
        
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.bp_layers:
            layer._reset_parameters()
        # init all layers with a diagonal matrix

        nn.init.xavier_uniform_(self.cons_embedding[0].weight)
        nn.init.xavier_uniform_(self.var_embedding[0].weight)
        nn.init.xavier_uniform_(self.output_module[0].weight)
        nn.init.xavier_uniform_(self.output_module[2].weight)

    def forward(self, batch): # , cons_x, var_x, x_edge, edge_index): 
        # adj matrix E_{i, j}: constraint i to variable j
        if self.dev:
            print("---> GCNPolicy.forward()")
            print("\t cons_features.shape", batch["left"].x.shape)
            print("\t var_features.shape", batch["right"].x.shape)
            print("\t edge_features.shape", batch["left", "to", "right"].edge_attr.shape)
            
        cons_features = self.cons_embedding(batch["left"].x)
        # n_cons x emb_size
        if self.dev: print("GCNPolicy, cons_features:", cons_features.shape)

        var_features = self.var_embedding(batch["right"].x)
        # n_vars x emb_size
        if self.dev: print("GCNPolicy, var_features:", var_features.shape)

        edge_features = self.edge_embedding(batch["left", "to", "right"].edge_attr)
        # n_dedges x 1 (identity)
        if self.dev: print("GCNPolicy, edge_features:", edge_features.shape)
        
        for i in range(self.n_layers):
            if self.dev:
                print("GCNPolicy.forward(): layer", i)
            cons_features, var_features = self.bp_layers[i](cons_features, var_features, batch)
        
        if self.dev:
            print(f"GCNPolicy.forward(), after computing bipartite layers, n_layes = {self.n_layers}")
            print("constraint features (left):", cons_features.shape)
            print("variable features (right):", var_features.shape)

        
        if self.is_graph_level:
            if self.dev: print(" *** COMPUTING GRAPH LEVEL OP ***")
            # graph level pooling using pytorch geometric
            cons_features = self.graph_level_op(cons_features, batch["left"].batch)
            var_features = self.graph_level_op(var_features, batch = batch["right"].batch)
            if self.dev:
                print(" *** AFTER GRAPH LEVEL OP ***")
                print(" *** cons_features:", cons_features.shape)
                print(" *** var_features:", var_features.shape)
                print(" *** ")
            # concatenate features
            out = torch.cat([cons_features, var_features], dim=1)
        else:
            assert False, "not yet developed"

        out = self.output_module(out)
        return out

# -----------------------------------------------------------------------------------------------