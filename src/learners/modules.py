import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import learners.common as LC

class LayerNormBasicLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.ln_i = nn.LayerNorm(hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_g = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.x2h.weight)
        nn.init.orthogonal_(self.h2h.weight)
        nn.init.zeros_(self.x2h.bias)

    def forward(self, x, state):
        h, c = state
        gates = self.x2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(self.ln_i(i))
        f = torch.sigmoid(self.ln_f(f))
        g = torch.tanh(self.ln_g(g))
        o = torch.sigmoid(self.ln_o(o))
        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new))
        return h_new, c_new
    
# the number of hidden layers corresponds to the length of hidden_dims
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=None, hidden_dims=None, activation_name="leaky_relu", bias=True, dropout=0.0):
        super(MLP, self).__init__()

        assert n_layers is None and type(hidden_dims) == list, "n_layers must be None and hidden_dims must be a list"
        assert (n_layers is not None) and (type(n_layers) == int) and (type(hidden_dims) == int), "n_layers must be an int and hidden_dims must be an int"
        if n_layers is None:
            self.layers = self.init_from_list(input_dim, output_dim, hidden_dims, activation_name, bias, dropout)
        else:
            self.layers = self.init_from_int(input_dim, output_dim, n_layers, hidden_dims, activation_name, bias, dropout)
        self.model = nn.Sequential(self.layers)


    def init_from_list(self, input_dim, output_dim, hidden_dims, activation_name, bias, dropout):
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            layers.append(LC.activations[activation_name])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim, bias=bias))
        return layers

    def init_from_int(self, input_dim, output_dim, n_layers, hidden_dims, activation_name, bias, dropout):
        hidden_dims = [input_dim] * n_layers
        return self.init_from_list(input_dim, output_dim, hidden_dims, activation_name, bias, dropout)


    def forward(self, x):
        return self.model(x)

def build_mlp(input_dim, 
              output_dim, 
              n_layers=None, 
              hidden_dims=None, 
              activation_name="leaky_relu", 
              bias=True, 
              dropout=0.0):
    
    def init_from_list(input_dim, output_dim, hidden_dims, activation_name, bias, dropout):
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            layers.append(LC.activations[activation_name])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim, bias=bias))
        return layers

    def init_from_int(input_dim, output_dim, n_layers, hidden_dims, activation_name, bias, dropout):
        hidden_dims = [input_dim] * n_layers
        return init_from_list(input_dim, output_dim, hidden_dims, activation_name, bias, dropout)

    if n_layers is None:
        assert type(hidden_dims) == list, "when n_layers is None, hidden_dims must be a list"
    else: 
        assert (type(n_layers) == int) and (type(hidden_dims) == int), "n_layers must be an int and hidden_dims must be an int"
    


    if n_layers is None:
        layers = init_from_list(input_dim, output_dim, hidden_dims, activation_name, bias, dropout)
    else:
        layers = init_from_int(input_dim, output_dim, n_layers, hidden_dims, activation_name, bias, dropout)
    
    res = nn.Sequential(*layers)
    
    return res

class BipartiteGraphConvolution_v2(MessagePassing):
    def __init__(self, 
                 emb_size, 
                 activation_name="leaky_relu", 
                 aggregation='add', 
                 right_to_left=False, 
                 bias=True,
                 conv_append_activation=False,
                 dev=False, name=None, 
                 force_bias_value=False,
                 mlp_n_hidden_layers=1,
                 dropout=0.0):
        super(BipartiteGraphConvolution_v2, self).__init__(aggr=aggregation, flow="source_to_target")
        
        self.emb_size = emb_size
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.right_to_left = right_to_left
        self.bias = bias
        self.conv_append_activation = conv_append_activation
        self.dev = dev
        self.name = name
        self.force_bias_value = force_bias_value
        self.mlp_n_hidden_layers = mlp_n_hidden_layers
        self.dropout = dropout
        self._build_layers()

    def _build_layers(self):
        # Left and right base modules
        self.feature_module_left = build_mlp(input_dim=self.emb_size, 
                                             output_dim=self.emb_size, 
                                             n_layers=self.mlp_n_hidden_layers, 
                                             hidden_dims=self.emb_size, 
                                             activation_name=self.activation_name, 
                                             bias=True, 
                                             dropout=self.dropout)

        self.feature_module_right = build_mlp(input_dim=self.emb_size, 
                                             output_dim=self.emb_size, 
                                             n_layers=self.mlp_n_hidden_layers, 
                                             hidden_dims=self.emb_size, 
                                             activation_name=self.activation_name, 
                                             bias=True, 
                                             dropout=self.dropout)

        if self.conv_append_activation:      
            self.feature_module_left.append(LC.activations[self.activation_name])
            self.feature_module_right.append(LC.activations[self.activation_name])

        self.feature_module_pos = build_mlp(input_dim=self.emb_size,
                            output_dim=self.emb_size,
                            n_layers=self.mlp_n_hidden_layers,
                            hidden_dims=self.emb_size,
                            activation_name=self.activation_name,
                            bias=True,
                                            dropout=self.dropout)
        self.feature_module_neg = build_mlp(input_dim=self.emb_size,
                            output_dim=self.emb_size,
                            n_layers=self.mlp_n_hidden_layers,
                            hidden_dims=self.emb_size,
                            activation_name=self.activation_name,
                            bias=True,
                                            dropout=self.dropout)
        if self.conv_append_activation:
            self.feature_module_pos.append(LC.activations[self.activation_name])
            self.feature_module_neg.append(LC.activations[self.activation_name])

        self.vcg_proj = nn.Linear(2 * self.emb_size, self.emb_size)

        self._reset_parameters()

    def _reset_parameters(self):
        modules = [self.feature_module_left, self.feature_module_right,
                   self.feature_module_pos, self.feature_module_neg]
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    gain = nn.init.calculate_gain(self.activation_name)
                    torch.nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        gain = nn.init.calculate_gain('linear')
        torch.nn.init.orthogonal_(self.vcg_proj.weight, gain=gain)
        nn.init.zeros_(self.vcg_proj.bias)


    def forward(self, x_left, x_right, batch):
        edge_index = batch["left", "to", "right"].edge_index
        edge_attr = batch["left", "to", "right"].edge_attr if hasattr(batch["left", "to", "right"], "edge_attr") else None
        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        if self.right_to_left:
            edge_index = edge_index[[1, 0]]
            src_raw = x_right
            tgt_raw = x_left
            tgt_features = self.feature_module_left(tgt_raw)
        else:
            src_raw = x_left
            tgt_raw = x_right
            tgt_features = self.feature_module_right(tgt_raw)

        if edge_attr is not None:
            pos_mask = (edge_attr.squeeze() > 0)
            neg_mask = (edge_attr.squeeze() < 0)

            edge_index_pos = edge_index[:, pos_mask]
            edge_index_neg = edge_index[:, neg_mask]

            edge_attr_pos = torch.abs(edge_attr[pos_mask])
            edge_attr_neg = torch.abs(edge_attr[neg_mask])
        else:
            edge_index_pos = edge_index
            edge_index_neg = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            edge_attr_pos = None
            edge_attr_neg = None

        src_pos = self.feature_module_pos(src_raw)
        src_neg = self.feature_module_neg(src_raw)

        aggr_pos = self.propagate(x=(src_pos, tgt_features), edge_index=edge_index_pos, edge_attr=edge_attr_pos, size=(src_pos.size(0), tgt_features.size(0))) if edge_index_pos.size(1) > 0 else torch.zeros((tgt_features.size(0), self.emb_size), dtype=tgt_features.dtype, device=tgt_features.device)
        aggr_neg = self.propagate(x=(src_neg, tgt_features), edge_index=edge_index_neg, edge_attr=edge_attr_neg, size=(src_neg.size(0), tgt_features.size(0))) if edge_index_neg.size(1) > 0 else torch.zeros((tgt_features.size(0), self.emb_size), dtype=tgt_features.dtype, device=tgt_features.device)

        aggr_out = torch.cat([aggr_pos, aggr_neg], dim=-1)
        aggr_out = self.vcg_proj(aggr_out)

        return aggr_out

    
    # message to node i:
    #   (j, i) in E and flow=source_to_target
    #   (i, j) in E and flow=target_to_source
    def message(self, x_j, edge_attr):
        return edge_attr.view(-1, 1) * x_j



class BipartiteLayer(nn.Module):
    def __init__(self, emb_size, activation_name="leaky_relu", aggregation="add", bias=True, anticipate_mp=False, conv_order="cons-vars", 
                 conv_append_activation=False, dev=False, force_bias_value=False, n_mlp_hidden_layers=1,
                 dropout=0.0):
        super(BipartiteLayer, self).__init__()
        print("---> BipartiteLayer: actual parameter 'bias' is ignored")
        self.emb_size = emb_size
        
        self.activation_name = activation_name
        
        self.aggregation = aggregation
        self.bias = bias
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.anticipate_mp = anticipate_mp
        self.conv_order = conv_order
        self.conv_append_activation = conv_append_activation
        self.dev = dev
        self.force_bias_value = force_bias_value
        self.dropout = dropout
        self._build_layers()

    def _build_layers(self):
        self.conv_v_to_c = BipartiteGraphConvolution_v2(self.emb_size, 
                                                       activation_name=self.activation_name, 
                                                       aggregation=self.aggregation, 
                                                       right_to_left=True, 
                                                       bias=self.bias,
                                                       conv_append_activation=self.conv_append_activation,
                                                       dev=self.dev, name="var-to-cons", 
                                                       force_bias_value=self.force_bias_value,
                                                       mlp_n_hidden_layers=self.n_mlp_hidden_layers,
                                                       dropout=self.dropout)

        self.conv_c_to_v = BipartiteGraphConvolution_v2(self.emb_size, 
                                                       activation_name=self.activation_name, 
                                                       aggregation=self.aggregation, 
                                                       right_to_left=False, 
                                                       bias=self.bias,
                                                       conv_append_activation=self.conv_append_activation,
                                                       dev=self.dev, name="cons-to-var", 
                                                       force_bias_value=self.force_bias_value,
                                                       mlp_n_hidden_layers=self.n_mlp_hidden_layers,
                                                       dropout=self.dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        self.conv_v_to_c._reset_parameters()
        self.conv_c_to_v._reset_parameters()

    def forward_with_anticipation(self, prev_cons_features, prev_var_features, prev_cons_c, prev_var_c, batch):
        if self.conv_order == "cons-vars":
            constraint_features = self.conv_v_to_c(prev_cons_features, prev_var_features, batch)
            
            variable_features = self.conv_c_to_v(constraint_features, prev_var_features, batch)
                
        elif self.conv_order == "vars-cons":
            variable_features = self.conv_c_to_v(prev_cons_features, prev_var_features, batch)
            
            constraint_features = self.conv_v_to_c(prev_cons_features, variable_features, batch)
        else:
            raise ValueError(f"Unknown conv_order: {self.conv_order}. Expected 'cons-vars' or 'vars-cons'.")
            
        return constraint_features, variable_features, prev_cons_c, prev_var_c


    # formula 35 in overleaf
    def forward(self, prev_cons_features, prev_var_features, prev_cons_c, prev_var_c, batch):
        if self.anticipate_mp:
            constraint_features, variable_features, prev_cons_c, prev_var_c = self.forward_with_anticipation(
                prev_cons_features, prev_var_features, prev_cons_c, prev_var_c, batch
            )
        else:
            constraint_features = self.conv_v_to_c(prev_cons_features, prev_var_features, batch)
            variable_features = self.conv_c_to_v(prev_cons_features, prev_var_features, batch)
            
        
        return constraint_features, variable_features, prev_cons_c, prev_var_c
