import torch
import torch.nn as nn
import torch_geometric as pyg
from torchinfo import summary
import learners.modules as MOD

class GCNPolicy2(nn.Module):
    def __init__(self, 
                 emb_size, 
                 n_cons_f, 
                 n_edge_f, 
                 n_var_f, 
                 n_layers=1,
                 n_iterations=0,
                 mlp_n_hidden_layers=1,
                 output_mlp_n_hidden_layers=1,
                 activation_name="leaky_relu", 
                 aggregation="sum",
                 bias=True, 
                 classification_method="graph", 
                 pretraining=0,
                 dev=False, 
                 name=None, 
                 graph_op="sum",
                 bip_layer_anticipate_mp=False,
                 bip_conv_order="cons-var",  
                 bip_conv_append_activation=False,
                 force_bias_value=False,
                 formula_eval_method="unsup2",
                 dropout=0.0
                 ):
        super(GCNPolicy2, self).__init__()

        self.emb_size = emb_size
        self.n_cons_f = n_cons_f
        self.n_edge_f = n_edge_f
        self.n_var_f = n_var_f
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.bias = bias
        self.n_layers = n_layers
        self.n_iterations = n_iterations
        self.mlp_n_hidden_layers = mlp_n_hidden_layers
        self.output_mlp_n_hidden_layers = output_mlp_n_hidden_layers
        self.classification_method = classification_method
        if formula_eval_method != "unsup2":
            raise ValueError(f"Unknown formula evaluation method: {formula_eval_method}")
        self.formula_eval_method = formula_eval_method
        self.pretraining = pretraining
        self.dev = dev
        if self.dev:
            print("----> DEV MODE SET <----")
        self.name = name
        self.graph_op_name = graph_op
        if graph_op == "mean":
            self.graph_level_op = pyg.nn.global_mean_pool
        elif graph_op == "sum":
            self.graph_level_op = pyg.nn.global_add_pool
        else:
            raise ValueError(f"Unknown graph_op: {graph_op}")
        self.bip_layer_anticipate_mp = bip_layer_anticipate_mp
        self.bip_conv_order = bip_conv_order
        self.conv_append_activation = bip_conv_append_activation
        self.force_bias_value = force_bias_value
        self.dropout = dropout
        self.stop_on_solution = True   # stop iterating as soon as the assignment satisfies the formula
        self._build_layers()

    def print_info(self):
        print(20 * "-")
        print("- GCNPolicy2:")
        print("\t - n_cons_f:", self.n_cons_f)
        print("\t - n_edge_f:", self.n_edge_f)
        print("\t - n_var_f:", self.n_var_f)
        print("\t - emb_size:", self.emb_size)
        print("\t - n_layers:", self.n_layers)
        print("\t - n_iterations:", self.n_iterations)
        print("\t - mlp_n_hidden_layers:", self.mlp_n_hidden_layers)
        print("\t - output_mlp_n_hidden_layers:", self.output_mlp_n_hidden_layers)
        print("\t - activation:", self.activation_name)
        print("\t - aggregation:", self.aggregation)
        print("\t - graph_op:", self.graph_op_name)
        print("\t - anticipate_mp:", self.bip_layer_anticipate_mp)
        print("\t - conv_order:", self.bip_conv_order)
        print("\t - classification_method:", self.classification_method)
        print("\t - dropout:", self.dropout)
        print(20 * "-")
        print("** SUMMARY **")
        summary(self)
        
        if len(self.bp_layers) > 0:
            summary(self.bp_layers[0])

    def _build_layers(self):
        self.cons_embedding = nn.Sequential(
            nn.Linear(self.n_cons_f, self.emb_size),
        )

        self.edge_embedding = nn.Identity()

        self.var_embedding = nn.Sequential(
            nn.Linear(self.n_var_f, self.emb_size),
        )

        self.bp_layers = nn.ModuleList([
            MOD.BipartiteLayer(
                self.emb_size, activation_name=self.activation_name, 
                aggregation=self.aggregation, bias=self.bias, dev=self.dev,
                anticipate_mp=self.bip_layer_anticipate_mp,
                conv_order=self.bip_conv_order,
                conv_append_activation=self.conv_append_activation,
                n_mlp_hidden_layers=self.mlp_n_hidden_layers,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
        ])

        if self.n_iterations > 0:
            self.cons_lstm = MOD.LayerNormBasicLSTMCell(self.emb_size, self.emb_size)
            self.var_lstm = MOD.LayerNormBasicLSTMCell(self.emb_size, self.emb_size)

        match self.classification_method:    
            case "graph":
                self.output_module = MOD.build_mlp(input_dim=2 * self.emb_size, 
                                             output_dim=1, 
                                             n_layers=self.output_mlp_n_hidden_layers, 
                                             hidden_dims=self.emb_size, 
                                             activation_name=self.activation_name, 
                                             bias=True, 
                                             dropout=self.dropout)

            case "node":
                self.output_module = MOD.build_mlp(input_dim=3 * self.emb_size, 
                                             output_dim=1, 
                                             n_layers=self.output_mlp_n_hidden_layers, 
                                             hidden_dims=self.emb_size, 
                                             activation_name=self.activation_name, 
                                             bias=True, 
                                             dropout=self.dropout)

            case _:        
                raise ValueError(f"Unknown classification method: {self.classification_method}")

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.bp_layers:
            layer._reset_parameters()
            

        if self.n_iterations > 0:
            self.cons_lstm.reset_parameters()
            self.var_lstm.reset_parameters()

        # calculate gain
        gain = nn.init.calculate_gain(self.activation_name)

        layers_to_init = [self.cons_embedding[0], self.var_embedding[0]]


        match self.classification_method:    
            case "graph": # graph level classification
                layers_to_init.extend([self.output_module[0], self.output_module[2]])
            case "node": # node level classification
                layers_to_init.extend([self.output_module[0], self.output_module[2]])
            case _:        
                raise ValueError(f"Unknown classification method: {self.classification_method}")



        for layer in layers_to_init:
            torch.nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)


    def _node_logits(self, cons_features, var_features, batch):
        # Per-variable logits computed from the CURRENT cons/var states.
        # Same input and same head used at the end of forward() (keep the two in sync).
        mean_cons = self.graph_level_op(cons_features, batch["left"].batch)
        mean_var = self.graph_level_op(var_features, batch["right"].batch)
        x = torch.cat([mean_cons[batch["right"].batch],
                       mean_var[batch["right"].batch],
                       var_features], dim=1)
        return self.output_module(x)

    @torch.no_grad()
    def _count_unsat_clauses(self, var_probs, batch):
        # Number of clauses NOT satisfied, for each formula in the batch,
        # by the binary assignment (prob > 0.5). A formula is solved when its count is 0.
        edge_index = batch["left", "to", "right"].edge_index   # row 0: clause, row 1: variable
        signs = batch["left", "to", "right"].edge_attr.reshape(-1)
        var_true = var_probs.reshape(-1) > 0.5
        v_true = var_true[edge_index[1]]
        lit_true = torch.where(signs > 0, v_true, ~v_true)      # sign > 0: positive literal
        n_cons = batch["left"].x.size(0)
        n_true_lits = torch.zeros(n_cons, device=var_probs.device)
        n_true_lits.index_add_(0, edge_index[0], lit_true.float())
        clause_unsat = (n_true_lits == 0).float()
        unsat_per_formula = torch.zeros(batch.num_graphs, device=var_probs.device)
        unsat_per_formula.index_add_(0, batch["left"].batch, clause_unsat)
        return unsat_per_formula

    def forward(self, batch): # , cons_x, var_x, x_edge, edge_index): 
        # adj matrix E_{i, j}: constraint i to variable j
            
        cons_features = self.cons_embedding(batch["left"].x)
        # n_cons x emb_size
        
        var_features = self.var_embedding(batch["right"].x)
        # n_vars x emb_size
        
        edge_features = self.edge_embedding(batch["left", "to", "right"].edge_attr)
        # n_dedges x 1 (identity)
        
        
        # Per-step bookkeeping (only meaningful when per-variable outputs exist)
        check_steps = self.classification_method == "node" and self.stop_on_solution
        if check_steps:
            solved = torch.zeros(batch.num_graphs, dtype=torch.bool, device=cons_features.device)

        if self.n_iterations > 0:
            cons_h = cons_features
            var_h = var_features
            cons_c = torch.zeros_like(cons_features)
            var_c = torch.zeros_like(var_features)
            
            for it in range(self.n_iterations):
                curr_cons = cons_h
                curr_var = var_h
                prev_cons_h, prev_cons_c, prev_var_h, prev_var_c = cons_h, cons_c, var_h, var_c
                
                for i in range(self.n_layers):
                    curr_cons, curr_var, _, _ = \
                        self.bp_layers[i](curr_cons, curr_var, None, None, batch)
                
                cons_h, cons_c = self.cons_lstm(curr_cons, (cons_h, cons_c))
                var_h, var_c = self.var_lstm(curr_var, (var_h, var_c))

                # --- Per-step interpretation check ---
                if check_steps:
                    # Formulas already solved keep their previous states, so their interpretation
                    # stays frozen while the other formulas of the batch keep iterating.
                    if self.stop_on_solution and solved.any():
                        keep_c = solved[batch["left"].batch].unsqueeze(-1)
                        keep_v = solved[batch["right"].batch].unsqueeze(-1)
                        cons_h = torch.where(keep_c, prev_cons_h, cons_h)
                        cons_c = torch.where(keep_c, prev_cons_c, cons_c)
                        var_h = torch.where(keep_v, prev_var_h, var_h)
                        var_c = torch.where(keep_v, prev_var_c, var_c)

                    # Decode the variables with the output module at THIS step
                    # (no gradient: this is only used for checking/recording; with dropout > 0
                    # in train mode the check is slightly noisy)
                    with torch.no_grad():
                        step_probs = torch.sigmoid(self._node_logits(cons_h, var_h, batch)).reshape(-1)
                        step_unsat = self._count_unsat_clauses(step_probs, batch)
                    solved = solved | (step_unsat == 0)

                    # Stop as soon as every formula in the batch has an interpretation
                    if self.stop_on_solution and bool(solved.all()):
                        break

            cons_features = cons_h
            var_features = var_h
            
        else:
            for i in range(self.n_layers):
                cons_features, var_features, _, _ = \
                    self.bp_layers[i](cons_features, var_features, None, None, batch)
                


            


        match self.classification_method:    
            case "graph": # graph level classification
                # graph level pooling using pytorch geometric
                cons_features = self.graph_level_op(cons_features, batch["left"].batch)
                var_features = self.graph_level_op(var_features, batch = batch["right"].batch)
                out = torch.cat([cons_features, var_features], dim=1)
                out = self.output_module(out)
                return out
            
            case "node": # node level classification
            

                mean_cons_features = self.graph_level_op(cons_features, batch["left"].batch)
                mean_var_features = self.graph_level_op(var_features, batch = batch["right"].batch)

                # estend mean cons feat and mean var feat to match var_features dimension [fancy indexing]
                mean_cons_features_ext = mean_cons_features[batch["right"].batch]
                mean_var_features_ext = mean_var_features[batch["right"].batch]
                
                out = torch.cat([mean_cons_features_ext, mean_var_features_ext, var_features], dim=1) 



                out = self.output_module(out)

                return out
            
            case _:        
                raise ValueError(f"Unknown classification method: {self.classification_method}")




        
        
        return out
