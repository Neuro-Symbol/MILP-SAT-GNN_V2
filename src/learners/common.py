import os
from os.path import join
import pandas as pd
import torch.nn as nn

from data.data_splitting import (
    load_splits
)

from data.data_loader import create_data_loader
import learners.models as M


activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(), 
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "silu": nn.SiLU(),
}


def build_gcnpolicy(args, cons_dim=None, vars_dim=None, edges_dim=1):
    model = M.GCNPolicy(
        emb_size = int(args.embedding_size),
        n_cons_f = cons_dim,
        n_var_f = vars_dim,
        n_edge_f = edges_dim,
        n_layers = args.gcnpolicy_n_layers,
        aggregation = args.gcnpolicy_aggregation,
        dev = args.dev_mode,
        bip_layer_anticipate_mp = args.bip_layer_anticipate_mp,
        bip_conv_order = args.bip_conv_order,
        bip_conv_append_activation = args.bip_conv_append_activation,
        force_bias_value = args.force_bias_value
    )
    return model

def build_gcnpolicy_ext(args, cons_dim=None, vars_dim=None, edges_dim=1, mlp_n_hidden_layers=1, output_mlp_n_hidden_layers=1):
    model = M.GCNPolicy2(
        emb_size = int(args.embedding_size),
        n_cons_f = cons_dim,
        n_var_f = vars_dim,
        n_edge_f = edges_dim,
        n_layers = int(args.gcnpolicy_n_layers),
        n_iterations = int(args.gcnpolicy_n_iterations),
        mlp_n_hidden_layers = mlp_n_hidden_layers,
        output_mlp_n_hidden_layers = output_mlp_n_hidden_layers,
        aggregation = args.gcnpolicy_aggregation,
        classification_method = args.classification_method, 
        pretraining = args.pretraining,
        formula_eval_method = args.formula_eval_method,
        dev = args.dev_mode,
        graph_op = args.gcn_final_graph_op,
        bip_layer_anticipate_mp = args.bip_layer_anticipate_mp,
        bip_conv_order = args.bip_conv_order,
        bip_conv_append_activation = args.bip_conv_append_activation,
        force_bias_value = args.force_bias_value
    )
    return model




def build_model(model_name, args, cons_dim=None, vars_dim=None, edges_dim=1):
    model = None
    if model_name == "gcnpolicy":
        model = build_gcnpolicy(args, cons_dim, vars_dim, edges_dim)
    elif model_name == "gcnpolicy2":
        print("Building ext model")
        model = build_gcnpolicy_ext(args, cons_dim, vars_dim, edges_dim, args.mlp_n_hidden_layers, getattr(args, 'output_mlp_n_hidden_layers', 1))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def check_non_overlapping_splits(train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert len(train_set.intersection(val_set)) == 0, "Train and validation sets overlap"
    assert len(train_set.intersection(test_set)) == 0, "Train and test sets overlap"
    assert len(val_set.intersection(test_set)) == 0, "Validation and test sets overlap"
    print("Train, validation, and test sets do not overlap")

# data_split, data_dir, stratified, bs, seed
def create_loaders(data_split, data_dir, train_stratified, bs, seed):
    parent_folder = os.path.dirname(os.path.normpath(data_split))
    df = pd.read_csv(join(parent_folder, "dataset.csv"), sep=",")
    print("dataset shape: ", df.shape)
    train_ids, val_ids, test_ids = load_splits(data_split)
    check_non_overlapping_splits(train_ids, val_ids, test_ids)    

    train_df = df[df["formula_id"].isin(train_ids)]
    val_df = df[df["formula_id"].isin(val_ids)]
    test_df = df[df["formula_id"].isin(test_ids)]
    print("|train|= ", len(train_df))
    print("|val|= ", len(val_df))

    assert len(train_df) == len(train_ids), f"Train df has {len(train_df)} rows, but {len(train_ids)} ids"
    assert len(val_df) == len(val_ids), f"Val df has {len(val_df)} rows, but {len(val_ids)} ids"
    assert len(test_df) == len(test_ids), f"Test df has {len(test_df)} rows, but {len(test_ids)} ids"

    train_dl = create_data_loader(train_df, 
                                  data_dir, 
                                  stratified=train_stratified, 
                                  bs=bs, 
                                  shuffle=True, 
                                  split_name="train", 
                                  seed=seed)
    val_dl = create_data_loader(val_df, 
                                data_dir, 
                                stratified=False, 
                                bs=bs, 
                                shuffle=False, 
                                split_name="val")
    test_dl = create_data_loader(test_df, 
                                data_dir, 
                                stratified=False, 
                                bs=bs, 
                                shuffle=False, 
                                split_name="test")
    return train_dl, val_dl, test_dl
