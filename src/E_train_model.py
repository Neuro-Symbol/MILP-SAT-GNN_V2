import argparse
from datetime import datetime
import gc
import humanize as H
import json
import os
from os.path import join
import pandas as pd
import time
import torch
import yaml
import wandb


import learners.common as LC
from learners.common import create_loaders
from learning import train_model

from utils.misc import (
        Timer,
        str2bool,
        check_output_folder,
        set_device,
        configure_determinism,
)




    




def build_output_dir(args):
    # last folder of data_dir
    data_dir = args.data_dir.split("/")[-1]
    # dataset
    dataset= args.data_split.split("/")[-2]
    # split
    split = args.data_split.split("/")[-1]
    #
    emb_size = args.embedding_size
    n_iterations = args.gcnpolicy_n_iterations
    mlp_layers = args.mlp_n_hidden_layers

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_fld = f"e{emb_size}-rl{n_iterations}-mlp{mlp_layers}-{timestamp}"
    return join(args.output_dir, out_fld)




def main(args):    
    model = None
    dev = args.dev_mode

    print(f"ARGS: ", args)  
    
    device = set_device(args.device)
    print(f"Using device: {device} (type {type(device)})")  

    if args.timestamp_output:
        args.output_dir = join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
      
    args.output_dir = build_output_dir(args)

    if args.pretraining == 1:
        pretrained_path = join(args.output_dir, "checkpoints", args.best_model_path)
        pretrained_weights = torch.load(pretrained_path, map_location=device, weights_only=True)

    check_output_folder(args.output_dir, args.overwrite)

    os.makedirs(args.output_dir, exist_ok=True)
    print("Output folder: ", args.output_dir)

    if args.use_wandb:
        wandb.init(project="milp-gnn", config=vars(args), name=args.output_dir.split('/')[-1])

    configure_determinism(args.seed)

    train_loader, val_loader, _ = create_loaders(args.data_split, args.data_dir, args.stratified_batches, args.bs, args.shuffle_seed)
    print("TRAINING size:", len(train_loader.dataset))
    print("VALIDATION size:", len(val_loader.dataset))

    example = next(iter(val_loader))[0]
    n_cons_f = example["left"].x.size(1)
    n_var_f = example["right"].x.size(1)
    print("Dim constraints:", n_cons_f)
    print("Dim variables:", n_var_f)

    # save args as yaml file before training starts
    if args.formula_eval_method == "unsup2" and args.classification_method == "node" and args.loss_function != "unsup2":
        args.loss_function = "unsup2"

    args.n_cons_f = n_cons_f
    args.n_var_f = n_var_f
    print(args)
    args_fn = join(args.output_dir, "args.yaml")
    with open(args_fn, "w") as fout:
        yaml.dump(vars(args), fout, default_flow_style=False)
        print("args saved to:", args_fn)

    # params: emb_size, n_cons_f, n_edge_f, n_var_f, n_layers=2,
    model = LC.build_model(args.model, args, cons_dim=n_cons_f, vars_dim=n_var_f)
    model.print_info()
    
    os.makedirs(join(args.output_dir, "checkpoints"), exist_ok=True)
    best_model_path = join(args.output_dir, "checkpoints", args.best_model_path)
    final_model_path = join(args.output_dir, "checkpoints", args.final_model_path)

    if model.pretraining == 1:
        print(f"\n[!] FINE-TUNING STAGE: Loading graph weights from: {best_model_path}")
        
        filtered_weights = {k: v for k, v in pretrained_weights.items() if not k.startswith("output_module.")}
        model.load_state_dict(filtered_weights, strict=False)
        
        print("[!] Freezing base embeddings and message passing layers...")
        for param in model.cons_embedding.parameters():
            param.requires_grad = False
        for param in model.var_embedding.parameters():
            param.requires_grad = False
        for param in model.bp_layers.parameters():
            param.requires_grad = False
            
        # Ensure the final output_module is active for node training
        for param in model.output_module.parameters():
            param.requires_grad = True
            
        # Filter optimizer to only use active parameters
        active_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print("[!] Ready to fine-tune the node-level output module.\n")
    else:
        active_parameters = model.parameters()

    if args.weight_decay is not None:
        optimizer = torch.optim.AdamW(active_parameters, lr=float(args.lr), weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(active_parameters, lr=float(args.lr))

    
    if args.lr_scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    elif args.lr_scheduler_type == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=float(args.lr)/10, max_lr=float(args.lr), cycle_momentum=False)
    else:
        scheduler = None
    # insert a learning rate scheduler with step size 10 and gamma 0.1
    
   
    t1 = time.perf_counter()
    
    results = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler=scheduler,
        epochs=args.epochs, 
        best_model_path = best_model_path, 
        final_model_path = final_model_path, 
        device = device, 
        model_selection_mode = args.model_selection_mode, 
        model_selection_measure = args.model_selection_measure,
        model_name=args.model,
        loss_function_name=args.loss_function,
        output_dir=args.output_dir,
        classification_method = args.classification_method,
        formula_eval_method = args.formula_eval_method,
        model_selection_threshold=args.model_selection_threshold,
        clip_grad_norm=args.clip_grad_norm
      )  # conf["model"])
    t2 = time.perf_counter()
    results["training_time"] = t2 - t1
    results["training_time_human"] = H.precisedelta(t2 - t1)
    
    model = None
    gc.collect()

    # save train_loss
    # save json file with training results
    results_fn = join(args.output_dir, "training_results.json")
    with open(results_fn, "w") as f:
        json.dump(results, f, indent=4)
    print("training info saved to:", results_fn)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcnpolicy2", choices=["gcnpolicy", "gcnpolicy2"], help="name of the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--data_split", type=str, required=True, help="Path to the train/val split file")
    parser.add_argument("--output_dir", type=str, default="./out", help="Output folder for generated data")
    parser.add_argument("--seed", type=int, default=987, help="Random seed for reproducibility")
    parser.add_argument("--shuffle_seed", type=int, default=2342, help="Random seed for reproducibility")
    parser.add_argument("--stratified_batches", type=str2bool, default=True, help="Use stratified batches")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    #
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--gcnpolicy_n_layers", type=int, default=1, help="Number of GCN layers")
    parser.add_argument("--gcnpolicy_n_iterations", type=int, default=32, help="Number of iterations of the GCN with recurrent cell")
    parser.add_argument("--gcnpolicy_aggregation", type=str, default="sum", choices=["mean", "sum", "max"], help="Aggregation method for GCNPolicy")    
    parser.add_argument("--bip_layer_anticipate_mp", type=str2bool, default=True, help="Anticipate message passing in bipartite layer")
    parser.add_argument("--bip_conv_order", type=str, default="cons-vars", choices=["cons-vars", "var-cons"], help="Order of bipartite convolution")
    parser.add_argument("--bip_conv_append_activation", type=str2bool, default=False, help="Append activation function after bipartite convolution")
    parser.add_argument("--force_bias_value", type=str2bool, default=False, help="Force bias value in GCNPolicy")
    parser.add_argument("--gcn_final_graph_op", type=str, default="mean", choices=["mean", "sum"], help="Final graph operation for GCNPolicy")
    parser.add_argument("--best_model_path", type=str, default="best_model.pt", help="Path to save the best model")
    parser.add_argument("--final_model_path", type=str, default="final_model.pt", help="Path to save the final model")
    #
    parser.add_argument("--model_selection_mode", type=str, default="accuracy_loss", choices=["standard", "accuracy_loss", "GenGap", "AdjScore"], help="Metric to select best model")
    parser.add_argument("--model_selection_measure", type=str, default="accuracy", choices=["accuracy", "loss", "f1"], help="Measure to use for model selection")
    parser.add_argument("--model_selection_threshold", type=float, default=0.05, help="Threshold/Penalty for GenGap and AdjScore model selection modes")
    parser.add_argument("--lr_scheduler_type", type=str, default="none", choices=["none", "CyclicLR", "ReduceLROnPlateau"], help="Type of learning rate scheduler")
    parser.add_argument("--loss_function", type=str, default="bce", choices=["mse", "bce", "focal", "unsup2"], help="Loss function to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--clip_grad_norm", type=float, default=None, help="Maximum gradient norm for clipping (e.g., 1.0). If None, no clipping is applied.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay for optimizer")
    parser.add_argument("--mlp_n_hidden_layers", type=int, default=1, help="Number of hidden layers in MLP")
    parser.add_argument("--output_mlp_n_hidden_layers", type=int, default=1, help="Number of hidden layers in the output MLP")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for MLP layers")
    parser.add_argument("--classification_method", type=str, default="graph", choices=["node", "graph"], help="Graph or node-level classification")
    parser.add_argument("--formula_eval_method", type=str, default="unsup2", choices=["unsup2"], help="Formula evaluation method")
    parser.add_argument("--pretraining", type=int, default=0,  help="Normal training (0) and training after pretraining (1)")
    #
    parser.add_argument("--dev_mode", type=str2bool, default=False, help="Run in dev mode")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing data")
    parser.add_argument("--timestamp_output", type=str2bool, default=False, help="Add timestamp to output folder")
    # 
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Use Weights & Biases")
    #
    parser.add_argument("--do_test", type=str2bool, default=True, help="Test the model after training")

    args, remaining = parser.parse_known_args()

    if args.do_test:
        parser.add_argument("--test_bs", type=int, default=None, help="Batch size for testing")
        parser.add_argument("--test_device", type=str, default=None, help="Device to use for testing")
        parser.add_argument("--gcnpolicy_test_iterations", type=int, default=None, help="Override number of test iterations")


    args = parser.parse_args()
    with Timer("Training"):
        main(args)

    if args.do_test:
        import sys
        from argparse import Namespace

        from F_test_model import main as test_main
        
        test_args = Namespace(
            checkpoint=join(args.output_dir, "checkpoints", args.best_model_path),
            device=args.test_device if args.test_device is not None else args.device,
            bs=args.test_bs if args.test_bs is not None else args.bs,
            dev_mode=args.dev_mode,
            data_dir=args.data_dir,
            test_data_split=args.data_split,
            gcnpolicy_test_iterations=getattr(args, "gcnpolicy_test_iterations", None)
        )
        
        if args.use_wandb:
            test_args.use_wandb = True
        print("starting test")
        test_main(test_args)
        print("test done")
    print("all done")