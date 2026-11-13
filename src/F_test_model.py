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
import wandb

import yaml

from data.data_loader import create_data_loader
from data.data_splitting import (
    load_splits
)


import learners.common as LC
from learning import compute_metrics, test_model

from learners.common import (
    create_loaders
)

from utils.misc import (
        Timer,
        str2bool,
        set_device
)






def main(args):    
    model = None
    dev = args.dev_mode
    
    if hasattr(args, "use_wandb") and args.use_wandb and wandb.run is None:
        wandb.init(project="milp-gnn", config=vars(args), name="test_run")

    device = set_device(args.device)
    print(f"Using device: {device}")

    
    # read training args
    norm_checkpoint = os.path.normpath(os.path.abspath(args.checkpoint))
    print(norm_checkpoint)
    if os.path.isfile(norm_checkpoint):
        norm_checkpoint = os.path.dirname(norm_checkpoint)
    print(norm_checkpoint)
    parent_folder = os.path.dirname(norm_checkpoint)
    print(f"Using checkpoint path: {norm_checkpoint}")

    # The `args.yaml` file is expected to be in the experiment's root directory.
    # The `args.checkpoint` can be a path to the checkpoint file (e.g., .../exp_dir/checkpoints/best_model.pt),
    # the checkpoints directory (.../exp_dir/checkpoints), or the experiment directory itself (.../exp_dir).
    # We need to robustly find the experiment directory.
    path_to_check = norm_checkpoint
    if not os.path.isdir(path_to_check):  # It's a file path (even if it doesn't exist)
        path_to_check = os.path.dirname(path_to_check)

    # At this point, path_to_check is a directory. It could be the experiment dir or the 'checkpoints' dir.
    if os.path.basename(path_to_check).lower() == "checkpoints":
        parent_folder = os.path.dirname(path_to_check)
    else:
        parent_folder = path_to_check
    args_fn = join(parent_folder, "args.yaml")
    print("Looking for args.yaml in: ", args_fn)
    print(f"Looking for args.yaml in: {args_fn}")
    with open(args_fn, "r") as f:
        training_args = argparse.Namespace(**yaml.safe_load(f))

    # print training_args as a nicely formatted dictionary
    training_args_dict = vars(training_args)
    print("---")
    print("Training args:")
    for i, (k, v) in enumerate(training_args_dict.items()):
        print(f"{i+1}\t {k}: {v}")
    print("---")

    print("Model name: ", training_args.model)
    print("n_cons_f: ", training_args.n_cons_f)
    print("n_edge_f: ", training_args.n_var_f)
    print("n hidden layers in MLP:",  training_args.mlp_n_hidden_layers)
    print("n hidden layers in output MLP:", getattr(training_args, 'output_mlp_n_hidden_layers', 1))
    if args.data_dir is None:
        args.data_dir = training_args.data_dir
        print("No directory pointing to data provided. Using the one from training args: ", args.data_dir)

    if args.test_data_split is None:
        print("No test data split provided. Using the one from training args: ", training_args.data_split)
        args.test_data_split = training_args.data_split

    _, _, test_loader = create_loaders(args.test_data_split, args.data_dir, False, args.bs, 1) 

    print("=========================")
    print("------> TEST size:", len(test_loader.dataset))
    print("=========================")


    # params: emb_size, n_cons_f, n_edge_f, n_var_f, n_layers=2,
    model = LC.build_model(training_args.model, training_args, cons_dim=training_args.n_cons_f, vars_dim=training_args.n_var_f)
    # now load the model from args.checkpoint
    print("Loading model from: ", args.checkpoint)
    
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    # load the model state dict
    model.load_state_dict(checkpoint)
    
    if args.gcnpolicy_test_iterations is not None and hasattr(model, 'n_iterations'):
        print(f"Overriding model iterations for testing from {model.n_iterations} to {args.gcnpolicy_test_iterations}")
        model.n_iterations = args.gcnpolicy_test_iterations
        
    model.print_info()

    time1 = time.perf_counter()
    res_df = test_model(model, test_loader, device)
    time2 = time1 = time.perf_counter()

    avg_time_per_ex = (time2 - time1) / len(test_loader.dataset)
    print(f"Average time per example: {avg_time_per_ex:.4f} s")
    res_df["test_t"] = time2 - time1
    res_df["avg_test_t"] = avg_time_per_ex

    out_fn = join(training_args.output_dir, f"test-examples-predictions.csv")
    res_df.to_csv(out_fn, index=False, sep=",")
    print(f"Results saved to {out_fn}")
    
    
    metrics_results, test_cm = compute_metrics(res_df) 
    df_cm = pd.DataFrame(test_cm, columns=["Pred 0", "Pred 1"], index=["0", "1"])
    pp_cm = df_cm.to_markdown()
    print(pp_cm)
    
    print("RESULTS:")
    for k, v in metrics_results.items():
        print(f"\t{k}: {v}")
    if wandb.run:
        wandb.log({f"test/{k}": v for k, v in metrics_results.items()})

    out_fn = join(training_args.output_dir, f"test-confusion-matrix.md")
    with open(out_fn, "w") as f:
        f.write(pp_cm)
    print(f"Saved markdown with test confusion matrix to {out_fn}")

    out_fn = join(training_args.output_dir, f"test-confusion-matrix.csv")
    df_cm.to_csv(out_fn, index=True, sep=",")
    print(f"Saved dataframe with test confusion matrix to {out_fn}")

    out_fn = join(training_args.output_dir, f"test-results.yaml")
    with open(out_fn, "w") as f:
        yaml.dump(metrics_results, f, indent=4)
    print(f"Test results saved to {out_fn}")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcnpolicy", choices=["gcnpolicy", "gcnpolicy2"], help="name of the model")
    parser.add_argument("--n_mlp_hidden_layers", type=int, default=1, help="Number of hidden layers in the MLP")
    parser.add_argument("--output_mlp_n_hidden_layers", type=int, default=1, help="Number of hidden layers in the output MLP")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for MLP layers")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, required=False, default=None, help="Path to the dataset directory")
    parser.add_argument("--test_data_split", type=str, required=False, default=None, help="Path to the train/val split file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    #
    parser.add_argument("--bs", type=int, default=32, help="Batch size")
    #
    parser.add_argument("--dev_mode", type=str2bool, default=False, help="Run in dev mode")
    # 
    parser.add_argument("--remote_log", type=str2bool, default=False, help="Log to remote server")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Use Weights & Biases")
    parser.add_argument("--classification_method", type=str, default="node", choices=["node", "graph"], help="Graph or node-level classification")
    parser.add_argument("--formula_eval_method", type=str, default="unsup2", choices=["unsup2"], help="Formula evaluation method")
    #
    parser.add_argument("--gcnpolicy_iterations", type=int, default=32, help="Number of recurrent GCN layers")
    parser.add_argument("--gcnpolicy_test_iterations", type=int, default=32, help="Override number of test iterations")
    args, remaining = parser.parse_known_args()
    
    args = parser.parse_args()
    with Timer("Testing"):
        main(args)
    print("all done")