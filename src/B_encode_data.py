import argparse
from itertools import product
import os
from os.path import join
import pandas as pd

formulae_dir = "formulae"

from utils.misc import (
        check_output_folder,
        str2bool, 
        Timer
        )

from data.data_generation import (
    encode_files
)

def get_folder_name(rf_p, rf_n):
    return f"rf-{rf_p}-n-{rf_n}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder for generated data")
    parser.add_argument("--random_features_p", nargs="+", type=float, default=[0.0,0.5,1.0], help="Probability of adding random features")
    parser.add_argument("--random_features_n", nargs="+", type=int, default=1, help="Number of random features to add")
    parser.add_argument("--seed", type=int, default=2342, help="Random seed for reproducibility")
    parser.add_argument("--n_procs", type=int, default=1, help="Number of processes to use for data generation")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing data")
    
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = join(args.input_dir, "processed")

    print("!!! Existing folders with previously computed RF are skipped.")

    # DO NOT check if output_dir exists.
    # In this way, we can add RF features without being forced to re-generate the existing encoded datasets. 

    

    n_files = len([fn for fn in os.listdir(join(args.input_dir, formulae_dir)) if fn.endswith(".cnf")])
    print("|files to encode|= ", n_files)
    print("number of rancom features ratio: ", len(args.random_features_p))
    df = pd.read_csv(join(args.input_dir, "dataset.csv"))
    assert df.shape[0] == n_files, "Number of files and number of rows in dataset.csv do not match"
    
    with Timer("Data encoding"):
        for rf_p in args.random_features_p:
            print(f"Encoding with rf_p: {rf_p}, rf_n: {args.random_features_n}")
            param_folder = get_folder_name(rf_p, args.random_features_n)
            out_fld = join(args.output_dir, param_folder)
            if not args.overwrite and os.path.exists(out_fld):
                print(f"!!! SKIPPING rf_p: {rf_p}, rf_n: {args.random_features_n}. Output folder {out_fld} already exists...")
                continue
            print(out_fld)
            os.makedirs(out_fld, exist_ok=True)
            #df, output_random_features_p, n_random_features, seed, n_procs=1):
            encode_files(df,
                         out_fld, 
                         rf_p, 
                         args.random_features_n, 
                         seed=args.seed, 
                         n_procs=args.n_procs, 
                         )
    print("all done.")
