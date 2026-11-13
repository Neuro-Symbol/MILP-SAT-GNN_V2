import argparse
import gzip
import os
from os.path import join
import pandas as pd

from data.data_splitting import (
    pipeline_data_splitting)

from utils.misc import (
        check_output_folder,
        str2bool, int_or_float, int_or_list_to_list,
        Timer
        )

def main_stratified(args):
    out_fld = join(args.dataset_dir, f"fold_{args.split_id}")
    check_output_folder(out_fld, args.overwrite)
    os.makedirs(out_fld, exist_ok=True)

    df = pd.read_csv(join(args.dataset_dir, "dataset.csv"), sep=",")
    print("dataset shape: ", df.shape)
    splits = pipeline_data_splitting(df=df,
        val_size=args.valid_p,
        test_size=args.test_p,
        stratify_cols=args.stratify_cols,
        seed=args.seed,
        train_only_sat=args.train_only_sat)
    
    # save the splits
    for split, ids in splits.items():
        out_fn = join(out_fld, f"{split}.txt.gz")
        with gzip.open(out_fn, "wt") as f:
            f.write( "\n".join( map(str, ids) ) )
        print(f"{split} ids saved to {out_fn}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--split_type", type=str, default="stratified", choices=["stratified"], help="Type of split to perform")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing data")

    args, remainder = parser.parse_known_args()
    if args.split_type == "stratified":
        parser.add_argument("--split_id", type=int, default=0, help="ID of the split to perform")
        parser.add_argument("--stratify_cols", type=str, nargs="+", default=["label", "hard"], help="Columns to stratify by")
        parser.add_argument("--seed", type=int, default=2342, help="Random seed for reproducibility")
        parser.add_argument("--valid_p", type=float, default=0.1, help="Proportion of data to use for validation")
        parser.add_argument("--test_p", type=float, default=0.1, help="Proportion of data to use for test")

        parser.add_argument("--train_only_sat", type=str2bool, default=False, help="Use 100/100 SAT for train, 50/50 for val/test")
    
    args = parser.parse_args()
    with Timer("Data splitting"):
        main_stratified(args)
    print("all done.")
