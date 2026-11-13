import argparse
import gzip
import os
from os.path import join
import pandas as pd
import sys
from data.data_splitting import (
    select_examples,
    compute_stats
)

from utils.misc import (
        check_output_folder,
        str2bool, int_or_float, int_or_list_to_list,
        Timer
        )

def main(args):
    out_fld = join(args.input_dir, f"subset-{args.num_examples}-seed-{args.seed}")
    check_output_folder(out_fld, args.overwrite)
    os.makedirs(out_fld, exist_ok=True)
    
    df = pd.read_csv(join(args.input_dir, "dataset.csv"), sep=",")
    sel_df = select_examples(df, args.num_examples, args.hard_ratio, args.pos_ratio, seed=args.seed)
    stats_df = compute_stats(sel_df)
    print(f"Selected examples, size: {len(sel_df)} -- stats:")
    print(stats_df)
    print()

    sel_df.to_csv(join(out_fld, "dataset.csv"), index=False)
    print(f"Subset saved to {join(out_fld, 'dataset.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--num_examples", type=int, default=2000, help="Number of examples to use")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing data")
    parser.add_argument("--hard_ratio", type=float, default=1.0, help="Ratio of hard clauses to soft clauses")
    parser.add_argument("--pos_ratio", type=float, default=0.5, help="Ratio of positive to negative examples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    parser.add_argument("--train_only_sat", type=str2bool, default=False, help="Train = 100/100 SAT, Val/Test = 50/100 SAT")
    parser.add_argument("--valid_p", type=float, default=0.1, help="Validation set percentage (e.g. 0.1)")
    parser.add_argument("--test_p", type=float, default=0.1, help="Test set percentage (e.g. 0.1)")

    args = parser.parse_args()
    if args.seed is None:
        args.seed = time.time()

    if args.train_only_sat:
        args.pos_ratio = 1 - (args.valid_p * 0.5) - (args.test_p * 0.5)
    with Timer("Data splitting"):
        main(args)
    print("all done.")