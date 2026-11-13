import argparse
import gc
import humanize as H
import os
from os.path import join
import time
from datasets import Dataset

from utils.misc import (
        check_output_folder, 
        get_n_procs,
        str2bool, int_or_float,
        Timer
        )

from data.data_generation import (
    generate_dataset_v3, 
    solve_SAT_problem2,
    add_is_hard_col
)

def print_stats(df):
    df_by_n_var = df.groupby("n_var").agg(
        unique_n_clauses=("n_clauses", "nunique"),
        min_n_clauses=("n_clauses", "min"),
        max_n_clauses=("n_clauses", "max"),
        unique_n_labels=("label", "nunique")
    )

    df_by_label = df.groupby("label").agg(
        unique_n_clauses=("n_clauses", "nunique"),
        min_n_clauses=("n_clauses", "min"),
        max_n_clauses=("n_clauses", "max"),
        unique_n_var=("n_var", "nunique"),
        min_n_vars=("n_var", "min"),
        max_n_vars=("n_var", "max")
    )
    print(df_by_label)
    print(df_by_n_var)

    print("data grouped by label:")
    print(df.groupby("label").size())
    print()
    print("data grouped by hard:")
    print(df.groupby("hard").size())
    print()
    print("data grouped by label and hard:")
    print(df.groupby(["label", "hard"]).size())
    print()
    print("distribution")
    print(df[["label", "hard"]].value_counts(normalize=True))




def generate(args):
    # process difficulty and n_vars
    if args.n_vars is None:
        if args.difficulty is None:
            args.difficulty = "easy"
        if args.difficulty == "easy":
            args.n_vars = (10, 40)
        elif args.difficulty == "medium":
            args.n_vars = (40, 200)

    
    # 1-- generate formulae
    out_fld = join(args.output_dir, "formulae")
    os.makedirs(out_fld, exist_ok=True)
    df = generate_dataset_v3(
        size = args.num_samples,
        data_seed=args.seed,
        min_n_var=args.n_vars[0],
        max_n_var=args.n_vars[1],
        nc_left=args.nc_ext[0],
        nc_right=args.nc_ext[1],
        k=args.k,
        output_dir=out_fld,
        n_procs=args.n_procs,
        fail_with_duplicates=args.fail_with_duplicates
    )
    return df

def main(args):
    check_output_folder(args.output_dir, args.overwrite)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("*********************")
    print("ARGS", args)
    print("*********************")
    

    # 1-- generate
    with Timer("Data generation"):
        df = generate(args)
    
    print(f"generated dataset, shape: {df.shape}")
    if args.difficulty is not None:
        print(f"dataset difficulty: {args.difficulty}")
    print(f"min_n_var: {args.n_vars[0]}")
    print(f"max_n_var: {args.n_vars[1]}")
    print(f"unique number of vars in generated dataset: {len(df.n_var.unique())}")
    print(f"min / max n_clauses in generated dataset: {df.n_clauses.min()} / {df.n_clauses.max()}")

    # SAVING TEMPORARY CSV
    temp_csv_fn = join(args.output_dir, "temp_dataset.csv")
    df.to_csv(temp_csv_fn, index=False, sep=",")
    
    n_actual_procs = get_n_procs(args.n_procs)
    # 2-- solve
    with Timer("Solving"):
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda row: solve_SAT_problem2(row), batched=True, num_proc=n_actual_procs, desc="solving")

    df = dataset.to_pandas()
    df = add_is_hard_col(df)

    # 3 -- print stats
    print_stats(df)

    # 4 -- save
    with Timer("Saving csv+pickle"):
        out_fn = os.path.join(args.output_dir, "dataset.csv")
        df.to_csv(out_fn, index=False, sep=",")
        print(f"dataset saved to {out_fn}")
        out_fn = out_fn.replace(".csv", ".pkl")
        df.to_pickle(out_fn)
        print(f"dataset saved to {out_fn}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for generated data")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--k", type=int, default=3, help="Literales per clauses (k-sat)")
    parser.add_argument("--n_procs", type=int, default=1, help="Number of processes to use for data generation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for data generation") 
    parser.add_argument("--fail_with_duplicates", type=str2bool, default=True, help="Fail if duplicate formulas are generated")
    parser.add_argument("--overwrite", type=str2bool, default=False, help="Overwrite existing data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium"], help="Difficulty level of the data")
    parser.add_argument("--nc_ext", nargs=2, type=int_or_float, metavar=("LEFT", "RIGHT"), default=(0.01, 0.01), help="Extend the computed range of hard clauses")
    group.add_argument("--n_vars", nargs=2, type=int, metavar=("MIN", "MAX"), default=None, help="Specify number of variables as two integers for the range [MIN, MAX]")
    args = parser.parse_args()

    main(args)
    print("all done.")
