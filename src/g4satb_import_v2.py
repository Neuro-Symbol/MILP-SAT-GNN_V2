import argparse
import gzip
import os
from os.path import join
import pandas as pd
import sys
import torch
from tqdm import tqdm
import re

from data.data_generation import build_dimacs_representation, add_is_hard_col


def rename_files(input_dir):  # , append_sat="no"):
    counter = 0
    for split in ['train', 'test', 'valid']:
        for label in ['sat', 'unsat']:
            dir_path = join(input_dir, split, label)
            files = [f for f in os.listdir(dir_path) if f.endswith('.cnf')]
            for file in files:
                old_file = os.path.join(dir_path, file)
                new_file = os.path.join(dir_path, f"{counter}_{label}.cnf")
                os.rename(old_file, new_file)
                print(f"Renamed {old_file} to {new_file}")
                counter += 1
    print(f"Renamed {counter} files in total.")


def collect_files(input_dir):
    data = {}
    total_files = 0
    for split in ['train', 'test', 'valid']:
        data[split] = {}
        for label in ['sat', 'unsat']:
            dir_path = join(input_dir, split, label)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.cnf')]
            total_files += len(files)
            data[split][label] = sorted(files)
    print(f"Total files: {total_files}")
    return data


def process_split(input_dir, split, sat_files, unsat_files):
    res = []
    label = 1.0
    for file in tqdm(sat_files):
        filename = os.path.basename(file)
        formula_id = int( filename.split(".")[0].split("_")[0])  # filename is e.g. 1_sat.cnf 
        res.append({"formula_id": formula_id, "label": label, "filename": filename, "folder": join(input_dir, split, "sat")})
    label = 0.0
    for file in tqdm(unsat_files):
        filename = os.path.basename(file)
        formula_id = int( filename.split(".")[0].split("_")[0])  # filename is e.g. 1_sat.cnf 
        res.append({"formula_id": formula_id, "label": label, "filename": filename, "folder": join(input_dir, split, "unsat")})
    df = pd.DataFrame.from_records(res)
    print(df.groupby("label").size().to_frame(name="count").reset_index())
    return df


def save_splits(df, output_dir):
    splits = ["train", "valid", "test"]
    for split in splits:
        filenames = []
        formula_ids = df[df.partition == split].formula_id.tolist()
        
        out_txt = "\n".join([str(x) for x in formula_ids])
        name = "val" if split == "valid" else split
        out_fn = os.path.join(output_dir, f"{name}.txt.gz")
        with gzip.open(out_fn, "wt") as f:
            f.write(out_txt)
        print(f"Saved {split} split to {out_fn}")




def prepare_pyg_datasets(df, output_dir, seed=1, n_procs=1, random_features_p=0.5, n_random_features=1):
    output_dir = join(output_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    print("using the same seed for all datasets:", seed)
    
    # no rf
    print(f"nodes with no RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, False, random_features_p=None, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, "pyg_dataset_no_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))

    # rf p=1.0
    print(f"all nodes with RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, True, random_features_p=1.0, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, "pyg_dataset_all_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))
    
    # rf p passed as an argument
    print(f"{random_features_p} nodes with RF - n={n_random_features}")
    pyg_ds = build_dimacs_representation(df, True, random_features_p=random_features_p, n_random_features=n_random_features, n_procs=n_procs, seed=seed)
    out_fn = join(output_dir, f"pyg_dataset_{random_features_p}_rf.pt")
    torch.save(pyg_ds, out_fn)
    print("PyTorch Geometric dataset saved to", out_fn)
    print("PyTorch Geometric dataset length:", len(pyg_ds))

def copy_files(input_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    # copy recursively all files from input_dir to output_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".cnf"):
                src = os.path.join(root, file)
                dst = os.path.join(dst_dir, os.path.relpath(src, input_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(src, "r") as f_in:
                    with open(dst, "w") as f_out:
                        f_out.write(f_in.read())

def build_formula_dataframe(file_dict):
    dfs = []
    base_counter = 0
    records = []
    for split in ["train", "valid", "test"]:
        for label in ["sat", "unsat"]:
            files = file_dict[split][label]
            for file in files:
                filename = os.path.basename(file)
                digits = re.findall(r'\d+', filename)
                formula_id = int(digits[-1]) + base_counter
                records.append({
                    "formula_id": formula_id,
                    "original": file,
                    "label": 1.0 if label == "sat" else 0.0,
                    "partition": split,
                })
            base_counter += len(files)
            print("base counter now is: ", base_counter)
    df = pd.DataFrame.from_records(records)


    # now move all files to "formulae" directory
    def build_formula_filename(row):
        formula_id = row["formula_id"]
        label = "sat" if row["label"] == 1.0 else "unsat"
        return f"{formula_id}_{label}.cnf"
    
    df["filename"] = df.apply(build_formula_filename, axis=1)
    
    return df

def flatten_folders(df, formulae_dir):
    def copy_file(row, out_fld):
        src = row["original"]
        print(src)
        dst = os.path.join(out_fld, row["filename"])
        with open(src, "r") as f_in:
            with open(dst, "w") as f_out:
                f_out.write(f_in.read())
    df.apply(lambda row: copy_file(row, formulae_dir), axis=1)


def main(args):
    # -- 0 --
    # copy all files from the input directory to the output directory
    original_dir = os.path.join(args.output_dir, "original")    
    copy_files(args.input_dir, original_dir)
    
    # -- 1 -- 
    file_dict = collect_files(original_dir)  
    
    # -- 2 --
    df = build_formula_dataframe(file_dict)
    print(df.groupby("partition").sample(1))

    # -- 3 -- move all files to the "formulae" directory
    formulae_dir = os.path.join(args.output_dir, "formulae")
    print("flattening files in: ", formulae_dir)
    os.makedirs(formulae_dir, exist_ok=True)
    flatten_folders(df, formulae_dir)
    df["folder"] = formulae_dir
    print(df.groupby("partition").sample(1))

    
    # -- 4 -- Extract "n_var" and "n_clauses" from the DIMACS header 'p cnf <vars> <clauses>'
    def extract_dimacs_stats(row):
        filepath = os.path.join(row["folder"], row["filename"])
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('p cnf'):
                    parts = line.split()
                    # parts[2] is n_var, parts[3] is n_clauses
                    return int(parts[2]), int(parts[3])
        return 0, 0
    
    print("Extracting n_var and n_clauses from CNF files...")
    df["n_var"], df["n_clauses"] = zip(*df.apply(extract_dimacs_stats, axis=1))

    # 2. Calculate the "Hard Region" using the imported function
    print("Computing hardness region...")
    df = add_is_hard_col(df)
    # -- 2 --
    # save train, valid, test splits
    default_split = join(args.output_dir, "fold_0")
    os.makedirs(default_split, exist_ok=True)
    save_splits(df, default_split)
    

    # -- 3 --
    # produce a dataframe as the one from "full_data_generation" in "src/data/data_generation.py"
    out_fn = os.path.join(args.output_dir, "dataset.csv")
    df.to_csv(out_fn, index=False)
    
    print("Full dataset, length:", len(df))
    print(f"Dataset saved to {out_fn}")

    
    # -- 4 --
    # NOW USE STANDARD ENCODING prepare_pyg_datasets(df, args.output_dir)
    # to pyTorch Geometric
    



# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G4SATBench data import")
    parser.add_argument("--input_dir", type=str, default="~/workspace/G4SATBench/dataset")
    parser.add_argument("--output_dir", type=str, default="../../input/dataset/g4satbench")
    parser.add_argument("--rename", type=str, default="no", choices=["yes", "no"], help="Rename files in the input directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} does not exist")
        sys.exit(1)
    
    main(args)
    print("all done.")
