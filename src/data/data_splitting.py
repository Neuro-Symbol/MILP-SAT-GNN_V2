
import os
from os.path import join
import gzip
import sys
import pandas as pd
from pulp import *
import os
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch

def pipeline_data_splitting(df,
        val_size,
        test_size,
        stratify_cols,
        seed=None,
        train_only_sat = False):

    
    if train_only_sat:
        print("\n--- Running CUSTOM Split (Train: 100% SAT | Val/Test: 50% SAT Stratified) ---")
        splits = do_train_val_test_split_stratified_custom(df, val_size, test_size, stratify_cols, seed)
    else:
        print("\n--- Running STANDARD STRATIFIED Split ---")
        splits = do_train_val_test_split_stratified(df, val_size=val_size, test_size=test_size, stratify_cols=stratify_cols, seed=seed)

    
    stats_dfs = []
    for k, l in splits.items():
        stat_df = compute_stats(df[df.formula_id.isin(l)])
        stat_df["split"] = k
        # relative size of the split
        stats_dfs.append(stat_df)
    stat_df = pd.concat(stats_dfs)

    # stats on splits
    split_totals = stat_df.groupby("split")["n_examples"].sum()
    total = split_totals.sum()
    split_percentages = (split_totals / total) * 100
    
    stat_df = stat_df.reset_index().set_index(["split", "label", "hard"])
    
    print("stats on the splits")
    print(stat_df)
    splits_df = pd.DataFrame({
        "total": split_totals,
        "percentage": split_percentages
    })
    # splits
    print(splits_df.reindex(["train", "val", "test"]))
    return splits



def select_examples(df, num_examples, hard_ratio=1.0, pos_ratio=0.5, seed=None):
    # 4 x 2 numpy matrix with the values of (label, hard) and the number of examples available in df
    np_mat = np.zeros((2, 2), dtype=int)
    # 0, 0 -> label=0, hard=0
    np_mat[0, 0] = len(df[(df.label == 0) & (df.hard == 0)])
    # 0, 1 -> label=0, hard=1
    np_mat[0, 1] = len(df[(df.label == 0) & (df.hard == 1)])
    # 1, 0 -> label=1, hard=0
    np_mat[1, 0] = len(df[(df.label == 1) & (df.hard == 0)])
    # 1, 1 -> label=1, hard=1
    np_mat[1, 1] = len(df[(df.label == 1) & (df.hard == 1)])
    assert np.sum(np_mat) == len(df), "sum of np_mat != len(df)"
    
    # dataframe for pretty printing
    tmp_df = pd.DataFrame(np_mat, index=["neg", "pos"], columns=["easy", "hard"])
    print("dataset")
    print(tmp_df)
    del tmp_df

    # various checks for the number of examples
    #   the first check does not consider the sat/unsat among the hard examples
    dataset_size = num_examples
    n_req_hard = math.ceil(hard_ratio * dataset_size)
    n_req_easy = num_examples - n_req_hard
    assert n_req_hard <= np.sum(np_mat[:, 1]), f"not enough hard examples, required / available (not considering sat): {n_req_hard} / {np.sum(np_mat[:, 1])}"
    assert n_req_easy <= np.sum(np_mat[:, 0]), f"not enough easy examples, required / available (not considering sat): {n_req_easy} / {np.sum(np_mat[:, 0])}"
    
    # fraction of positive (sat) vs negative (unsat) examples
    pos_p = pos_ratio
    neg_p = 1.0 - pos_p
    print(f"required POS / NEG ratios: {pos_p} / {neg_p}")
    
    n_req_hard_p = math.ceil(n_req_hard * pos_p)
    n_req_hard_n = n_req_hard - n_req_hard_p
    n_req_easy_p = math.ceil(n_req_easy * pos_p)
    n_req_easy_n = n_req_easy - n_req_easy_p
    assert n_req_hard_p <= np_mat[1, 1], f"not enough hard pos examples, required / available: {n_req_hard_p} / {np_mat[1, 1]}"
    assert n_req_hard_n <= np_mat[0, 1], f"not enough hard neg examples, required / available: {n_req_hard_n} / {np_mat[0, 1]}"
    assert n_req_easy_p <= np_mat[1, 0], f"not enough easy pos examples, required / available: {n_req_easy_p} / {np_mat[1, 0]}"
    assert n_req_easy_n <= np_mat[0, 0], f"not enough easy neg examples, required / available: {n_req_easy_n} / {np_mat[0, 0]}"


    # randomly select examples from the easy and hard regions
    rng = np.random.default_rng(seed)
    easy_pos_df = df[(df.label == 1) & (df.hard == 0)].sample(n=n_req_easy_p, random_state=rng)
    easy_neg_df = df[(df.label == 0) & (df.hard == 0)].sample(n=n_req_easy_n, random_state=rng)
    hard_pos_df = df[(df.label == 1) & (df.hard == 1)].sample(n=n_req_hard_p, random_state=rng)
    hard_neg_df = df[(df.label == 0) & (df.hard == 1)].sample(n=n_req_hard_n, random_state=rng)
    # combine then shuffle using data_seed
    df = pd.concat([easy_pos_df, easy_neg_df, hard_pos_df, hard_neg_df]).sample(frac=1, random_state=rng).reset_index(drop=True)
    assert len(df) == dataset_size, f"final dataset size {len(df)} != {dataset_size}"
    return df


def compute_stats(df, group_cols=["label", "hard"]):
    res = df.groupby(group_cols).agg(
        n_examples=("formula_id", "count"),
        n_var_min=("n_var", "min"),
        n_var_max=("n_var", "max"),
        n_clauses_min=("n_clauses", "min"),
        n_clauses_max=("n_clauses", "max"),
    )

    return res

# Custom split for training on only SAT formulas
def do_train_val_test_split_stratified_custom(df, val_size, test_size, stratify_cols, seed=None):
    # Train gets 100% SAT. Val/Test are 50/50 SAT AND stratified by other columns
    n_total = len(df)
    te_size = test_size if isinstance(test_size, int) else math.ceil(n_total * test_size)
    va_size = val_size if isinstance(val_size, int) else math.ceil(n_total * val_size)
    eval_size = te_size + va_size

    
    # Separate SAT and UNSAT
    df_sat = df[df.label == 1].sample(frac=1, random_state=seed)
    df_unsat = df[df.label == 0].sample(frac=1, random_state=(seed + 1) if seed else None)


    # Build the 50/50 Evaluation Pool
    eval_unsat_size = eval_size // 2
    eval_sat_size = eval_size - eval_unsat_size # handles odd numbers
    
    
    eval_pool = pd.concat([df_sat.iloc[:eval_sat_size], df_unsat.iloc[:eval_unsat_size]])
    
    # Train gets all the remaining SAT instances
    train_pool = df_sat.iloc[eval_sat_size:]
    train_ids = train_pool["formula_id"].tolist()

    # Perform STRATIFIED split ON THE EVALUATION POOL ONLY
    group_cols = [c.strip() for c in stratify_cols]
    test_fraction = te_size / eval_size # Relative size within the eval pool
    
    _, test_ids_series = train_test_split(
        eval_pool["formula_id"], 
        test_size=test_fraction, 
        stratify=eval_pool[group_cols], 
        random_state=seed
    )
    
    test_ids = test_ids_series.tolist()
    val_ids = eval_pool[~eval_pool.formula_id.isin(test_ids)]["formula_id"].tolist()
    

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def do_train_val_test_split_stratified(df, val_size, test_size, stratify_cols=["label", "hard"], seed=None):
    
    # split with a stratification on ["label", "hard"] returning the list of formula_ids in the split
    te_size = test_size if isinstance(test_size, int) else math.ceil(len(df) * test_size)
    va_size = val_size if isinstance(val_size, int) else math.ceil(len(df) * val_size)
    tr_size = len(df) - te_size - va_size
    
    print("REQUESTED SIZES OF THE SPLITS")
    print(f"train size: {tr_size}, val size: {va_size}, test size: {te_size}")

    assert te_size + va_size + tr_size == len(df), "split sizes do not match, check your values"

    # set grouping columns: column hard is excluded when hard_ratio=-1 (ignored) or when hard_ratio=1 or hard_ratio=0
    group_cols = [c.strip() for c in stratify_cols]
    
    print(f"stratifying on {len(group_cols)} columns: {group_cols}")

    # train,val / test split
    train_ids, test_ids = train_test_split(df["formula_id"], test_size=te_size, stratify=df[group_cols], random_state=seed)
    # train / val split
    if va_size > 0:
        # See https://stackoverflow.com/questions/79595277/stratification-fails-in-train-test-split
        train_df = df[df.formula_id.isin(train_ids)]
        train_ids, valid_ids = train_test_split(train_df.formula_id, test_size=va_size, stratify=train_df[group_cols], random_state=(seed + 1) % sys.maxsize if seed is not None else None)
    else:
        valid_ids = []

    # stats on train_ids
    
    
    
    train_df = df.loc[df.formula_id.isin(train_ids), "formula_id"]
    val_df = df.loc[df.formula_id.isin(valid_ids), "formula_id"]
    test_df = df.loc[df.formula_id.isin(test_ids), "formula_id"]
    res = {"train": train_df, "val":val_df, "test":test_df}
    return res


import gc
def load_dataset_splits(meta_args, splits=["train", "val", "test"]):
    out_fld = meta_args["output_dir"]
    data_fld = meta_args["dataset_dir"]
    exp_id = meta_args["exp_id"]
    dataset_fn = join(data_fld, f"dataset_{exp_id}.pkl")
    pyg_data_fld = join(data_fld, "pyg")

    dataset_fn = build_enc_dataset_filename(meta_args)
    dataset_fn = join(pyg_data_fld, dataset_fn)
    torch_ds = torch.load(dataset_fn, weights_only=False)
    print("full torch (pyg) dataset read from:", join(pyg_data_fld, dataset_fn))
    print("\t examples:", len(torch_ds))
    for i, (k, v) in enumerate(torch_ds[0].items()):
        print(f"{i+1}) \t{k}: {type(v)}", end="")
        if torch.is_tensor(v):
            print(f", {v.size()}, {v.dtype}, value={v}")
        else:
            print()
   
    torch_ds2 = {int(torch_ds[i].formula_id): torch_ds[i] for i in range(len(torch_ds))}
    print("torch dataset size:", len(torch_ds2))
    out = defaultdict(list)
    for split in splits:
        split_fn = join(out_fld, f"{split}_{exp_id}.txt.gz")
        with gzip.open(split_fn, "rt") as f:
            ids = map(int, f.read().splitlines())
        
        m = [torch_ds2[e] for e in ids]
        print(f"{split} size:", len(m))

        # select rows with formula_id in ids from the hugging face dataset gr_df
        
        out[split] = m
    torch_ds2 = None
    # force collector
    gc.collect()
    # print info on available RAM memory
    
    return out



def load_splits_dataframes(splits_fld):
    filenames = ["train", "val", "test"]
    res = {}
    for fn in filenames:
        split_fn = join(splits_fld, f"{fn}.csv.gz")
        res[fn] = pd.read_csv(split_fn, sep=",")
    return res.values()


def load_splits(splits_fld):
    filenames = ["train", "val", "test"]
    res = {}
    for fn in filenames:
        split_fn = join(splits_fld, f"{fn}.txt.gz")
        with gzip.open(split_fn, "rt") as f:
            ids = map(int, f.read().splitlines())
        res[fn] = list(ids)
    return res.values()




import gc
def load_dataset_splits_g4s(meta_args, splits=["train", "val", "test"]):
    out_fld = meta_args["output_dir"]
    data_fld = meta_args["dataset_dir"]
    exp_id = meta_args["exp_id"]
    pyg_data_fld = join(data_fld, "pyg")

    dataset_fn = meta_args["dataset"]
    torch_ds = torch.load(join(pyg_data_fld, dataset_fn), weights_only=False)
    
    print("full torch (pyg) dataset read from:", join(pyg_data_fld, dataset_fn))
    
    print("\t examples:", len(torch_ds))
    for i, (k, v) in enumerate(torch_ds[0].items()):
        print(f"{i+1}) \t{k}: {type(v)}", end="")
        if torch.is_tensor(v):
            print(f", {v.size()}, {v.dtype}, value={v}")
        else:
            print()
   
    torch_ds2 = {int(torch_ds[i].formula_id): torch_ds[i] for i in range(len(torch_ds))}
    print("torch dataset size:", len(torch_ds2))
    out = defaultdict(list)
    for split in splits:
        split_fn = join(data_fld, f"{split}.txt.gz")
        with gzip.open(split_fn, "rt") as f:
            ids = map(int, f.read().splitlines())
        
        m = [torch_ds2[e] for e in ids]
        labels = defaultdict(int)
        for e in m:
            labels[int(e.label)] += 1
        for k, v in labels.items():
            print(f"{split} label {k}: {v}")
        print(f"{split} size:", len(m))

        # select rows with formula_id in ids from the hugging face dataset gr_df
        
        out[split] = m
    torch_ds2 = None
    # force collector
    gc.collect()
    # print info on available RAM memory
    
    return out



from torch_geometric.data import Dataset, Data

class GraphDataset(Dataset):
    def __init__(self, examples):
        super(GraphDataset).__init__()
        self.examples = examples
            
    def len(self):
        return len(self.examples)
    
    def get(self, idx):
        return self.examples[idx]


def build_enc_dataset_filename(conf):
    n_random_features, random_features_p, random_features_seed = (0, 0, 0) if not conf["add_random_features"] \
        else (conf["n_random_features"], conf["random_features_p"], conf["random_features_seed"])

    filename = f"rf_{conf['add_random_features']}_n{n_random_features}_p{random_features_p}_s{random_features_seed}_pyg_dataset.pt"
    return filename