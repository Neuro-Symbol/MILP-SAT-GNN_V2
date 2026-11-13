import torch
import numpy as np
from os.path import join
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.utils.data import Dataset, Sampler
from typing import List 

from data.data_generation import BipartiteHeteroData

class CNFDataset(Dataset):
    def __init__(self, filenames: List[str], data_fld: str):
        self.filenames = filenames
        self.data_fld = data_fld
    

    def __len__(self):
        return len(self.filenames)

# this returns a pytorch geometric object
    def __getitem__(self, idx: int) -> BipartiteHeteroData:
        abs_fn = join(self.data_fld, self.filenames[idx]).replace(".cnf", ".pt")
        
        # read pytorch tensor
        out = torch.load(abs_fn, weights_only=False)
        
        return out

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels: List[float], batch_size:int, shuffle:bool=True, seed:int=None, split_name:str=None):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_name = split_name
        self.pos_indices = np.where(self.labels == 1)[0].tolist()
        self.neg_indices = np.where(self.labels == 0)[0].tolist()

        self.pos_ratio = len(self.pos_indices) / len(self.labels)
        self.neg_ratio = 1 - self.pos_ratio
        self.num_batches = len(self.labels) // batch_size
        self.num_remaining = len(self.labels) % batch_size
        self.pos_per_batch = int(round(self.batch_size * self.pos_ratio))
        self.neg_per_batch = self.batch_size - self.pos_per_batch
        print(f"Sampler <{self.split_name}> (shuffle? {self.shuffle}, seed: {seed}), num batches {self.num_batches}, last batch {self.num_remaining} exs")
        print(f"Sampler <{self.split_name}> Ratios pos: {self.pos_ratio}, neg: {self.neg_ratio} ({self.pos_per_batch}+{self.neg_per_batch})")
        
        self.rng = random.Random(seed) if seed is not None else random

    def __iter__(self):
        # iter called at the beginning of each epoch
        if self.shuffle:
            self.rng.shuffle(self.pos_indices)
            self.rng.shuffle(self.neg_indices)

        pos_iter = iter(self.pos_indices)
        neg_iter = iter(self.neg_indices)

        for _ in range(self.num_batches):
            pos_batch = [next(pos_iter, None) for _ in range(self.pos_per_batch)]
            neg_batch = [next(neg_iter, None) for _ in range(self.neg_per_batch)]
            batch = [i for i in pos_batch + neg_batch if i is not None]
            if self.shuffle:
                self.rng.shuffle(batch)
            yield batch

        # Handle last partial batch
        remaining = [i for i in pos_iter] + [i for i in neg_iter]
        if remaining:
            if self.shuffle:
                self.rng.shuffle(remaining)
            yield remaining

    def __len__(self):
        return self.num_batches + (1 if self.num_remaining > 0 else 0)

def collate_fn(batch_list):
    return Batch.from_data_list(batch_list)    


def create_data_loader(df_split, data_fld, stratified=True, bs=32, shuffle=True, split_name=None, seed=None):
    filenames = df_split["filename"].tolist()
    labels = df_split["label"].tolist()
    
    dataset = CNFDataset(filenames, data_fld)
    if stratified:
        sampler = StratifiedBatchSampler(labels, batch_size=bs, shuffle=shuffle, split_name=split_name, seed=seed)
        dataloader = DataLoader(dataset, 
                                    batch_sampler=sampler, 
                                    collate_fn=collate_fn,
                                    pin_memory=False)
                
    else:
        g = torch.Generator(device="cpu")
        if seed is not None:
            print("Setting seed for standard PyTorch DataLoader:", seed)
            g.manual_seed(seed)
        dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bs, pin_memory=False, generator=g)
    return dataloader
