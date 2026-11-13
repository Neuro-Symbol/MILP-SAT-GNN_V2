import argparse
import os
import math
import time
import humanize as H
import time
import torch


def check_output_folder(output_dir, overwrite):
    if os.path.exists(output_dir):
        if not overwrite:
            print(f"!!! Output directory {output_dir} already exists. Use --overwrite to overwrite it.")
            exit()
        else:
            print(f"Output directory {output_dir} already exists. Deleting content...")
            clean_folder(output_dir)

def clean_folder(folder):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(folder)

def get_n_procs(n_procs):
    avail = os.cpu_count()
    n_procs = n_procs if isinstance(n_procs, int) else math.floor(n_procs * avail)
    if n_procs == 0:
        n_procs = avail
    elif n_procs < 0:
        n_procs = avail + n_procs
    return n_procs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not int or float")

def int_or_list_to_list(value):
    try:
        return [ int(value) ]
    except ValueError:
        try:
            return [int(i) for i in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not int or list of ints")

def float_or_list_to_list(value):
    try:
        return [ float(value) ]
    except ValueError:
        try:
            return [float(i) for i in value.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not float or list of floats")



class Timer:
    def __init__(self, msg):
        self.msg = msg
    def __enter__(self):
        self.start = time.perf_counter()
        return self  # allows access to .elapsed later

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"{self.msg} - elapsed time: {H.precisedelta(self.elapsed)}")



def set_device(comp_device):


    if comp_device == "mps" and torch.cuda.is_available():
        print("mps requested, but cuda available - switching to cuda:0")
        comp_device = "cuda:0"
    elif comp_device.startswith("cuda") and torch.backends.mps.is_available():
        print("cuda requested, but mps available - switching to mps")
        comp_device = "mps"
    
    device = torch.device(comp_device)
    
    print("set_device- returning torch device:", device)    
    return device

def configure_determinism(seed, torch_deterministic=True, cudnn_benchmark=False):
    import numpy as np
    import random
    # fix random seed for numpy, torch, random, etc
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    print(f"Random seed set to {seed}")