#!/bin/bash

# This script must be run in the ./src directory

# First, generate a dataset with 50K examples
DIFFICULTY=easy
N_SAMPLES=400
N_PROCS_DATA_GEN=1

DATASET_DIR=../MILPEXP/debug/200/easy
# DATASET_DIR=../G4SATBench/import/2000/easy/3-sat
# DATASET_DIR=../G4SATBench/import/ricca/2000new/medium/3-sat
#DATASET_DIR=../G4SATBench/import/20000/medium/3-sat
# /data-disk/cardillo-data/milp-gnn/dev-data
# milp-data-50k-medium

GEN_SEED=1


python3.14 A_generate_data.py \
    --output_dir $DATASET_DIR \
    --num_samples $N_SAMPLES \
    --k 3 \
    --n_procs $N_PROCS_DATA_GEN \
    --seed $GEN_SEED \
    --fail_with_duplicates yes \
    --overwrite yes \
    --nc_ext 0.01 0.01 \
    --logic bool \
    --n_vars 10 10
    

    # --difficulty $DIFFICULTY \
#<

ENCODE_SEED=2
python3.14 B_encode_data.py \
    --input_dir $DATASET_DIR \
    --seed $ENCODE_SEED \
    --n_procs $N_PROCS_DATA_GEN \
    --random_features_p 0.0 0.5 1.0 \
    --overwrite yes 
#     --output_dir $DATASET_DIR
    


