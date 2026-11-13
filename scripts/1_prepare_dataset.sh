#!/bin/bash

# DATASET DIR MUST BE SPECIFIED
DATASET_DIR=../MILPEXP/debug/200/easy

SEED_SELECTION=3
N_EXAMPLES=200


python3.14 C_select_subset.py --input_dir $DATASET_DIR \
      --num_examples $N_EXAMPLES \
      --seed $SEED_SELECTION \
      --only_sat yes \
      --overwrite yes

SEED_SPLIT=4

python3.14 D_train_val_test_split.py \
    --dataset_dir $DATASET_DIR/subset-$N_EXAMPLES-seed-$SEED_SELECTION \
    --split_type stratified \
    --stratify_cols label hard \
    --only_sat yes \
    --valid_p 0.1 \
    --test_p 0.1 \
    --split_id 0 \
    --seed $SEED_SPLIT \
    --overwrite yes