#!/bin/zsh

# DEVICE=cuda
DEVICE=mps

#G4SATBENCH 2K, easy
DATASET_DIR=../G4SATBench/import/2000/easy/3-sat
OUTPUT_DIR=../G4SATBench/import/2000/easy/3-sat/output

#2K, medium
# DATASET_DIR=../G4SATBench/import/2000/medium/3-sat
# OUTPUT_DIR=../G4SATBench/import/2000/medium/3-sat/output

#20K, easy
# DATASET_DIR=../G4SATBench/import/20000/easy/3-sat
# OUTPUT_DIR=../G4SATBench/import/20000/easy/3-sat/output

#20K, medium
# DATASET_DIR=../G4SATBench/import/20000/medium/3-sat
# OUTPUT_DIR=../G4SATBench/import/20000/medium/3-sat/output



# NOTICE: You must specify which encoded dataset to use
python E_train_model.py \
    --model gcnpolicy2 \
    --data_dir $DATASET_DIR/processed/rf-0.0-n-1 \
    --data_split $DATASET_DIR/subset-2000-seed-3/fold_0 \
    --shuffle_seed 111 \
    --output_dir $OUTPUT_DIR \
    --embedding_size 128 \
    --gcnpolicy_n_iterations 32 \
    --epochs 100 \
    --bs 32 \
    --loss_function bce \
    --lr_scheduler_type ReduceLROnPlateau \
    --device $DEVICE \
    --overwrite yes \
    --classification_method node \
    --do_test yes \

# Here you need to specify the full path to the checkpoint
# Please notice that you an evaluate the model on this test partition 
# by setting --do_test yes in the training command. Use F_test_model.py
# to test a trained model on a "different" test partition.

# python F_test_model.py \
#     --model gcnpolicy2 \
#     --checkpoint $OUTPUT_DIR/e128-rl32-mlp1-20261110_142830/checkpoints/best_model.pt \
#     --test_data_split $DATASET_DIR/subset-2000-seed-3/fold_0 \
#     --device $DEVICE \
#     --classification_method node \
