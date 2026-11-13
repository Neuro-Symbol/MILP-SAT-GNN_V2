
# 2K, easy
DATASET_DIR=../G4SATBench/import/2000/easy/3-sat
OUTPUT_DIR=../G4SATBench/import/2000/easy/3-sat


python g4satb_import_v2.py \
    --input_dir $DATASET_DIR \
    --output_dir $OUTPUT_DIR \
    --rename yes

N_PROCS_DATA_GEN=1
ENCODE_SEED=2

python3.14 B_encode_data.py \
    --input_dir $DATASET_DIR \
    --seed $ENCODE_SEED \
    --n_procs $N_PROCS_DATA_GEN \
    --random_features_p 0.0 0.5 1.0 \
    --overwrite yes

N_EXAMPLES=2000
SEED_SELECTION=3
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