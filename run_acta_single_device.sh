#!/usr/bin/env bash
#SBATCH --job-name=roberta-rc
#SBATCH -o output/slurm_%A.out

#SBATCH --time=0-36:00:00
#SBATCH --account=debates
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load miniconda
module load gcc/11.3.0
conda activate acta_venv

set -ex

# This scripts runs ACTA train and evaluation in the same process, but only uses
# a single device to avoid inconsistencies when evaluating

INPUT_DIR=./data/relation_data/
OUTPUT_DIR=./output
CHECKPOINT_PATH=checkpoints
TASK_TYPE=rel-class
MODEL=roberta
CACHE_DIR=./cache
EVALUATION_SPLIT=test
EPOCHS=4
BATCH_SIZE=32
MAX_SEQ_LENGTH=64
LEARNING_RATE=4e-5
NUM_DEVICES=1
NUM_WORKERS=-1
LOG_STEPS=500
SAVE_STEPS=500
RANDOM_SEED=42

python ./run_acta.py \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR \
  --task-type $TASK_TYPE \
  --model $MODEL \
  --cache-dir $CACHE_DIR \
  --checkpoint-path $CHECKPOINT_PATH \
  --train \
  --evaluation-split $EVALUATION_SPLIT \
  --validation \
  --num-devices $NUM_DEVICES \
  --num-workers $NUM_WORKERS \
  --epochs $EPOCHS \
  --train-batch-size $BATCH_SIZE \
  --eval-batch-size $BATCH_SIZE \
  --max-seq-length $MAX_SEQ_LENGTH \
  --learning-rate $LEARNING_RATE \
  --lower-case \
  --log-every-n-steps $LOG_STEPS \
  --save-every-n-steps $SAVE_STEPS \
  --eval-all-checkpoints \
  --overwrite-output \
  --random-seed $RANDOM_SEED \
  --weighted-loss

conda deactivate
