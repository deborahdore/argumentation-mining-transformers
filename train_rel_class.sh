#!/usr/bin/env bash

set -ex

TRAIN_FILE="./data/relation_data/train_relations.tsv"
TEST_FILE="./data/relation_data/test_relations.tsv"
VALIDATION_FILE="./data/relation_data/dev_relations.tsv"
OUTPUT_DIR="./output"
TASK_TYPE="rel-class"
MODEL="roberta"
EXPERIMENT_NAME="relation_data"
RUN_NAME="roberta-model-relevant-labels"
LABELS="Attack Support"
RELEVANT_LABELS="Attack Support"
TRAINING_LABELS="Attack Support"

EPOCHS=4
EARLY_STOP=2
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32
GRADIENT_ACCUMULATION=1
MAX_GRAD=1
MAX_SEQ_LENGTH=64
LEARNING_RATE=4e-5
WEIGHT_DECAY=1e-6
WARMUP_STEPS=0
LOG_STEPS=1000
SAVE_STEPS=1500
RANDOM_SEED=42
ACCELERATOR="cpu"
NUM_DEVICES=1
NUM_WORKERS=1

python /scripts/train.py \
  --train-data "$TRAIN_FILE" \
  --validation-data "$VALIDATION_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --task-type "$TASK_TYPE" \
  --model "$MODEL" \
  --experiment-name "$EXPERIMENT_NAME" \
  --run-name "$RUN_NAME" \
  --labels "$LABELS" \
  --num-devices "$NUM_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --early-stopping "$EARLY_STOP" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
  --max-grad-norm "$MAX_GRAD" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --lower-case \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-steps "$WARMUP_STEPS" \
  --weighted-loss \
  --log-every-n-steps "$LOG_STEPS" \
  --save-every-n-steps "$SAVE_STEPS" \
  --random-seed "$RANDOM_SEED" \
  --accelerator "$ACCELERATOR" \
  --training_labels "$TRAINING_LABELS"

python /scripts/eval.py \
  --test-data "$TEST_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --task-type "$TASK_TYPE" \
  --model "$MODEL" \
  --experiment-name "$EXPERIMENT_NAME" \
  --run-name "$RUN_NAME" \
  --eval-all-checkpoints \
  --labels "$LABELS" \
  --relevant-labels "$RELEVANT_LABELS" \
  --num-devices "$NUM_DEVICES" \
  --num-workers "$NUM_WORKERS" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --lower-case \
  --weighted-loss \
  --random-seed "$RANDOM_SEED" \
  --accelerator "$ACCELERATOR"
