export DATA_PATH=/Users/vitou/Workspaces/AizawaLab/scientific_question_generation/data/clean/ai.stackexchange.com

export VOCAB_SOURCE=${DATA_PATH}/vocabs_2689.txt
export VOCAB_TARGET=${DATA_PATH}/vocabs_2689.txt

export TRAIN_SOURCES=${DATA_PATH}/train/sources.txt
export TRAIN_TARGETS=${DATA_PATH}/train/targets.txt

export DEV_SOURCES=${DATA_PATH}/dev/sources.txt
export DEV_TARGETS=${DATA_PATH}/dev/targets.txt

export DEV_TARGETS_REF=${DATA_PATH}/dev/targets.txt
export TRAIN_STEPS=1000000

export PROJECT_DIR=/Users/vitou/Workspaces/AizawaLab/scientific_question_generation

export TMPDIR=${PROJECT_DIR}/tmp
export MODEL_DIR=${TMPDIR}/attention_seq2seq_003
mkdir -p $MODEL_DIR

python3 -m bin.train \
  --config_paths="
      ./example_configs/nmt_small.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 64 \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps 439532
  --output_dir $MODEL_DIR


