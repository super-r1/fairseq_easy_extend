#!/bin/bash

criterias=('sacrebleu' 'bleu' 'chrf' 'meteor' 'bleurt' 'comet') 

for criteria in "${criterias[@]}"; do
  python train.py --config-dir "/content/fairseq_easy_extend/fairseq_easy_extend/models/nat/" --config-name "cmlm_config.yaml" \
task.data=/content/drive/MyDrive/NLP2-2023-ET/iwslt14.tokenized.de-en \
checkpoint.restore_file=/content/drive/MyDrive/NLP2-2023-ET/checkpoint_best.pt \
checkpoint.reset_optimizer=True \
optimization.max_update=1 \
checkpoint.save_dir=/content/checkpoint/"$criteria" \
common.log_file=/content/checkpoint/"$criteria"/log_file.log \
criterion.sentence_level_metric=${criteria}
done