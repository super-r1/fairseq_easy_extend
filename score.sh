#!/bin/bash

scores=('bert_score' 'sacrebleu' 'bleu' 'chrf' 'meteor' 'wer' 'bleurt')

for score in "${scores[@]}"; do
    python decode.py /content/drive/MyDrive/NLP2-2023-ET/iwslt14.tokenized.de-en --source-lang de --target-lang en \
    --path /content/checkpoint/"$1"/checkpoint_best.pt \
    --task translation_lev \
    --iter-decode-max-iter 9 \
    --gen-subset test \
    --print-step \
    --remove-bpe \
    --tokenizer moses \
    --scoring "$score" > /content/checkpoint/"$1"/trained_on_"$1"_scored_with_"$score".txt
done
