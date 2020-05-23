#!/usr/bin/env bash


python -u -m relogic.main \
--task_name sql_reranking \
--mode train \
--output_dir saves/semparse/spider/2 \
--bert_model bert-base-uncased \
--raw_data_path ../text2sql-research/data/reranking/bart_deliberate_fine_grain \
--label_mapping_path data/preprocessed_data/binary_classification.json \
--model_name default \
--local_rank 0 \
--learning_rate 1e-3 \
--train_batch_size 16 \
--test_batch_size 16 \
--gradient_accumulation_steps 4 \
--eval_dev_every 300 \
--metrics accuracy \
--early_stop_at 40000 \
--metrics accuracy \
--bert_hidden_size 768 \
--config_file configurations/semparse/bert_sql.json