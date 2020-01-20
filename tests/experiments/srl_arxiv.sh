#!/usr/bin/env bash

output_dir=saves/srl/conll2012/test3

python -u -m relogic.main \
--task_name pipe_srl \
--mode train \
--output_dir ${output_dir} \
--encoder_type bert \
--bert_model bert-large-cased  \
--raw_data_path tests/datasets/conll2012 \
--label_mapping_path data/preprocessed_data/srl_conll12_BIO_label_mapping.json \
--model_name default \
--local_rank 0 \
--learning_rate 3e-5 \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--eval_dev_every 10 \
--early_stop_at 90 \
--metrics f1 \
--config_file configurations/srl_configuration.json \
--predicate_reveal_method srl_predicate_extra_surface \
--srl_label_format srl_label_seq_based \
--hidden_size 1024


 python -u -m relogic.main \
 --mode eval \
 --restore_path ${output_dir} \
 --no_cuda

 rm -r ${output_dir}