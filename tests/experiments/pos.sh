output_dir=saves/ner/conll2006/pos/es/test

python -u -m relogic.main \
--task_name pos \
--mode train \
--output_dir ${output_dir} \
--encoder_type bert \
--bert_model bert-base-multilingual-cased  \
--raw_data_path tests/datasets/conll2006/pos/es/ \
--label_mapping_path data/preprocessed_data/conll0607_pos_label_mapping.json \
--model_name default \
--no_cuda \
--learning_rate 1e-5 \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--eval_dev_every 5 \
--pretokenized \
--metrics accuracy \
--early_stop_at 5 \
--config_file configurations/mbert_config.json

python -u -m relogic.main \
--mode eval \
--restore_path ${output_dir} \
--no_cuda

rm -r ${output_dir}