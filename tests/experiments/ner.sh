python -u -m relogic.main \
--task_name ner \
--mode train \
--output_dir saves/ner/CoNLL2002/English/test \
--encoder_type xlmr \
--bert_model xlmr.large.v0 \
--raw_data_path tests/datasets/CoNLL2002/English/ \
--label_mapping_path data/preprocessed_data/ner_BIOES_label_mapping.json \
--model_name default \
--local_rank 0 \
--learning_rate 1e-5 \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--eval_dev_every 5 \
--pretokenized \
--metrics f1 \
--early_stop_at 5 \
--trainer_config configurations/xlmr_large_config.json \
--hidden_size 1024

python -u -m relogic.main \
--mode eval \
--restore_path saves/xlmr/CoNLL2002/English/test \
--local_rank 0

rm -r saves/xlmr/CoNLL2002/English/test