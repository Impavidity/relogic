python -u -m relogic.main \
--task_name ner \
--mode train \
--output_dir saves/ner/CoNLL2002/English/test \
--encoder_type bert \
--bert_model bert-base-multilingual-cased \
--raw_data_path tests/datasets/CoNLL2002/English/ \
--label_mapping_path data/preprocessed_data/ner_BIOES_label_mapping.json \
--model_name default \
--no_cuda \
--learning_rate 1e-5 \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--eval_dev_every 5 \
--pretokenized \
--metrics f1 \
--early_stop_at 5 \
--trainer_config configurations/mbert_config.json

python -u -m relogic.main \
--mode eval \
--restore_path saves/ner/CoNLL2002/English/test \
--no_cuda

rm -r saves/ner/CoNLL2002/English/test