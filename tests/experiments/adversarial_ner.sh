output_dir=saves/ner/conll2002/en/test

python -u -m relogic.main \
--task_name ner,language_identification_seq \
--mode train \
--output_dir ${output_dir} \
--encoder_type bert \
--bert_model bert-base-multilingual-cased  \
--raw_data_path tests/datasets/conll2002/en/,tests/datasets/language_identification_seq \
--label_mapping_path data/preprocessed_data/ner_BIOES_label_mapping.json,none \
--model_name default \
--no_cuda \
--learning_rate 1e-5 \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--print_every 5 \
--eval_dev_every 5 \
--pretokenized \
--metrics f1 \
--config_file configurations/adversarial_ner.json \
--adversarial_training GAN \
--training_scheme adversarial_training \
--training_scheme_file configurations/training_scheme/adversarial_example_ner.json \
--early_stop_at 20 \
--selected_non_final_layers 8;8

python -u -m relogic.main \
--mode eval \
--restore_path ${output_dir} \
--no_cuda

rm -r ${output_dir}