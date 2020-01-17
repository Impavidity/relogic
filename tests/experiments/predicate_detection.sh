output_dir=saves/pe/conll2012/test

python -u -m relogic.main \
--task_name predicate_detection \
--mode train \
--output_dir ${output_dir} \
--encoder_type lstm \
--bert_model fasttext-en  \
--raw_data_path tests/datasets/predicate_detection_conll2012/ \
--label_mapping_path data/preprocessed_data/predicate_detection_label_mapping.json \
--model_name default \
--hidden_size 400 \
--no_cuda \
--train_batch_size 3 \
--test_batch_size 3 \
--epoch_number 3 \
--eval_dev_every 10 \
--metrics f1 \
--early_stop_at 30 \
--only_adam \
--adam_learning_rate 0.001 \
--config_file configurations/lstm.json

python -u -m relogic.main \
--mode eval \
--restore_path ${output_dir} \
--no_cuda

rm -r ${output_dir}
