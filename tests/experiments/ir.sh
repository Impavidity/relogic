pushd evals/trec_eval/
tar xvf trec_eval.9.0.4.tar && cd trec_eval.9.0.4 && make
popd

python -u -m relogic.main \
--task_name ir \
--mode train \
--output_dir saves/ir/MicroBlog/test  \
--encoder_type bert \
--bert_model bert-base-multilingual-cased \
--raw_data_path tests/datasets/MicroBlog/ \
--label_mapping_path data/preprocessed_data/binary_classification.json \
--model_name default \
--no_cuda \
--train_batch_size 3 \
--test_batch_size 3 \
--learning_rate 1e-5 \
--epoch_number 3 \
--eval_dev_every 5 \
--early_stop_at 5 \
--qrels_file_path tests/datasets/MicroBlog/qrels.mb.txt \
--fix_embedding \
--config_file configurations/mbert_config.json

python -u -m relogic.main \
--mode eval \
--restore_path saves/ir/MicroBlog/test \
--no_cuda

rm -r saves/ir/MicroBlog/test

