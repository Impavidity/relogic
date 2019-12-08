pushd evals/trec_eval/
tar xvf trec_eval.9.0.4.tar && cd trec_eval.9.0.4 && make
popd

output_dir=saves/doc_ir/microblog/test

python -u -m relogic.main \
--task_name doc_ir \
--mode train \
--output_dir ${output_dir}  \
--encoder_type bert \
--bert_model bert-base-multilingual-cased \
--raw_data_path tests/datasets/doc_microblog/ \
--label_mapping_path data/preprocessed_data/binary_classification.json \
--model_name default \
--no_cuda \
--train_batch_size 1 \
--test_batch_size 1 \
--learning_rate 1e-5 \
--epoch_number 3 \
--eval_dev_every 5 \
--early_stop_at 20 \
--qrels_file_path tests/datasets/doc_microblog/qrels.doc_mb.txt \
--fix_embedding \
--config_file configurations/mbert_config.json

python -u -m relogic.main \
--mode eval \
--restore_path ${output_dir} \
--no_cuda

rm -r ${output_dir}

