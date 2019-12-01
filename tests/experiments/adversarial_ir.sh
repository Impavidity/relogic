pushd evals/trec_eval/
tar xvf trec_eval.9.0.4.tar && cd trec_eval.9.0.4 && make
popd

python -u -m relogic.main \
--task_name ir,language_identification_ir \
--mode train \
--output_dir saves/ir/Adversarial_IR/test  \
--encoder_type bert \
--bert_model bert-base-multilingual-cased \
--raw_data_path tests/datasets/MicroBlog/,tests/datasets/LangIdentification \
--label_mapping_path data/preprocessed_data/binary_classification.json,none \
--model_name default \
--no_cuda \
--train_batch_size 3 \
--test_batch_size 3 \
--learning_rate 1e-5 \
--epoch_number 3 \
--eval_dev_every 5 \
--print_every 5 \
--early_stop_at 10 \
--qrels_file_path tests/datasets/MicroBlog/qrels.mb.txt,none \
--fix_embedding \
--config_file configurations/adversarial_ir.json \
--adversarial_training GAN \
--training_scheme adversarial_training \
--training_scheme_file configurations/training_scheme/adversarial_example.json

python -u -m relogic.main \
--mode eval \
--restore_path saves/ir/Adversarial_IR/test \
--local_rank 0

rm -r saves/ir/Adversarial_IR/test
