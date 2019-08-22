python -u -m relogic.main  \
--task_name joint_srl \
--mode train \
--output_dir saves/semantic_role_labeling/conll05_large_joint_fixed_two_stage_optim_3000_warmup_fix_eval \
--bert_model bert-large-cased \
--raw_data_path data/raw_data/srl/json/conll05/tuple_label \
--train_file train.json \
--dev_file dev.json \
--test_file test.json \
--label_mapping_path data/preprocessed_data/srl_conll05_label_mapping.json \
--model_name default \
--local_rank $1 \
--train_batch_size 12 \
--test_batch_size 12 \
--learning_rate 3e-5 \
--epoch_number 3 \
--eval_dev_every 1000 \
--hidden_size 1024 \
--no_bilstm \
--srl_module_type span_gcn \
--span_inference \
--gradient_accumulation_steps 2 \
--two_stage_optim \
--adam_learning_rate 0.005 \
# --external_embeddings data/embeddings/wiki-news-300d-1M.npy \
# --use_span_candidates \
