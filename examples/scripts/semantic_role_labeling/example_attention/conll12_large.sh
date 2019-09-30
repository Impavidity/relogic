python -u -m relogic.main  \
--task_name joint_srl \
--mode train \
--output_dir saves/semantic_role_labeling/conll12_large_joint_$2 \
--bert_model bert-large-cased \
--raw_data_path data/raw_data/srl/json/conll12/tuple_label \
--train_file train.json \
--dev_file dev.json \
--test_file test.json \
--label_mapping_path data/preprocessed_data/srl_conll12_label_mapping.json \
--model_name default \
--local_rank $1 \
--train_batch_size 12 \
--test_batch_size 12 \
--learning_rate 3e-5 \
--epoch_number 4 \
--eval_dev_every 5000 \
--hidden_size 1024 \
--no_bilstm \
--srl_module_type span_gcn \
--span_inference \
--gradient_accumulation_steps 2 \
--adam_learning_rate 0.005 \
--srl_arg_span_repr kenton_lee \
--srl_pred_span_repr kenton_lee \
--two_stage_optim \
# --schedule_method warmup_cosine_warmup_restarts \
# --external_embeddings data/embeddings/wiki-news-300d-1M.npy \
# --use_span_candidates \
