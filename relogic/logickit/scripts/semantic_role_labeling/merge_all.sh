# python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
# --gold_data_path data/raw_data/srl/json/conll05/origin/train.json \
# --pred_data data/raw_data/srl/json/conll05/origin/train.json \
# --output_data_path data/raw_data/srl/json/conll05/auto_augmented/train.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll05/origin/dev.json \
--pred_data saves/semantic_role_labeling/conll05_large_boundary_1/valid_dump.json \
--output_data_path data/raw_data/srl/json/conll05/auto_augmented/dev.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll05/origin/test.json \
--pred_data saves/semantic_role_labeling/conll05_large_boundary_1/test_dump.json \
--output_data_path data/raw_data/srl/json/conll05/auto_augmented/test.json

# python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
# --gold_data_path data/raw_data/srl/json/conll12/origin/train.json \
# --pred_data data/raw_data/srl/json/conll12/origin/train.json \
# --output_data_path data/raw_data/srl/json/conll12/auto_augmented/train.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll12/origin/dev.json \
--pred_data saves/semantic_role_labeling/conll12_large_boundary_1/valid_dump.json \
--output_data_path data/raw_data/srl/json/conll12/auto_augmented/dev.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll12/origin/test.json \
--pred_data saves/semantic_role_labeling/conll12_large_boundary_1/test_dump.json \
--output_data_path data/raw_data/srl/json/conll12/auto_augmented/test.json