python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll05/origin/train.json \
--pred_data data/raw_data/srl/json/conll05/origin/train.json \
--output_data_path data/raw_data/srl/json/conll05/augmented/train.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll05/origin/dev.json \
--pred_data data/raw_data/srl/json/conll05/origin/dev.json \
--output_data_path data/raw_data/srl/json/conll05/augmented/dev.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll05/origin/test.json \
--pred_data data/raw_data/srl/json/conll05/origin/test.json \
--output_data_path data/raw_data/srl/json/conll05/augmented/test.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll12/origin/train.json \
--pred_data data/raw_data/srl/json/conll12/origin/train.json \
--output_data_path data/raw_data/srl/json/conll12/augmented/train.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll12/origin/dev.json \
--pred_data data/raw_data/srl/json/conll12/origin/dev.json \
--output_data_path data/raw_data/srl/json/conll12/augmented/dev.json

python -m relogic.logickit.scripts.semantic_role_labeling.merge_data \
--gold_data_path data/raw_data/srl/json/conll12/origin/test.json \
--pred_data data/raw_data/srl/json/conll12/origin/test.json \
--output_data_path data/raw_data/srl/json/conll12/augmented/test.json