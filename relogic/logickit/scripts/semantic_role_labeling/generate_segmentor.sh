python -m relogic.logickit.scripts.semantic_role_labeling.generate_segmentor \
--json_file data/raw_data/srl/json/conll05/origin/train.json \
--dump_file data/raw_data/srl/json/conll05/segmentation/train.json

python -m relogic.logickit.scripts.semantic_role_labeling.generate_segmentor \
--json_file data/raw_data/srl/json/conll05/origin/dev.json \
--dump_file data/raw_data/srl/json/conll05/segmentation/dev.json

python -m relogic.logickit.scripts.semantic_role_labeling.generate_segmentor \
--json_file data/raw_data/srl/json/conll05/origin/test.json \
--dump_file data/raw_data/srl/json/conll05/segmentation/test.json

python -m relogic.logickit.scripts.semantic_role_labeling.generate_segmentor \
--json_file data/raw_data/srl/json/conll05/origin/brown.json \
--dump_file data/raw_data/srl/json/conll05/segmentation/brown.json