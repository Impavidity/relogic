# relogic

## Introduction

This toolkit is for developing the Natural Language Processing (NLP) pipelines, including model design and development, model deployment and distributed computing.

## Preparation

```bash
# clone the repo.
git clone https://github.com/Impavidity/relogic.git
# create data/raw_data folder, where raw datasets are stored.
mkdir data/raw_data
# create a folder for saving the logs.
mkdir logs
```


## Models

There are several models are implemented in this toolkit and the instructions will be presented in this section.

Basically there model are based on contextual encoder such as BERT. For more information, please refer to [Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)

### Named Entity Recognition

### Relation Extraction
### Semantic Role Labeling
### Cross-lingual Entity Matching over Knowledge Graphs 
### Reading Comprehension
### End to End QA with Reading Comprehension
### Entity Linking

The entity linking model is based on the Wikipedia and Baidu Baike anchors.

#### How to use

```python
from relogic.graphkit.linking.simple_entity_linker import SimpleEntityLinker
linker = SimpleEntityLinker(["en_wikipedia", "zh_baike"])
linker.link("Obama", "en_wikipedia").ranked_uris
linker.link("范冰冰", "zh_baike").ranked_uris
```

## Citation
If you use this package, please cite. 
