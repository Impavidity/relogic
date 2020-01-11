# relogic

[![Build Status](https://travis-ci.org/Impavidity/relogic.svg?branch=master)](https://travis-ci.org/Impavidity/relogic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Work Flow

If you want to understand how relogic works, you can start from the [`relogic/main.py`](relogic/main.py),
which is the entry of the training/evaluation process.

If we start the model training, a [`Trainer`](relogic/logickit/training/trainer.py) object will be created.
In the `Trainer`, `Task` objects will be initialized based on the argument `--task_names`.

For example, if we want to create a new task for IR (Information Retrieval), we need to do the following steps.

### Data Processing

- **Implement Dataflow.** You need to define the [`Example`](relogic/logickit/dataflow/pointwise.py), 
[`Feature`](relogic/logickit/dataflow/pointwise.py), 
[`Batch`]((relogic/logickit/dataflow/pointwise.py)) and [`Dataflow`](relogic/logickit/dataflow/pointwise.py) for your task.
Here are several examples in [`relogic/logickit/dataflow/`](relogic/logickit/dataflow). After you implement the dataflow,
you can implement test for the class -- you can refer to [`tests/dataflow/pointwise_test`](tests/dataflow/pointwise_test.py).

### Task definition

- **Define the task type.** Currently we categorize the tasks into three types: classification, span_extraction, and tagging.
For IR task, it can be categorized as classification problem, because we can use cross-entropy as loss function to
train the model, and directly use the probability of softmax as ranking score. For NER task, it can be categorized as tagging
problem.

- **Implement module for the task.** Assume that we already obtain the contextual representation from encoder, the next
step you need to do is to implement a task-specific module. Basically this module takes the contextual representation and
some other arguments and do some magic, and then return the logits/final representation.
 
- **Add your task.** You need to give a name for your task. After you implement this module,
you just add this module under `get_module` function in, for example, 
[`classification task`](relogic/logickit/tasks/classification.py). Also, remember to add your task in function
[`get_task`](relogic/logickit/tasks/__init__.py). Also, because slow process for migrating the code base, you also need
to add your task in [`get_dataset`](relogic/logickit/dataset/labeled_data_loader.py) function.

### Loss Function
- **Implement the loss function.**

### Evaluation
- **Implement the scorer for your task.** 
- **Add your scorer.**


## Models

There are several models are implemented in this toolkit and the instructions will be presented in this section.

Basically there model are based on contextual encoder such as BERT. For more information, please refer to [Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)

### Named Entity Recognition

### Relation Extraction
### Semantic Role Labeling

The Semantic Role Labeler is trained on CoNLL 2005, CoNLL 2009 and CoNLL 2012. Currently the model of CoNLL 2012 is 
available.

#### How to use

```python
from relogic.pipelines.core import Pipeline

pipeline = Pipeline(
  component_names=["predicate_detection", "srl"],
  component_model_names= {"predicate_detection" : "spacy" ,"srl": "srl-conll12"})

from relogic.structures.sentence import Sentence

sent = Sentence(
  text="The Lexington-Concord Sesquicentennial half dollar is a fifty-cent piece struck by the United States Bureau of the Mint in 1925 as a commemorative coin in honor of the 150th anniversary of the Battles of Lexington and Concord.")

pipeline.execute([sent])
```

You will observe the `srl_labels` in Sentence and their labels sequence matches with the sequence of the predicates, 
which is predicted with spacy pos tagger (we simply regard VERB as predicate).

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

## Adversarial Training

The basic training procedure of adversarial training is as follows

```python
# 5 discriminator update + 1 generator update
if update_discriminator:
  real_encode = network(source_language_sample)
  fake_encode = network(target_language_sample)
  AdvAgent.update(real_encode["output"].detach(), fake_encode["output"].detach())
  # Because we only consider to update discriminator
if update_generator:
  optim.zero_grad()
  real_encode = network(source_language_sample)
  fake_encode = network(target_language_sample)
  adversarial_loss = AdvAgent.gen_loss(real_encode["output"], fake_encode["output"])
  label_loss = cross_entropy_loss(real_encode["output"], gold_label)
  loss = label_loss + adversarial_loss
  loss.backward()
  clip_grad_norm_(network.parameters(), clip)
  optim.step()
```

## Data Exploration

It is recommended to use
```commandline
cat data.json | jq . | less
```
to explore the data file.

## Documentation

- What is the docsting style to follow?
  Refer to https://sphinxcontrib-napoleon.readthedocs.io/en/latest/#
  or https://learn-rst.readthedocs.io/zh_CN/latest/Sphinx简单入门.html
  
- How to generate the documentation?

  ```bash
  cd docs
  make html
  # Open html file for checking
  open _build/html/index.html
  ```

- How to publish the documentation to website?

  I host the pages in another repo instead of another branch to make the code repo clean.

  ```bash
  git clone https://github.com/Impavidity/relogic-docs.git
  ```

  And just copy the generated file in `_build/html` into the repo and commit.

  ```bash
  cp -r relogic/docs/_build/html/* relogic-docs
  cd relogic-docs
  git add *
  git commit -m "Update the pages"
  git push
  ```

  And you can check the website here https://impavidity.github.io/relogic-docs

## Publish the code

- How to publish the code to support `pip install`?

  Refer to https://packaging.python.org/tutorials/packaging-projects/.

  Here is the example to publish the package to test environment.

  ```bash
  # Generage dist directory.
  python setup.py sdist bdist_wheel
  # Distribute the package to test environment.
  python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
  # Install newly uploaded package
  python -m pip install --index-url https://test.pypi.org/simple/ --no-deps relogic
  ```

  To publish to permanent storage

  ```bash
  python -m twine upload dist/*
  pyhton -m pip install relogic
  ```

## Citation
If you use this package, please cite. 
