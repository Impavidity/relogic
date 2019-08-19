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
