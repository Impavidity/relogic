## Semantic Role Labeling

There are several semantic role labeling datasets available. 
They can be divided into two types: span-based SRL and dependency-based SRL.

In our experiment, we use CoNLL 2005, CoNLL 2009 and CoNLL 2012 datasets.

### Data Preprocessing

- CoNLL 2005
  
    ```bash
    bash fetch_and_make_conll05_data.sh /path/to/wsj
    ```
  
- CoNLL 2012
  
    Follow the instruction [here](http://cemantix.org/data/ontonotes.html)
    and get the scripts [here](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/tree/master/conll-formatted-ontonotes-5.0/scripts).
    Because the scripts are executed in python2 environment, so you can create python2 environment with conda, if your default
    python is version 3.

    ```bash
    conda create --name py27 python=2.7
    conda activate py27
    cd /path/to/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/scripts
    bash skeleton2conll.sh -D ~/ontonotes-release-5.0/data/files/data/ ~/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/
    bash make_conll2012_data.sh /path/to/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/
    ```

- CoNLL 2009

    Text and annotation are directly available in the dataset file downloaded from LDC.

### Annotate Wikipedia

bash