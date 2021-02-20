## Named Entity Recognition

- OntoNote5

    Follow the instruction [here](http://cemantix.org/data/ontonotes.html)
    and get the scripts [here](https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO/tree/master/conll-formatted-ontonotes-5.0/scripts).
    Because the scripts are executed in python2 environment, so you can create python2 environment with conda, if your default
    python is version 3.

    ```bash
    conda create --name py27 python=2.7
    conda activate py27
    cd /path/to/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/scripts
    bash skeleton2conll.sh -D ~/ontonotes-release-5.0/data/files/data/ ~/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/
    python agg.py
    ```