Configure Tasks
===============

Training
-------------------

.. code:: bash

    python -m relogic.main --mode train

Testing
-------------------

.. code:: bash

    python -m relogic.main --mode eval

Data
--------------------

Two main components of the data: label mapping and examples.

Label mapping is a pickle file that contains a dictionary mapping from label string to int id, starting from 0.
Examples are stored in jsonline format.


