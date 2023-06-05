.. Copyright 2023 Vincent Jacques

==========
User Guide
==========

Before you read this document, we strongly recommend you to read the :doc:`conceptual overview <conceptual-overview>` as it establishes the bases for this guide.


@todo Write


.. _user-learning-strategies:

Learning strategies
===================

@todo Talk about strategies (from a user perspective) and the structure of command-line options for strategies


.. START other-learnings/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    cp ../command-line-example/{domain.yml,learning-set.csv} .
    cp ../command-line-example/expected-trained-model.yml .
.. STOP

The following examples assume you've followed our :doc:`"Get started" guide <get-started>` and have ``domain.yml`` and ``learning-set.csv`` in your current directory.

.. EXTEND other-learnings/run.sh

If you have a CUDA-compatible GPU and its drivers correctly installed, you can try another strategy to learn the model using it::

    lincs learn classification-model domain.yml learning-set.csv --output-model gpu-trained-model.yml --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu

.. APPEND-TO-LAST-LINE --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43
.. STOP

This should generate exactly the same model as ``trained-model.yml``, but possibly faster.

.. EXTEND other-learnings/run.sh
    diff expected-trained-model.yml gpu-trained-model.yml
.. STOP
