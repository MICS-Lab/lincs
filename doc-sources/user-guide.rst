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

    cp ../command-line-example/{problem.yml,learning-set.csv} .
    cp ../command-line-example/expected-trained-model.yml .
.. STOP

The following examples assume you've followed our :doc:`"Get started" guide <get-started>` and have ``problem.yml`` and ``learning-set.csv`` in your current directory.

.. EXTEND other-learnings/run.sh

You can use Alglib's linear programming solver instead of the default GLOP one::

    lincs learn classification-model problem.yml learning-set.csv --output-model alglib-trained-model.yml --mrsort.weights-profiles-breed.linear-program.solver alglib

.. APPEND-TO-LAST-LINE --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43
.. STOP

This should output a similar model, with slight numerical differences.

.. START other-learnings/expected-alglib-trained-model.yml
    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile:
          - 0.00770056946
          - 0.0549556538
          - 0.162616938
          - 0.193127945
        sufficient_coalitions:
          kind: weights
          criterion_weights:
            - 0.0181287061
            - 0.981870294
            - 0.981870294
            - 9.92577656e-13
      - profile:
          - 0.0342072099
          - 0.324480206
          - 0.672487617
          - 0.427051842
        sufficient_coalitions:
          kind: weights
          criterion_weights:
            - 0.0181287061
            - 0.981870294
            - 0.981870294
            - 9.92577656e-13
.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-alglib-trained-model.yml alglib-trained-model.yml
.. STOP

.. EXTEND other-learnings/run.sh

If you have a CUDA-compatible GPU and its drivers correctly installed, you can try another strategy to learn the model using it::

    lincs learn classification-model problem.yml learning-set.csv --output-model gpu-trained-model.yml --mrsort.weights-profiles-breed.accuracy-heuristic.processor gpu

.. APPEND-TO-LAST-LINE --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43
.. STOP

.. EXTEND other-learnings/run.sh
    diff expected-trained-model.yml gpu-trained-model.yml
.. STOP

@todo Put this after the discussion about --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed: This should generate exactly the same model as ``trained-model.yml``, but possibly faster.
