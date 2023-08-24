.. Copyright 2023 Vincent Jacques

===========
Get started
===========


Get *lincs*
===========

You have a few options:

- run *lincs* using our Docker image: simple and works on any machine with Docker installed
- install *lincs* using binary wheels: simple and works on fairly recent Linux distributions
- install *lincs* from the source distribution: works on Ubuntu 20.04, and you'll need to install a few dependencies first

We strongly recommend you don't waste your time trying to install *lincs* on another system for now; we do plan to do that soon-ish.
@todo Support other operating systems; distribute wheels.

Get and run the Docker image
----------------------------

.. highlight:: shell

Get the image::

    docker pull jacquev6/lincs

Run the image::

    docker run --rm -it jacquev6/lincs

This will put you in a basic Ubuntu shell with the ``lincs`` command-line interface installed.
You can skip the next sections and go to :ref:`Start using *lincs*' command-line interface <start-command-line>`.

More details about the Docker image: the default tag ``latest`` always points at the latest published version of *lincs*.
`Other tags <https://hub.docker.com/repository/docker/jacquev6/lincs/tags>`_ are available for specific versions, *e.g.* ``jacquev6/lincs:0.3.7``.

Make sure to get familiar with Docker and containers: in particular, all changes you make in the container will be lost when you exit it.
You'll need to use the ``--volume`` option to access your local filesystem from within the container.
See `Docker documentation <https://docs.docker.com/>`_ for more information.

Install *lincs* on your Linux system, using binary wheels
---------------------------------------------------------

``pip install lincs --only-binary lincs`` should be enough.
You can now skip the next section and go to :ref:`Start using *lincs*' command-line interface <start-command-line>`.

Install *lincs* on your Ubuntu 20.04 system, from sources
---------------------------------------------------------

(The same procedure should work on Ubuntu 22.04 as well.)

.. highlight:: shell

.. START install/mandatory-dependencies.sh

First, you need to install a few dependencies::

    # System packages
    sudo apt-get install --yes g++ libboost-python-dev python3-dev libyaml-cpp-dev

    # OR-tools
    wget https://github.com/google/or-tools/releases/download/v8.2/or-tools_ubuntu-20.04_v8.2.8710.tar.gz
    tar xf or-tools_ubuntu-20.04_v8.2.8710.tar.gz
    sudo cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/include/* /usr/local/include
    sudo cp -r or-tools_Ubuntu-20.04-64bit_v8.2.8710/lib/*.so /usr/local/lib
    sudo ldconfig
    rm -r or-tools_Ubuntu-20.04-64bit_v8.2.8710 or-tools_ubuntu-20.04_v8.2.8710.tar.gz

.. STOP

.. START install/optional-dependencies.sh

Optionally, to use GPU-based learning strategies::

    # CUDA (Optional, recommended if you have an NVIDIA GPU)
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    sudo add-apt-repository 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /'
    sudo apt-get update
    sudo apt-get install --yes cuda-cudart-dev-12-1 cuda-nvcc-12-1
    export PATH=/usr/local/cuda-12.1/bin:$PATH

.. STOP

.. START install/Dockerfile-pre
    FROM ubuntu:20.04

    RUN apt-get update

    RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes \
          sudo wget python3-pip dirmngr gpg-agent software-properties-common

    RUN useradd user --create-home
    RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user
    USER user
    ENV PATH=$PATH:/home/user/.local/bin
    WORKDIR /home/user

.. STOP

.. START install/Dockerfile-post
    WORKDIR /home/user
    # Speed-up build when requirements don't change
    ADD project/requirements.txt .
    RUN pip3 install -r requirements.txt
    ADD --chown=user project /home/user/lincs
    RUN pip3 install ./lincs

.. STOP

.. START install/is-long
.. STOP

.. START install/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    mkdir project
    cp -Lr ../../../{lincs,requirements.txt,setup.py} project
    touch project/README.rst  # No need for the actual readme, so don't bust the Docker cache

    # Transform the dependencies.sh file into a Dockerfile to benefit from the Docker build cache
    (
      cat Dockerfile-pre
      echo
      cat mandatory-dependencies.sh optional-dependencies.sh \
      | grep -v -e '^#' -e '^$' \
      | sed 's/^/RUN /' \
      | sed 's/^RUN cd/WORKDIR/' \
      | sed 's/^RUN export/ENV/'
      echo
      cat Dockerfile-post
    ) >Dockerfile

    sudo docker build . --tag lincs-development--install-with-cuda --quiet >/dev/null
    sudo docker run --rm lincs-development--install-with-cuda lincs --help >/dev/null

    # Transform the dependencies.sh file into a Dockerfile to benefit from the Docker build cache
    (
      cat Dockerfile-pre
      echo
      cat mandatory-dependencies.sh \
      | grep -v -e '^#' -e '^$' \
      | sed 's/^/RUN /' \
      | sed 's/^RUN cd/WORKDIR/' \
      | sed 's/^RUN export/ENV/'
      echo
      cat Dockerfile-post
    ) >Dockerfile

    sudo docker build . --tag lincs-development--install-without-cuda --quiet >/dev/null
    sudo docker run --rm lincs-development--install-without-cuda lincs --help >/dev/null
.. STOP

Finally ``pip install lincs --no-binary lincs`` should finalize the install.


.. _start-command-line:

Start using *lincs*' command-line interface
===========================================

.. START help/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR

    lincs --help >actual-help.txt
    diff expected-help.txt actual-help.txt
.. STOP

.. START help/expected-help.txt

.. highlight:: text

The command-line interface is the easiest way to get started with *lincs*, starting with ``lincs --help``, which should output something like::

    Usage: lincs [OPTIONS] COMMAND [ARGS]...

      lincs (Learn and Infer Non-Compensatory Sorting) is a set of tools for
      training and using MCDA models.

    Options:
      --help  Show this message and exit.

    Commands:
      classification-accuracy  Compute a classification accuracy.
      classify                 Classify alternatives.
      generate                 Generate synthetic data.
      learn                    Learn a model.
      visualize                Make graphs from data.

.. STOP

It's organized using sub-commands, the first one being ``generate``, to generate synthetic pseudo-random data.

.. START command-line-example/run.sh
    set -o errexit
    set -o nounset
    set -o pipefail
    trap 'echo "Error on line $LINENO"' ERR
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

Generate a classification problem with 4 criteria and 3 categories (@todo Link to concepts and file formats)::

    lincs generate classification-problem 4 3 --output-problem problem.yml

.. APPEND-TO-LAST-LINE --random-seed 40
.. STOP

.. highlight:: yaml

.. START command-line-example/expected-problem.yml

The generated ``problem.yml`` should look like::

    kind: classification-problem
    format_version: 1
    criteria:
      - name: Criterion 1
        value_type: real
        category_correlation: growing
      - name: Criterion 2
        value_type: real
        category_correlation: growing
      - name: Criterion 3
        value_type: real
        category_correlation: growing
      - name: Criterion 4
        value_type: real
        category_correlation: growing
    categories:
      - name: Category 1
      - name: Category 2
      - name: Category 3

.. STOP

You can edit this file to change the criteria names, the number of categories, *etc.* as long as you keep the same format.

.. EXTEND command-line-example/run.sh
    diff expected-problem.yml problem.yml
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

Then generate an NCS classification model (@todo Link to concepts and file formats)::

    lincs generate classification-model problem.yml --output-model model.yml

.. APPEND-TO-LAST-LINE --random-seed 41
.. STOP

.. highlight:: yaml

.. START command-line-example/expected-model.yml

It should look like::

    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile: [0.2559052, 0.0551739, 0.1622522, 0.05260009]
        sufficient_coalitions: &coalitions
          kind: weights
          criterion_weights: [0.1477713, 0.6186877, 0.4067865, 0.09600859]
      - profile: [0.6769613, 0.3245539, 0.6732799, 0.5985559]
        sufficient_coalitions: *coalitions

.. STOP

Note that *lincs* uses [YAML anchors and references](https://yaml.org/spec/1.2-old/spec.html#id2765878) to avoid repeating the same sufficient coalitions in all profiles.
All ``*coalitions`` means is "use the same value as the ``&coalitions`` anchor".

.. EXTEND command-line-example/run.sh
    diff expected-model.yml model.yml
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

You can visualize it using::

    lincs visualize classification-model problem.yml model.yml model.png

.. STOP

.. EXTEND command-line-example/run.sh
    cp model.png ../../../doc-sources
.. STOP

It should output something like:

.. image:: model.png
    :alt: Model visualization
    :align: center

.. EXTEND command-line-example/run.sh

And finally generate a set of classified alternatives (@todo Link to concepts and file formats)::

    lincs generate classified-alternatives problem.yml model.yml 1000 --output-classified-alternatives learning-set.csv

.. APPEND-TO-LAST-LINE --random-seed 42
.. STOP

@todo Should we provide utilities to split a set of alternatives into a training set and a testing set?
Currently we suggest generating two sets from a synthetic model, but for real-world data it could be useful to split a single set.
Then we'll need to think about the how the ``--max-imbalance`` option interacts with that feature.

.. highlight:: text

.. START command-line-example/expected-learning-set.csv

It should start with something like this, and contain 1000 alternatives::

    name,"Criterion 1","Criterion 2","Criterion 3","Criterion 4",category
    "Alternative 1",0.37454012,0.796543002,0.95071429,0.183434784,"Category 3"
    "Alternative 2",0.731993914,0.779690981,0.598658502,0.596850157,"Category 2"
    "Alternative 3",0.156018645,0.445832759,0.15599452,0.0999749228,"Category 1"
    "Alternative 4",0.0580836125,0.4592489,0.866176128,0.333708614,"Category 3"
    "Alternative 5",0.601114988,0.14286682,0.708072603,0.650888503,"Category 2"

.. STOP

.. EXTEND command-line-example/run.sh
    diff expected-learning-set.csv <(head -n 6 learning-set.csv)
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

You can visualize its first five alternatives using::

    lincs visualize classification-model problem.yml model.yml --alternatives learning-set.csv --alternatives-count 5 alternatives.png

.. STOP

.. EXTEND command-line-example/run.sh
    cp alternatives.png ../../../doc-sources
.. STOP

It should output something like:

.. image:: alternatives.png
    :alt: Alternatives visualization
    :align: center

@todo Remove the legend, place names (categories and alternatives) directly on the graph

You now have a (synthetic) learning set.

.. highlight:: shell

.. EXTEND command-line-example/run.sh

You can use it to train a new model::

    # @todo Rename the command to `train`?
    lincs learn classification-model problem.yml learning-set.csv --output-model trained-model.yml

.. APPEND-TO-LAST-LINE --mrsort.weights-profiles-breed.accuracy-heuristic.random-seed 43
.. STOP

.. highlight:: yaml

.. START command-line-example/expected-trained-model.yml

The trained model has the same structure as the original (synthetic) model because they are both MR-Sort models for the same problem.
The learning set doesn't contain all the information from the original model,
and the trained model was reconstituted from this partial information,
so it is numerically different::

    kind: ncs-classification-model
    format_version: 1
    boundaries:
      - profile: [0.007518337, 0.05495565, 0.1626169, 0.1931279]
        sufficient_coalitions: &coalitions
          kind: weights
          criterion_weights: [0.499999, 0.5, 0.5, 0]
      - profile: [0.03402985, 0.3244802, 0.6724876, 0.4270518]
        sufficient_coalitions: *coalitions

.. STOP

.. EXTEND command-line-example/run.sh
    diff expected-trained-model.yml trained-model.yml
.. STOP

If the training is effective, the resulting trained model should however behave closely to the original one.
To see how close a trained model is to the original one, you can reclassify a testing set.

.. highlight:: shell

.. EXTEND command-line-example/run.sh

First, generate a testing set::

    lincs generate classified-alternatives problem.yml model.yml 10000 --output-classified-alternatives testing-set.csv

.. APPEND-TO-LAST-LINE --random-seed 44
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

And ask the trained model to classify it::

    lincs classify problem.yml trained-model.yml testing-set.csv --output-classified-alternatives reclassified-testing-set.csv

.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

There are a few differences between the original testing set and the reclassified one::

    diff testing-set.csv reclassified-testing-set.csv

.. APPEND-TO-LAST-LINE >classification-diff.txt || true
.. STOP

.. highlight:: diff

.. START command-line-example/expected-classification-diff.txt

That command should show a few alternatives that are not classified the same way by the original and the trained model::

    2595c2595
    < "Alternative 2594",0.234433308,0.780464768,0.162389532,0.622178912,"Category 2"
    ---
    > "Alternative 2594",0.234433308,0.780464768,0.162389532,0.622178912,"Category 1"
    5000c5000
    < "Alternative 4999",0.074135974,0.496049821,0.672853291,0.782560945,"Category 2"
    ---
    > "Alternative 4999",0.074135974,0.496049821,0.672853291,0.782560945,"Category 3"
    5346c5346
    < "Alternative 5345",0.815349102,0.580399215,0.162403136,0.995580792,"Category 2"
    ---
    > "Alternative 5345",0.815349102,0.580399215,0.162403136,0.995580792,"Category 1"
    9639c9639
    < "Alternative 9638",0.939305425,0.0550933145,0.247014269,0.265170485,"Category 1"
    ---
    > "Alternative 9638",0.939305425,0.0550933145,0.247014269,0.265170485,"Category 2"
    9689c9689
    < "Alternative 9688",0.940304875,0.885046899,0.162586793,0.515185535,"Category 2"
    ---
    > "Alternative 9688",0.940304875,0.885046899,0.162586793,0.515185535,"Category 1"
    9934c9934
    < "Alternative 9933",0.705289483,0.11529737,0.162508503,0.0438248962,"Category 2"
    ---
    > "Alternative 9933",0.705289483,0.11529737,0.162508503,0.0438248962,"Category 1"

.. STOP

.. EXTEND command-line-example/run.sh
    diff expected-classification-diff.txt classification-diff.txt
.. STOP

.. highlight:: shell

.. EXTEND command-line-example/run.sh

You can also measure the classification accuracy of the trained model on that testing set::

    lincs classification-accuracy problem.yml trained-model.yml testing-set.csv

.. APPEND-TO-LAST-LINE >classification-accuracy.txt
.. STOP

.. START command-line-example/expected-classification-accuracy.txt

.. highlight:: text

It should be close to 100%::

    9994/10000

.. STOP

.. EXTEND command-line-example/run.sh
    diff expected-classification-accuracy.txt classification-accuracy.txt
.. STOP


What now?
=========

If you haven't done so yet, we recommend you now read our :doc:`conceptual overview documentation <conceptual-overview>`.

Keep in mind that we've only demonstrated the default learning strategy in this guide.
This package implements several strategies accessible via options of ``lincs learn``.
See the :ref:`learning strategies documentation <user-learning-strategies>` in our user guide for more details.

Once you're comfortable with the concepts and tooling, you can use a learning set based on real-world data and train a model that you can use to classify new real-world alternatives.
