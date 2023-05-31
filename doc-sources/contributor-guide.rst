.. Copyright 2023 Vincent Jacques

=================
Contributor guide
=================

Run ``./run-development-cycle.sh``.

.. Or:
    docker run --rm -it -v $PWD:/wd --workdir /wd lincs-development
    After changes in C++:
        pip install --user --no-build-isolation --editable .
    Then test whatever:
        lincs --help

General design
==============

How-tos
=======

Add a new step in an existing strategy
--------------------------------------

Add a variant of a strategy
---------------------------
