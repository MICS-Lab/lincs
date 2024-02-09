.. Copyright 2023-2024 Vincent Jacques

=======
Roadmap
=======

Here are the changes we consider integrating in the upcoming versions.
This is not set in stone and can change based on the time we can spend working on the project,
external contributions, *etc.*
Items listed in named versions are more likely to happen than items listed in "later versions".

Version 1.2
===========

(planned late 2024)

- Support Python 3.12
- Support single-peaked criteria (where intermediate values are preferred to extreme values)
- Explore the possibility of optimizing the WPB weights on the GPU

Later versions
==============

- Interface with `XMCDA <http://www.decision-deck.org/xmcda/>`_
- Publish the C++ API
- Integrate with `Decision Deck <http://www.decision-deck.org/>`_, by implementing an `XMCDA web service <http://www.decision-deck.org/ws/>`_
- Support Intel Silicon on macOS
- Support ARM processors on Linux and Windows
- Explore the use of neural networks to learn MR-Sort models
- Explore "parsimony" to select significant criteria (a.k.a. features) and discard the others
