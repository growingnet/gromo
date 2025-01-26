.. _whats_new:

.. currentmodule:: gromo

What's new
==========

.. NOTE: there are 3 separate sections for changes, based on type:
- "Enhancements" for new features
- "Bugs" for bug fixes
- "API changes" for backward-incompatible changes
.. _current:


Develop branch
----------------

Enhancements
~~~~~~~~~~~~

- Add support for Conv2d layers in the sequential case (:gh:`34` by `Théo Rudkiewicz`_)
- Replaced the `assert` statements with `self.assert*` methods in the unit tests (:gh:`50` by `Théo Rudkiewicz`_)
- Reduce unit tests computational load, add parametrized unit tests (:gh:`46` by `Sylvain Chevallier`_)
- Add the possibly to separate S for natural gradient and S for new weights (:gh:`33` by `Théo Rudkiewicz`_)
- Added GPU tracking (:gh:`16` by `Stella Douka`_)
- Added Bayesian Information Criterion for selecting network expansion (:gh:`16` by `Stella Douka`_)
- Unified documentation style (:gh:`14` by `Stella Douka`_)
- Updated Unit Tests (:gh:`14` by `Stella Douka`_)
- Option to disable logging (:gh:`14` by `Stella Douka`_)
- Add CI (:gh:`2` by `Sylvain Chevallier`_)

Bugs
~~~~

- Use a different scaling factor for input and output extensions. In addition, ``apply_change`` and ``extended_forward`` have now compatible behavior in terms of scaling factor. (:gh:`48` by `Théo Rudkiewicz`_)
- Fix the change application when updating the previous layer (:gh:`48` by `Théo Rudkiewicz`_)
- Fix the sub-selection of added neurons in the sequential case (:gh:`41` by `Théo Rudkiewicz`_)
- Correct codecov upload (:gh:`49` by `Sylvain Chevallier`_)
- Fix the data augmentation bug in get_dataset (:gh:`58` by `Stéphane Rivaud`_)
- Fix dataset input_shape: remove the flattening in data augmentation (:gh:`56` by `Stéphane Rivaud`_)

API changes
~~~~~~~~~~~

- None


.. _Sylvain Chevallier: https://github.com/sylvchev
.. _Stella Douka: https://github.com/stelladk
.. _Théo Rudkiewicz: https://github.com/TheoRudkiewicz
.. _Stéphane Rivaud: https://github.com/streethagore
