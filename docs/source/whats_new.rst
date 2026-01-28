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

- Added documentation linting in CI/CD and reduced warnings in tests (:gh:`158` by `Stella Douka`_)
- New tutorial for `GrowingContainer` (:gh:`188` by `Théo Rudkiewicz`_)
- Update `GrowingBlock` to include recently added features in `GrowingModule` such as `in_neurons` property, `target_in_neurons` parameter, and methods for multi-step growth processes (:gh:`186` by `Théo Rudkiewicz`_)
- Add `in_neurons` property and `target_in_neurons` parameter to `GrowingModule`, `LinearGrowingModule`, and `Conv2dGrowingModule` for tracking neuron counts during growth. Add `missing_neurons`, `number_of_neurons_to_add`, and `complete_growth` methods to simplify multi-step growth processes (:gh:`187` by `Théo Rudkiewicz`_)
- Add new normalization methods (:gh:`185` by `Théo Rudkiewicz`_)
- Update `output_volume` in `Conv2dMergeGrowingModule` based on post_merge_function and reshaping (:gh:`177` by `Stella Douka`_)
- Implement lazy loading datasets that read directly from the disk (:gh:`169` by `Stella Douka`_)
- Modify `in_channels` and `out_channels` as properties in `Conv2dGrowingModule` (:gh:`174` by `Stella Douka`_)
- Introduce a `SequentialGrowingContainer` structure specialized for container with sequential layers.  Introduce a `ResNetBasicBlock` class to create resnet 18/34 like structure with growable blocks and the possibility of adding blocks. (:gh:`168` by `Théo Rudkiewicz`_)
- Allow to create layer extension with different simple initialization (different random and zero). (:gh:`165` by `Théo Rudkiewicz`_)
- Add `TensorStatisticWithEstimationError` and corresponding class `TestTensorStatisticWithEstimationError`. It computes an estimation of the quadratic error done when estimating the given tensor statistic. Modify `TensorStatistic` so that there is no need to call init (:gh:`149` by `Félix Houdouin`_).
- Add a method `normalize_optimal_updates` in `GrowingModule` to normalize the optimal weight updates before applying them (:gh:`164` by `Théo Rudkiewicz`_)
- Add setter for scaling factor in `GrowingModule` (:gh:`157` by `Stella Douka`_)
- Minor improvements of `GrowingContainer` (:gh:`161` by `Théo Rudkiewicz`_)
- Add `in_features` and `out_features` properties to `GrowingModule` and `LinearGrowingModule` (:gh:`160` by `Théo Rudkiewicz`_)
- Add support for convolutional DAGs in `GrowingDAG` and `GrowingGraphNetwork` (:gh:`148` by `Stella Douka`_)
- Handle previous and next layers when deleting `GrowingModule` and `MergeGrowingModule` objects (:gh:`148` by `Stella Douka`_)
- Add `weights_statistics` method to `GrowingModule` and `GrowingContainer` to retrieve statistics of weights in all growing layers. (:gh:`152` by `Théo Rudkiewicz`_)
- Add `ruff` linter to pre-commit hooks and to the CI (:gh:`151` by `Théo Rudkiewicz`_)
- Add `GrowingBlock` to mimic a ResNet 18/34 block. (:gh:`106` by `Théo Rudkiewicz`_)
- fix(RestrictedConv2dGrowingModule.bordered_unfolded_extended_prev_input): Use the correct input size to compute the border effect of the convolution. (:gh:`147` by `Théo Rudkiewicz`_)
- Create a `input_size` property in GrowingModule. (:gh:`143` by `Théo Rudkiewicz`_)
- Improve `GrowingContainer` to allow `GrowingContainer` as submodules (:gh:`133` by `Théo Rudkiewicz`_ and `Stella Douka`_).
- Fix sign errors in `compute_optimal_added_parameters` when using `tensor_m_prev` and in `tensor_n` computation. Add unit tests to cover these cases (:gh:`118` and :gh:`115` by `Théo Rudkiewicz`_).
- Estimate dependencies between activities for faster expansion (:gh:`100` by `Stella Douka`_)
- Makes flattening of input optional in GrowingMLP. Default value is True for backward compatibility (:gh:`108` by `Stéphane Rivaud`_).
- Add the option to handle post layer function that need to grow like BatchNorm (:gh:`105` by `Théo Rudkiewicz`_).
- Add robust `compute_optimal_delta` function to `gromo.utils.tools` with comprehensive dtype handling, automatic LinAlgError fallback to pseudo-inverse, float64 retry mechanism for negative decrease scenarios, and extensive test suite achieving 95% coverage. Function computes optimal weight updates for layer expansion using mathematical formula dW* = M S⁻¹ with full backward compatibility (:gh:`114` by `Stéphane Rivaud`_)
- Refactor `GrowingModule` to centralize `tensor_s_growth` handling (:gh:`109` by `Stéphane Rivaud`_)
- Add `use_projected_gradient` parameter to growing modules to control whether to use projected gradient (`tensor_n`) or raw tensor (`tensor_m_prev`) for computing new neurons. This provides more flexibility in the optimization strategy and improves test coverage for critical code paths (:gh:`104` by `Théo Rudkiewicz`_).
- Fix statistics normalization in `LinearGrowingModule` (:gh:`110` by `Stéphane Rivaud`_)
- Implement systematic test coverage improvement initiative achieving 92% → 95% overall coverage through 4-phase strategic enhancement targeting critical modules: utils.py (80% → 96%), tools.py (78% → 98%), and growing_module.py (92% → 94%). Added 27 comprehensive test methods covering multi-device compatibility, error handling paths, mathematical algorithm edge cases, and abstract class testing via concrete implementations (:gh:`113` by `Stéphane Rivaud`_).
- Fix the `tensor_n` computation in `RestrictedConv2dGrowingModule` (:gh:`103` by `Théo Rudkiewicz`_).
- Add `GrowingBatchNorm1d` and `GrowingBatchNorm2d` modules to support batch normalization in growing networks (:gh:`101` by `Théo Rudkiewicz`_).
- Implemented Conv2dMergeGrowingModule and added support for computing number of parameters in Conv2dGrowingModule (:gh:`94` by `Stella Douka`_)
- Optimize RestrictedConv2dGrowingModule to fasten the simulation of the side effect of a convolution (:gh:`99` by `Théo Rudkiewicz`_).
- Split Conv2dGrowingModule into two subclass `FullConv2dGrowingModule`(that does the same as the previous class) and  `RestrictedConv2dGrowingModule` (that compute only the best 1x1 convolution as the second layer at growth time) (:gh:`92` by `Théo Rudkiewicz`_).
- Code factorization of methods `compute_optimal_added_parameters` and `compute_optimal_delta` that are now abstracted in the `GrowingModule` class. (:gh:`87` by `Théo Rudkiewicz`_).
- Stops automatically computing parameter update in `Conv2dGrowingModule.compute_optimal_added_parameters`to be consistent with `LinearGrowingModule.compute_optimal_added_parameters` (:gh:`87` by `Théo Rudkiewicz`_) .
- Adds a generic GrowingContainer to simplify model management along with unit testing. Propagates modifications to models. (:gh:`77` by `Stéphane Rivaud`_)
- Refactor and simplify repo structure (:gh:`72` and :gh:`73` by `Stella Douka`_)
- Simplify global device handling (:gh:`72` by `Stella Douka`_)
- Integrate an MLP Mixer (:gh:`70` by `Stéphane Rivaud`_)
- Integrate a Residual MLP (:gh:`69` by `Stéphane Rivaud`_)
- Option to restrict action space (:gh:`60` by `Stella Douka`_)
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
- Modify LinearGrowingModule to operate on the last dimension of an input tensor with arbitrary shape (:gh:`54` by `Stéphane Rivaud`_)

Bugs
~~~~

- Fix sub-modules that are not registered in pytorch (:gh:`179` by `Stella Douka`_)
- Fix persistent value of input volume (:gh:`174` by `Stella Douka`_)
- Fix memory leak in tensor updates (:gh:`138` by `Stella Douka`_)
- Device handling in GrowingMLP, GrowingMLPMixer, and GrowingResidualMLP (:gh:`129` by `Stella Douka`_)
- Delete leftover activity tensors (:gh:`78` & `100` by `Stella Douka`_)
- Fix inconsistency with torch.empty not creating empty tensors (:gh:`78` by `Stella Douka`_)
- Expansion of existing nodes not executed in GrowingDAG (:gh:`78` by `Stella Douka`_)
- Fix the computation of optimal added neurons without natural gradient step (:gh:`74` by `Stéphane Rivaud`_)
- Fix the data type management for growth related computations. (:gh:`79` by `Stéphane Rivaud`_)
- Revert global state changes, solve test issues (:gh:`70` by `Stella Douka`_)
- Fix the data augmentation bug in get_dataset (:gh:`58` by `Stéphane Rivaud`_)
- Use a different scaling factor for input and output extensions. In addition, ``apply_change`` and ``extended_forward`` have now compatible behavior in terms of scaling factor. (:gh:`48` by `Théo Rudkiewicz`_)
- Fix the change application when updating the previous layer (:gh:`48` by `Théo Rudkiewicz`_)
- Fix the sub-selection of added neurons in the sequential case (:gh:`41` by `Théo Rudkiewicz`_)
- Correct codecov upload (:gh:`49` by `Sylvain Chevallier`_)
- Fix dataset input_shape: remove the flattening in data augmentation (:gh:`56` by `Stéphane Rivaud`_)
- Fix memory leak from issue :gh:`96` (:gh:`97` by `Théo Rudkiewicz`_)

API changes
~~~~~~~~~~~

- Allow growth between two `GrowingDAG` modules (:gh:`148` & :gh:`179` by `Stella Douka`_)
- Apply all candidate expansions on the same `GrowingDAG` without deepcopy (:gh:`148` by `Stella Douka`_)
- Moved `compute_optimal_delta` function from LinearMergeGrowingModuke to MergeGrowingModule (:gh:`94` by `Stella Douka`_)
- Renamed AdditionGrowingModule to MergeGrowingModule for clarity (:gh:`84` by `Stella Douka`_)
- Added support for configuration files that override default class arguments (:gh:`38` by `Stella Douka`_)


.. _Sylvain Chevallier: https://github.com/sylvchev
.. _Stella Douka: https://github.com/stelladk
.. _Théo Rudkiewicz: https://github.com/TheoRudkiewicz
.. _Stéphane Rivaud: https://github.com/streethagore
