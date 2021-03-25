# Changelog
I should probably have started doing it long ago, but better late than never. So here it is (for older entries see commit history)

## 2021 Mar 25

Forgotten maintainance related commit. Updates to inspectors and train_data interfaces, reworked `nnet::init4fixedBatchFprop()` and it's callers.


## 2021 Feb 17

- fixed both `_i_train_data` implementations to obey `batchIndex` parameter of `on_next_batch()/next_subset()`.
- placeholder `imemmgr` for proper intermediate memory manager is introduced. To be done later.
- logarithmic activation is implemented: `A = {log(A+1)|A>0, 0|A<=0}`
- bug fixes and minor upgrades

## 2021 Feb 07

- bug fixes and some minor updates

## 2021 Feb 05

- weight initialization algorithm now may have a non-static implementaion. Consequently weights creation function is renamed from `init()` to `make_weights()`
- layer's interface `init()` is renamed to `layer_init()` and `deinit()` to `layer_deinit()` (it was bad idea to give these functions too generic names - hard to grep 'em all). Same applies to `_i_grad_works<>::init/deinit` - they are now `gw_init/gw_deinit`.
- `on_batch_size_change()` has changed the signature to get new incoming batch size and return outgoing.
- Now a layer may (technically) change the batch size (rows count of data matrices) during data propagation (allows more flexibility in algorithms). Corresponding `common_nn_data<>` members no longer applies to the whole nnet, just to an input_layer. All nests are passing, however due to huge scale of changes, it is possible that some very rarely used code not covered by tests remained unchanged (I just forgot about its existence) and therefore it won't compile. Changes required are quite straightforward though, so I don't feel it wouldn't be unsolvable anyone who dares to use it.

## 2021 Feb 02

- cleanup of internal memory consumption.
- layers' `is_activations_shared()` no longer relyes on `m_activations.bDontManageStorage()` flag, but uses it's own. This allows to substitute activation storage in a derived class without breaking layers packing code.

## 2021 Feb 01

- proper `m_layer_stops_bprop` layer marker introduced to distinguish between real `m_layer_input` marked layer and a layer that just terminates a back propagation chain (it was the job of `m_layer_input` marker earlier).
- refactored common code of `LFC` & `layer_output` for fully-connected forward propagation into `LFC_FProp` which also became spawnable (in case one needs the `fprop()` -only part of `LFC` alone).

## 2021 Jan 29

- Refactored and moved dataset normalization code from `_train_data_simple<>` into independent separate class `_impl::td_norm<>` and partly into base class `_impl::_td_base<>`. Finally normalization code is abstract enough to be used with many various `train_data` implementations (provided that they have 4 additional special functions implemented that does actual data-transformation work).
- `transf_train_data<>` is introduced to easily generate training data on the fly.
- `LSUVExt` finally works with an object of `_i_train_data<>` interface (that refactoring - totally coincedentally of course! - fixed some old odd featurebug)
- Upgrades to `math::smatrix<>` interface
- `iMath_t::mTranspose()` and it's variants now properly supports `m_bBatchInRow` mode with all variations of bias/no_bias combination.
- Tested with latest boost library v.1.75.0
- Some minor updates

## 2021 Jan 05

- slightly changed filenames format for `inspectors::dumper<>`
- `m_gradientWorks` member variable of `LFC<>` and `layer_output<>` are no longer public. Just use getter method `get_gradWorks()`
- `bool math::smatrix<>::m_bBatchesInRows` flag is introduced to give more flexibility to algorithms. Make sure you've read a comment for the flag, b/c it is very experimental and unstable (in a sence of support by most of `nntl`) feature. The following classes was audited and should probably work fine with the feature (note - not extensively tested yet!):

  - all in `interface\math\smatrix.h`
  - `inmem_train_data<>` and all it's base classes
  - `nnet<>` and `layers<>`
  - `LFC<>` and `layer_output<>` accepts only activations of previous layer in `bBatchesInRows()` mode and doesn't support own activations in this mode. No general issue to support it for `layer_output<>`, however, there's one for `LFC` with stripping rowvector of biases from `m_activations` before creation of dL/dZ.

## 2020 Dec 27

- corrected semantic of `_grad_works::max_norm()`. The old function now properly named as `_grad_works::max_norm2()` to represent the fact the argument is treated as square of maximum norm. New `_grad_works::max_norm()` treats argument as pure norm value.
- changed `DeCov` regularizer implementation to normalize it to columns/neurons count. The old implementation (as published in paper) required to fit regularizer scale to width of a layer and change it every time the width of layer changes. The new implementation is width-stable, so once a decent regularizer scale was found (big enough to work well but not so big to destroy the signal in dL/dZ), it's much safer to experiment with layer width.

## 2020 Dec 22

- renamed `activation::linear` -> `activation::identity`, `activation::linear_output` -> `activation::identity_custom_loss` and other related stuff.
- small performance optimization for identity activation at `LFC::bprop()`
- now you may use special loss functions that doesn't require exactly the same number of values as there are neurons in output layer. Just remember to override `_i_train_data<>::isSuitableForOutputOf()` with a proper check and supply proper `nnet_evaluator / training_observer`, as well as a custom loss function.
- Y matrices of training data interface are no longer required to have the same base data type as X matrices. Some non-core general purpose elements of the library that built on this assumption are still rely on it, but the core was upgraded.
- `LsuvExt` implementation was significantly reworked and improved.
  - NOTE: `LsuvExt` is still supporting only mean/variance metrics.
- Data normalization to std/mean algo was extracted from `LsuvExt` and now available as a generic in `utils\mtx2Normal.h`.
- Dataset normalization algo on basis of `mtx2Normal` was implemented as a part of `_train_data_simple` and available to use as `inmem_train_data<>`.

## 2020 Apr 07
- Refactored `LPT` class: removed deprecated template parameter `bExpectSpecialDataX` and moved `neurons_count K_tiles` from class' template to constructor parameters.
- minor updates & improvements

## 2020 Apr 05

Huge and important update of the whole framework.
* Refactored the idea of data feeding of the main `nnet::train()` function and therefore whole trainig/testing data usage in the library. Now what previusly was the `train_data` fixed class now can be any class as long as it obeys `i_train_data<>` interface. This makes possible many things including the following important features:

    - on-the-fly training data augmentation
    - working with datasets that doesn't fit into available RAM
    - simultaneous use of any number of different datasets (e.g. now possible to check a validation dataset during nnet performance evaluation) (--though this feature was not completely tested yet)

* Training batchSize is no longer have to be a multiple of training set size. If it is uneven to the training set size, some random training samples will just be skipped during epoch of training.

* Finally made an option to restrict maximum batch size for inferencing (`fprop` mode), see the `nnet_train_opts::maxFpropSize()` function (and example of use in `TEST(Simple, NesterovMomentumAndRMSPropOnlyFPBatch1000)` of `examples` project, file `simple.cpp`).

  Before this option introduced, each layer of neural network allocated as much memory as it needed to do inference on the biggest dataset, which obviously led to enormous memory consumption for sophysticated nnet architectures and/or datasets. Now just use `nnet_train_opts::maxFpropSize(maxInferenceBatch)` to restrict the biggest possible batch size to employ for inference.

  The last thing to note about mini-batch inferencing is... error value computed for a dataset in mini-batch mode doesn't match error computed in full-batch mode (it is much bigger). And I don't get why that happens. It should be (almost) the same if I recall the algorithm properly... Everything else seems to work absolutely correctly fine thought. Either I forgot about some algo details and that behaviour should be expected, or there's a bug I'm not yet aware about of.

* Updated format of binary data files used to store datasets (see `bin_file.h` for details). Older `.bin` files are no longer usable, create new ones with updated matlab scripts (see `./_supp/matlab/` folder). `MNIST` dataset archive was also [updated](https://yadi.sk/d/DpvoqtGGUqh5JQ).

* Note that many internal API's were changed.

* And as always, some old bugs fixed (some new bugs introduced) :D


