# Changelog
I should probably have started doing it long ago, but better late than never. So here it is (for older entries see commit history)

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


