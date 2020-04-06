# Changelog
I should probably have started doing it long ago, but better late than never. So here it is (for older entries see commit history)

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


