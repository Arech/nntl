# nntl
Neural Network Template Library is a set of C++14 template classes that helps to implement fast vectorized feedforward neural networks. It is multithreaded, x64 friendly and uses OpenBLAS only as a back-end to multiply matrices. NNTL is a header only library and requires no other dependencies, except for OpenBLAS and Boost (header only).

### Performance
*This paragraph is slightly outdated. The library was improved in many ways since the performance measurements, so current results should be a bit better.

Here is the performance of training a 3 layer `768->500->300->10` network with a sigmoid activation function and a quadratic loss function over the MNIST dataset (60000 training samples and 10000 validation samples) for 20 epochs in a minibatches of size 100 using double precision floating point math. A NN implementation from [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) on a Matlab R2014a x64 is taken as a baseline (it also uses vectorized computations, multithreading and double as a basic floating-point type). The hardware in both cases are the same: AMD Phenom II X6 1090T @3500Mhz CPU (with all power-saving features turned off) with 16Gb of RAM under Windows 7 (swap file turned off, so no paging occur during testing). The CPU is pretty old today, it has only SSE2+ instructions (no AVX/AVX2), so everything should work a way faster on newer CPUs).

Model|Baseline|NNTL|ratio
-----|--------|----|-----
base|271s|**137s**|**x2.0**
base + momentum|295s|**159s**|**x1.9**
base + momentum + dropout|332s|**166s**|**x2.0**

One may switch computations to use a float data type instead of a double to run the code even more faster (roughly at about 2/3 of the time required to run with a double precision). Also it's possible to tune a loss evaluation strategy to skip evaluation at some/all epochs, which will allow to train the NN even more faster.

I wouldn't state the NNTL is the fastest CPU implementation of feedforward neural networks, but nonetheless it's pretty fast and BSD-licensed (except for [random number generators](https://github.com/Arech/AF_randomc_h), which is GPL licensed, - but it's easy to substitute RNG (as well as almost any other part of the library) for another implementation if needed).

## Currently Implemented NN Features
* A full-batch or a mini-batch SGD
* Individually tunable feedforward layers (i.e. almost all layer properties such as an activation function, a learning rate, a dropout, etc and so on are defined on a per layer basis).
* The following feedforward layer types has been implemented allowing one to create many different types of feedforward neuron connectivity that helps to encapsulate some important prior knowledge of a data into a neural network architecture:
  * ordinary layers:
    * layer_**fully_connected** is a basic layer type where all the magic happens
    * layer_**output** is a variation of fully connected layer specialized to be an output of a neural network
    * layer_**input** provides a common interface to a training data for hidden layers
    * layer_**identity** allows one to pass an incoming data to upper layers unmodified
    * layer_**identity_gate** passes incoming neurons up to a layer stack while allowing *_gated layer types to use them as a gating source.
  * compound layers (these layer types allows to encapsulate other layers in some way to produce more sophisticated architectures):
    * layer_**pack_gated** helps to deal with an optional source data and allows to train a feature detector that is specialized only on the optional part of a data.
    * layer_**pack_horizontal** is designed to feed different sequential (and possibly overlapping) ranges of underlying neurons to a corresponding different layers. For example, this allows one to build a set of feature detectors each of which is specialized on a specific subset of data features.
    * layer_**pack_horizontal_gated** adds a gating ability to a layer_pack_horizontal. It allows one to build a set of feature detectors for an optional data.
    * layer_**pack_tile** allows to process different sequential subsets of neurons by a single layer producing different output for each subset. It's kind a 'predecessor' of a convolutional layer, that implies strong regularization by a network architecture means.
    * layer_**pack_vertical** helps to build a vertical stack of layers that is represented as a single (compound) layer.
* Activation units for an output layer:
  * sigmoid with a **quadratic** and a **cross-entropy** (for binary target data) loss function
  * **softmax** with a cross-entropy loss function.
  * SoftSigm unit with a **quadratic** loss function.
* Activation units for hidden layers:
  * **Sigm**oid
  * Rectified linear units (**ReLU**)
  * **Leaky ReLU**
  * Exponential Linear Units (**ELU**). ( alpha*(exp(x)-1) for x<0 and (x) for x>0; See Clevert et al. "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", 1511.07289 )
  * Exponential logarithmic Units (ELogU) ( alpha*(exp(x)-1) for x<0 and log(x+1)/log(b) for x>0 ). I haven't seen a description of this kind of units in academic papers, however, I'm sure someone else has also tryed them. They are good to squash too big output (you may get from ReLU-family units) to more reasonable values and aren't saturating contrary to sigmoids.
  * Logarithmic units (LogLogU) is -log(1-x)/log(b_neg) for x<0 and log(x+1)/log(b_pos) for x>0. See the note for the ELogU.
  * **SoftSign** (y=x/(a+|x|)) and SoftSigm (version of softsign scaled to a range of (0,1)) units. These units offer a better learning performance in some setups as they saturate much slower than corresponding sigmoids (they approach their limits in a polynomial way instead of exponential).
  * Nondiffirentiable step unit (y = 0|x<0 & 1|x>=0)
* Neuron weights initialization schemes:
  * According to **Xavier** et al. "Understanding the difficulty of training deep feedforward neural networks" 2010 (so called "Xavier initialization" - good for sigmoids)
  * According to **He, Zhang** et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" 2015 (amazing for ReLU family units)
  * According to **Martens** "Deep learning via Hessian-free optimization" 2010 and Sutskever, Martens et al. "On the importance of initialization and momentum in deep learning" 2013 (so called "Sparse initialization" or SI - for sigmoids)
* Optimizers:
  * "classical" constant learning rate
  * **RMSProp** as Geoffrey Hinton introduced it in the "Neural Networks for Machine Learning" course, lecture 6
  * RMSProp modification by Alex Graves (as described in his paper “Generating Sequences With Recurrent Neural Networks” (2013), equations (38)–(45))
  * **RProp** (using a sign of a gradient)
  * my own slightly mad modification of the RMSProp (probably, someone is also invented it, don't know), which I call a ModProp, that uses an abs() of a gradient in EMA instead of a square as in the RMSProp. It's slightly faster, than the RMSProp, because it eliminates the need of squaring and square rooting. Sometimes it helps to learn weights when no other technique helps (it may be related to some specific properties of the data I used, but anyway, it might be helpful to try).
  * **Adam** and **AdaMax** (Kingma, Ba "Adam: A Method for Stochastic Optimization" 2014). I've added a numeric stabilizer coefficient to the latter method (it's absent in the original description, though it probably should be there). It's possible to turn it off completely if necessary to get the AdaMax exactly as described in the paper.
* Classical **momentum** / **Nesterov momentum** (a.k.a. Nesterov Accelerated Gradient or NAG for short)
* Regularizers:
  * **Dropout** (actually, it's so called "inverted dropout" where activations is scaled only at a training time; during a testing activations/weights with and without dropout remains the same).
  * **L1** and **L2** (weight decay) regularizers.
  * Constraint for a total length of a neuron incoming weight vector - so called **max-norm** regularization. Once a neuron weights grow too much, they are getting scaled so their norm will fit into a some predefined value (Srivastava, Hinton, et.al "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" 2014)
  * Constraints for a magnitude of derivative of a loss function in an output layer (idea taken from the aforementioned “Generating Sequences With Recurrent Neural Networks” (2013) by Alex Graves)
* Individual Adaptive Learning Rates (an ILR in code) based on a agreement in signs of a current and a previous gradient or a momentum velocity.
* Early stopping, learning rates decay, momentum modification and etc.. Any tuning of learning variables you would like during actual training process.
* Tasks supported out of the box (i.e. all you need to do to be able to work with this tasks is to assemble a proper architecture from components provided; other tasks, such as regression, however, may require some special components coding - please, submit your solutions):
  * one-hot vector classification via sigmoid or softmax activations
  * one dimensional binary classification via sigmoid activation
* Debugging and baby-sitting a neural network learning process is implemented through a special interface which allows to monitor/dump any temporary variable or matrix you might want to examine during a training session (activation/preactivation values, weights, weigth updates and many more - and much more will be added as it'll be needed). The interface is easily extensible and incurs zero run-time cost if it isn't used (it is off by default). Dumping data to Matlab's `.mat` files is already available at the moment. See an example at test_inspectors.cpp.
* Actually, I have left one real **gem** unpublished - a component that turns an architecture description process from the hell pain of writing tonns of extremely hard to maintain C++ code which defines layer objects and their relationships, into a real joy of drawing/updating something very similar to UML class diagramm. I'll upload it sometime later, but feel free to contact me to hurry up this process.

## The Pros and Cons
### Nuances
Just want to stress again: NNTL is not a kind of a Plug-n-Play system to solve typical tasks. And it's not mean to do so (however, it's perfectly capable with some tasks out of the box). NNTL is a C++ framework to build fast neural networks and experiment with them. That means, in particular, that you should understand what are you doing, because it's not a completely fool-proof and you may "shoot your leg" if you're not sufficiently familiar with the C++ or neural networks. If you just want to play with neural networks and see what happens, it's probably better to start with something more suitable like [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) for Matlab or [Theano](http://deeplearning.net/tutorial/) for Python. [TensorFlow](http://tensorflow.org/) or [DMTK](http://www.dmtk.io/) is great if you have a lot of computing power.

### The Pros
* pretty fast x64 vectorized multithreaded header only C++14 implementation, that allows to build almost any kind of feedforward neural network architecture.
* single (float) and double precision floating point data types are supported.
* modular low coupled architecture that is (I think) easy to understand, maintain and use. Replace / update / derive any module you need, like:
  * math subsystem
  * random number generators
  * multithreading
  * layers
  * activation functions
  * weights initialization schemes
  * ...
* OpenBLAS (for matrix*matrix multiplications) is the only external code dependency (not counting the Boost, which is de facto industry standard). OpenBLAS also could be easily replaced/substituted if needed.
* Three data loaders already available (see `/nntl/_supp/io` folder):  
  * `matfile` implements "Saving" and "Loading Archive" Concepts from boost::serialization, therefore allowing reading and writing data to and from Matlab's native `.mat` files (including dumping a neural network object state into a .mat file for an inspection). Requires a Matlab installed as well as `#define NNTL_MATLAB_AVAILABLE 1` to compile. `NNTL_MATLAB_AVAILABLE` is defined by default for all projects (see it in `stdafx.h`). See `nntl/utils/matlab.h` for details how to setup a build environment to be able to compile the NNTL with a Matlab support (thanks to a Mathworks, it's not harder than to plug in a Boost support).
  * `binfile` reads an input data from a nntl-custom binary file format. It's the fastest method to read a data, however `.bin` files require the most disk space and are prone to cross-platform issues with data-sizes and endianness.
  * `jsonreader` reads input data from `.json` files (requires the RapidJson package, see below). It's the most crossplatform, but the slowest solution.

### The Cons
* achieving the best possible performance with small data sizes (for example, when using very small minibatches and/or a small number of neurons) may require some manual tuning of thresholds that define when to use a single- or multi-threaded branch of a code. At this moment this thresholds are hardcoded into a `\nntl\interface\math\mathn_thr.h` and a `\nntl\interface\rng\afrand_mt_thr.h` respectively. So, you'll need to fix them all to suit your own hardware needs in order to get the best possible performance (It's a real hell, btw). However, if you're not going to use too small nets/batches, you'll probably be fine with a current multithreading-by-default implementation of the i_math interface.
* Current mathematical subsystem implementation is for a CPU only. Other types (GPU, for example) could 'easily' be added, however, someone has to write&test them.
* Random number generator is made on very fast RNGs developed by [Agner Fog](http://www.agner.org/random/randomc.zip). But they are GPL-licensed, therefore are distributed as a separate package [AF_randomc_h](https://github.com/Arech/AF_randomc_h) that has to be downloaded and placed at the `/_extern/agner.org/AF_randomc_h` folder. If you don't like it, you can easily use your own RNG by implementing a few interface functions. I wouldn't recommend using a `\nntl\interface\rng\std.h`, because it is about a 100-200 times slower than Agner Fog's RNGs (it matters a lot for a dropout, for example).
* Built and tested with only one compiler: the MSVC2015 on Windows 7. That means, that most likely you'll have to fix some technical issues and incompatibles before you'll be able to compile it with another compiler. Please, submit patches.
* Due to some historical reasons the code lacks handling exceptions that can be thrown by external components (such as indicating low memory conditions in STL). Moreover, most of a code has the noexcept attribute. Probably, it won't hurt you much, if you have enought RAM.
* There is almost no documentation at this moment. You shouldn't be scared by necessity of reading a code and a code comments in order to undestand how to use some components. However, I tried to make this process easy by extensively commenting a code and making it clean&clear. You decide if it helps and feel free to contact me if you need clarifications.

## Compilers Supported
Developed and tested on the MSVC2015 (update 3 at this moment) on Windows 7. Other modern compilers will probably require some hacks to compile, however, I don't think these hacks would be critical. Please, submit your patches.

### How to Use The NNTL
1. Download the NNTL and unpack it to a some `%NNTL_ROOT%` folder.
2. Download the RNGs from the repository [AF_randomc_h](https://github.com/Arech/AF_randomc_h) and unpack it to `%NNTL_ROOT%/_extern/agner.org/AF_randomc_h` (actually it's only a single header file)
3. Download the latest [Boost](http://www.boost.org/) and setup correct paths in Solution's "VC++ Directories" for include and library files of the Boost. In fact, compilation of the Boost is not required, it's used in header-only mode, therefore actually only an include folder path should be updated. However this may not be the case for a future versions of the NNTL.
4. Download the [OpenBLAS](http://www.openblas.net/) SDK and build suitable x64 binaries or download them from the [repository](https://sourceforge.net/projects/openblas/files/) (use the [v0.2.14](https://sourceforge.net/projects/openblas/files/v0.2.14/) for a quickstart as it's the version that was used during development/testing of the NNTL. Download the [OpenBLAS-v0.2.14-Win64-int32.zip](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip/download) and the supplemental [mingw64_dll.zip](https://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download) ). Place binaries in a PATH or into a corresponding debug/release solution folder. Update paths to the SDK in the Solution's "VC++ Directories" property page.
5. If your target CPU supports AVX/AVX2 instructions, then update the "Enable Enhanced Instruction Set" solution's setting accordingly.
6. If you have a Matlab installed and want to use their `.mat` files to interchange a data with the NNTL, then leave the line `#define NNTL_MATLAB_AVAILABLE 1` as is in the `stdafx.h` and see instructions in the `nntl/utils/matlab.h` on how to update solution's build settings. If not, change the difinition to the `#define NNTL_MATLAB_AVAILABLE 0` and don't use the `nntl/_supp/io/matfile.h`.
7. if I didn't forget anything, now you can take a look at the [.\nntl\examples\simple.cpp](https://github.com/Arech/nntl/blob/master/examples/simple.cpp) to see how to build your first feedforward neural network with the NNTL. I'll write more about it later.

Don't hesitate to ask for help, if you are interested.

#### Note
There may be some other projects referenced in the `nntl.sln` solution file, but absent in the distribution, - it's ok, just ignore them.

### How to Build the `tests` Project
1. You'll also need to download the [Google Test](https://github.com/google/googletest) (preferably version 1.7) to the `%NNTL_ROOT%/_extern/gtest-1.7.0/` folder and to build the `/msvc/gtest.vcxproj` project. Also download the [RapidJson](http://rapidjson.org/) and unpack it to the `%NNTL_ROOT%/_extern/rapidjson/`
2. [Download](https://yadi.sk/d/xr4BJKV4vRjh6) or (provided you have a Matlab/Octave installed) convert the MNIST data with the `%NNTL_ROOT%/nntl/_supp/matlab/mnist2bin.m` script from the [mnist_uint8.mat](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat) file (supplied with the DeepLearnToolbox) to a corresponding small and a full file. Put the `mnist60000.bin` and the `mnist200_100.bin` (the last file is the MNIST dataset cropped to the 200 train and the 100 test samples for use in debug builds) to the `%NNTL_ROOT%/data/` folder.
3. I guess, that's enough to build and run the `tests` project. It should pass all tests in a debug and release modes. To run an individual test case use the [--gtest_filter](https://github.com/google/googletest/blob/master/googletest/docs/V1_7_AdvancedGuide.md#running-a-subset-of-the-tests) command line option.

## The Code Status
The only thing I can guarantee at this moment is that I'm actively working on and with this project. I won't issue any guarantees on any parts of API being stable. I'm pretty sure it may be changed from a commit to a commit, so keep it in mind while working on derived projects.

The code itself however should be fairly stable, but I won't guarantee that also. Please note, it's mostly an experimental project therefore you may expect any kind of weirdness. Remember to check commit descriptions to find out if there're some instabilities expected.

Feel free to contact me if you need some assistance.

## Warning and Disclaimer
The code uses very tight loops of mathematical operations that creates a huge load on a CPU. I've encountered some hardware faults and BSODs on my overclocked system (that I thought to be very stable for a long time), until I relaxed the overclocking significantly.
