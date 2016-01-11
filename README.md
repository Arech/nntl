# nntl
Neural Network Template Library is a set of C++14 template classes that helps to implement fast vectorized feedforward neural networks. It is multithreaded, x64 friendly and uses OpenBLAS only as a back-end to multiply matrices. NNTL is a header only library and require no other dependencies, except for OpenBLAS and Boost.

### Performance
Here is the performance of training 3 layer 768->500->300->10 network with sigmoid activation and quadratic loss function over MNIST dataset (60000 training samples and 10000 validation samples) for 20 epochs in minibatches of size 100 using double precision floating point math. NN implementation from [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) on Matlab R2014a x64 is taken as a baseline (it is also uses vectorized computations, multithreading and double as basic floating-point type). Hardware in both cases the same: AMD Phenom II X6 1090T @3500Mhz CPU (with all power-saving features turned off) with 16Gb of RAM under Windows 7 (swap file turned off, so no paging occur during testing). The CPU is pretty old today, it has only SSE2+ instructions (no AVX/AVX2), so everything should work a way faster on newer CPUs).

Model|Baseline|NNTL|ratio
-----|--------|----|-----
base|271s|**137s**|**x2.0**
base + momentum|295s|**159s**|**x1.9**
base + momentum + dropout|332s|**166s**|**x2.0**

So, it's about a two times faster (and has a room for further improvements, btw). Not so much, but I'm not aware of anything better (please, contact me if you know). I also tried [tiny-cnn](https://github.com/nyanp/tiny-cnn), but failed to achive even Matlab-comparable performance (not counting that there is only x32 version available out-of-the-box).

Also, one may switch to float data type instead of double to run the code yet about two times faster.

I wouldn't state the NNTL is the fastest CPU implementation of feedforward neural networks, but nonetheless it's pretty fast and BSD-licensed (except for [random number generators](https://github.com/Arech/AF_randomc_h), that is GPL licensed, - but you can easily substitute RNG for you own if you want). It's intended to be as fast as possible, provided that the code is easy to understand and maintain.

## Currently Implemented NN Features
* full-batch or mini-batch learning
* individually tunable feedforward layers (i.e. almost all properties such as activation function, learning rate, dropout and other regularizers coefficients, and so on are defined on per layer basis). Construct any deep learning architecture you want.
* Activation units for output layer:
  * sigmoid with quadratic and cross-entropy (for binary target data) loss function
  * softmax with cross-entropy loss
* Activation units for hidden layers:
  * sigmoid
  * rectified linear units (ReLU)
* Neuron weights initialization schemes:
  * According to Xavier et al. "Understanding the difficulty of training deep feedforward neural networks" 2010 (so called "Xavier initialization" - good for sigmoids)
  * According to He, Zhang et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" 2015 (amazing for ReLU)
  * According to Martens "Deep learning via Hessian-free optimization" 2010 and Sutskever, Martens et al. "On the importance of initialization and momentum in deep learning" 2013 (so called "Sparse initialization" or SI - for sigmoids)
* Optimizers:
  * "classical" constant learning rate
  * RMSProp as Geoffrey Hinton introduced it in "Neural Networks for Machine Learning" course, lecture 6
  * RMSProp modification by Alex Graves (as described in his paper “Generating Sequences With Recurrent Neural Networks” (2013), equations (38)–(45))
  * RProp (sign of a gradient)
  * my own slightly mad modification of RMSProp (probably, someone is also invented it, don't know), which I call ModProp, that uses abs() of gradient in EMA instead of square as in RMSProp. It's slightly faster, than RMSProp, because it eliminates the need of squaring and square rooting, and sometimes it helps to learn weights when no other techniques helps (the latter is probably related to some specific properties of data I used, but anyway, it might be helpful to try it).
* Classical momentum / Nesterov momentum (a.k.a. Nesterov Accelerated Gradient or NAG for short)
* Regularizers:
  * Dropout
  * L1 and L2 (weight decay) regularizers
  * Constraint for a total length of neuron incoming weight vector - so called max-norm regularization. Once neuron weights grow too much, they are getting scaled so their norm will fit into some predefined value (Srivastava, Hinton, et.al "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" 2014)
  * Constraints for a magnitude of derivative of loss function in outer layer (idea taken from aforementioned “Generating Sequences With Recurrent Neural Networks” (2013) by Alex Graves)
* Individual Adaptive Learning Rates (ILR in code) based on agreement in signs of current and previous gradient or momentum velocity.
* Early stopping, learning rates decay, momentum modification and etc.. Any tuning of learning variables you would like during actual training process.
* Tasks supported out of the box (i.e. all you need to do to be able to work with this tasks is to assemble a proper architecture from components provided; other tasks, such as regression, however, may require some special components coding - please, submit your solutions):
  * one-hot vector classification via sigmoid or softmax activations
  * one dimensional binary classification via sigmoid activation

## The Pros and Cons
### Nuances
Just want to stress again: NNTL is not a kind of Plug-n-Play system to solve typical tasks. And it's not mean to do so (however, it's perfectly capable with some tasks out of the box). NNTL is a framework to build fast neural networks and experiment with them. That means, in particular, that you should understand what are you doing, because it's not completely fool-proof and you may "shoot your leg" if you're not sufficiently familiar with C++ or neural networks. If you just want to play with ANN and see what happens, it's probably better to start with something more suitable like [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) for Matlab or [Theano](http://deeplearning.net/tutorial/) for Python. [TensorFlow](http://tensorflow.org/) or [DMTK](http://www.dmtk.io/) is great if you have a lot of computing power.

### Pros
* pretty fast x64 vectorized multithreaded header only C++14 implementation
* single (float) and double precision floating point data types supported.
* modular low coupled architecture that is (I think) easy to understand, maintain and use. Replace / update any module you need, like:
  * math subsystem
  * random number generators
  * multithreading
  * layers
  * activation functions
  * weights initialization schemes
  * ...
* OpenBLAS (for matrix*matrix multiplications) is the only external code dependency (not counting the Boost, which is de facto industry standard). OpenBLAS could be easily replaced/substituted if needed.

### Cons
* achieving the best possible performance with small data sizes (for example, when using very small minibatches and/or small number of neurons) may require some manual tuning of thresholds that define when to use single- or multi-threaded branch of code. At this moment this thresholds are hardcoded into `\nntl\interface\math\imath_basic_thresholds.h` and `\nntl\interface\rng\AFRandom_mt_thresholds.h` respectively. So, you'll need to fix them all to suit your own hardware needs in order to get the best possible performance. However, if you're not going to use too small nets/batches, you'll probably be absolutely fine with current multithreading-by-default implementation of imath interface.
* Random number generator is made on very fast RNGs developed by [Agner Fog](http://www.agner.org/random/randomc.zip). But they are GPL-licensed, therefore are distributed as a separate package [AF_randomc_h](https://github.com/Arech/AF_randomc_h) that has to be downloaded and placed at `/_extern/agner.org/AF_randomc_h` folder. If you don't like it, you can easily use your own RNG by implementing a few interface functions. I wouldn't recommend using a `\nntl\interface\rng\std.h`, because it is about 100-200 times slower than Agner Fog's RNGs (it matters a lot for dropout, for example).
* Built and tested with only one compiler: MSVC2015 on Windows 7. That means, that most likely you'll have to fix some technical issues and incompatibles before you'll be able to compile it with another compiler. Please, submit patches.
* Due to some historical reasons the code lacks handling exceptions, that can be thrown by external components (such as indicating low memory conditions in STL). Moreover, most of code has noexcept attribute. Probably, it won't hurt you much, if you have enought RAM.

## Compilers Supported
Developed and tested on MSVC2015 on Windows 7. Other modern compilers will probably require some hacks to compile. Please, submit your patches.

### How to Use NNTL
1. Download NNTL and unpack it to some %NNTL_ROOT%
2. Download RNGs from repository [AF_randomc_h](https://github.com/Arech/AF_randomc_h) and unpack it to `%NNTL_ROOT%/_extern/agner.org/AF_randomc_h` (actually it's only a single header file)
3. Download latest [Boost](http://www.boost.org/) and setup correct paths in Solution's "VC++ Directories" for include and library files of Boost. In fact compilation of Boost is not required, it's used in header-only mode, therefore actually only include folder should be updated. However this may not be the case for future versions of NNTL.
4. Download or build suitable [OpenBLAS](http://www.openblas.net/) x64 [binaries](http://sourceforge.net/projects/openblas/files) and SDK. Place binaries in PATH or in corresponding debug/release solution folder. Correct paths to SDK in Solution's "VC++ Directories" property page.
5. If your target CPU supports AVX/AVX2 instructions, update "Enable Enhanced Instruction Set" solution setting accordingly.
6. if I didn't forget anything, now you can take a look at [.\nntl\examples\simple.cpp](https://github.com/Arech/nntl/blob/master/examples/simple.cpp) to see how to build your first feedforward neural network with NNTL. I'll write more about it later.

Don't hesitate to ask for help, if you are interested.

#### Note
There may be some other projects referenced in nntl.sln solution file, but absent in distribution, - it's ok, just ignore them.

### How to Build tests Project
1. You'll also need to download and build [Google Test](https://github.com/google/googletest) (preferably version 1.7) to `%NNTL_ROOT%/_extern/gtest-1.7.0/` (you'll need to build /msvc/gtest.vcxproj project). Also download [RapidJson](http://rapidjson.org/) and unpack it to %NNTL_ROOT%/_extern/rapidjson/
2. [Download](https://yadi.sk/d/Mx_6JxTukgJoN) or (provided you have Matlab/Octave installed) convert MNIST data with `%NNTL_ROOT%/nntl/_supp/matlab/mnist2bin.m` from [mnist_uint8.mat](https://github.com/rasmusbergpalm/DeepLearnToolbox/blob/master/data/mnist_uint8.mat) (supplied with DeepLearnToolbox) to correspongind small and full file. Put `mnist60000.bin` and `mnist200_100.bin` (the last file is MNIST dataset cropped to 200 train and 100 test samples version of full MNIST database for use in debug builds) to `%NNTL_ROOT%/data/`.
3. I guess, that's enough to build tests project. It should pass all tests in debug and release modes. To run individual test case use [--gtest_filter](https://github.com/google/googletest/blob/master/googletest/docs/V1_7_AdvancedGuide.md#running-a-subset-of-the-tests) command line option.

## Code Status
I'm actively working on and with this project therefore I won't give any guarantees on API being stable. I'm pretty sure it may be changed from commit to commit, so keep it in mind.

The code itself however should be fairly stable - I'll try to push only stable and well tested changes, though again - no guarantees. Remember to check the latest commit comments to find out if there're some instabilities possible.

## Warning and Disclaimer
The code uses very tight loops of mathematical operations that creates a huge load on CPU. I've encountered some hardware faults and BSODs on my overclocked system (that I thought to be very stable for a long time), until I relaxed overclocking significantly. So, if it'll burn your PC to the ash - you've been warned.
