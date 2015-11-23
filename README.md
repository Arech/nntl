# nntl
Neural Network Template Library is a set of C++14 template classes that helps to implement fast vectorized feedforward neural networks. It is multithreaded, x64 friendly and uses OpenBLAS and Yeppp! as a mathematical back-ends (the latter however proved to be useless for my hardware - or may be it's thanks to Microsoft who'd done a nice job with their optimizing compiler). NNTL is a header only library and require no other dependencies.

I wouldn't state it's the fastest CPU implementation of FF NN, but nonetheless it's fast and BSD-licensed (except for [random number generators](https://github.com/Arech/AF_randomc_h), that is GPL licensed, - but you can easily substitute RNG for you own if you want).

## Currently Implemented NN Features
* individually tunable feedforward layers
* sigmoid activation units with quadratic and cross-entropy loss function for outer layer
* sigmoid and rectified linear units (ReLU) for hidden layers
* dropout
* momentum / Nesterov momentum
* Optimizers:
  * "classical" constant learning rate
  * RMSProp as Geoffrey Hinton introdiced it in "Neural Networks for Machine Learning" course, lecture 6
  * RMSProp modification by Alex Graves (as described in his paper “Generating Sequences With Recurrent Neural Networks” (2013), equations (38)–(45))
  * RProp (sign of a gradient)
  * my own slightly mad modification of RMSProp (probably, someone is also invented it, don't know), which I call ModProp, that uses abs() of gradient in EMA instead of square as in RMSProp. It's slightly faster, than RMSProp, because it eliminates the need of squaring and square rooting, and sometimes it helps to learn weights when no other techniques helps (the latter is probably related to some specific properties of data I used, but anyway, it might be helpful to try it).
* Individual Adaptive Learning Rates (ILR in code) based on agreement in signs of current and previous gradient or momentum velocity.
* Constraints for a magnitude of derivative of loss function in outer layer (idea taken from aforementioned “Generating Sequences With Recurrent Neural Networks” (2013) by Alex Graves)

## The Pros and Cons
### Pros
* pretty fast x64 vectorized multithreaded header only implementation
* modular low coupled architecture that is (I think) easy to understand, maintain and use. Replace / update any module you need, like:
  * math subsystem
  * random number generators
  * multithreading
  * layers
  * activation functions
  * ...
* Depends only on OpenBLAS (for matrix*matrix multiplications) and Yeppp (in fact, current configuration doesn't use Yeppp for NN computations, because naive implementations of necessary operations proved to be more effective)

### Cons
* The best performance requires a lot of manual tuning of thresholds that define when to use single or multi-threaded branch of code. At this moment this thresholds are hardcoded into implementation of math and rng interface (\nntl\interface\math\i_yeppp_openblas.h and \nntl\interface\rng\AFRandom_mt.h respectively). So, you'll need to fix them all to suit your own hardware needs in order to get the best possible performance. And then you'll have a hard time to merge updates... It's a major drawback, but I have no time to solve it now.
* Module, that does multithreading, has two implementations. If you run Windows Vista/Server 2008 or newer Windows OS, you will be able to use \nntl\interface\threads\winqdu.h, that is built on fast native [SRWLOCK](https://msdn.microsoft.com/en-us/library/windows/desktop/aa904937%28v=vs.85%29.aspx)'s and [CONDITION_VARIABLE](https://msdn.microsoft.com/en-us/library/windows/desktop/ms682052%28v=vs.85%29.aspx)'s. If not, you'll have to use \nntl\interface\threads\std.h which is build on standard std::mutex and std::condition_variable, which is about 1.5 times slower (on my hardware). I think, most OSes have something similar to mentioned SRWLOCK's, so please, submit updates.
* Random number generator uses very fast RNGs made by [Agner Fog](http://www.agner.org/random/randomc.zip). But they are GPL-licensed, therefore they are distributed in separate package [AF_randomc_h](https://github.com/Arech/AF_randomc_h) that has to be downloaded and placed at /_extern/agner.org/AF_randomc_h folder. Or use your own RNG. I wouldn't recommend using a \nntl\interface\rng\std.h, because it is about 100-200 times slower (it matters a lot for dropout, for example)

## Compilers Supported
Developed and tested on MSVC2015 on Windows 7. Other modern compilers will probably require some hacks to compile. Please, submit your patches.

## Code Status
I'm actively working on and with this project therefore I won't give any guarantees on API being stable. I'm pretty sure it may be changed from commit to commit, so keep it in mind.

The code itself however should be fairly stable - I'll try to push only stable and well tested changes, though again - no guarantees.
