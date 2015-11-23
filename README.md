# nntl
Neural Network Template Library is a set of C++14 template classes that helps to implement fast vectorized feedforward neural networks. It is multithreaded, x64 friendly and uses OpenBLAS and Yeppp! as a mathematical back-ends (the latter however proved to be useless for my hardware - or may be it's thanks to Microsoft who'd done a nice job with their optimizing compiler). NNTL is a header only library and require no other dependencies.

I wouldn't state it's the fastest CPU implementation of FF NN, but nonetheless it's fast and BSD-licensed.

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
  * my own slightly mad modification of RMSProp (probably, someone is also invented it, don't know), which I call ModProp, that uses abs() of gradient in EMA instead of square as in RMSProp. It's slightly faster, than RMSProp, because it eliminates the need of squaring and square rooting, and sometimes it helps to learn weights when no other techniques helps (the latter is probably related to some specific of data I used, but anyway, it may be helpful to try).
* Individual Adaptive Learning Rates (ILR in code) based on agreement in signs of current and previous gradients or momentum velocity.
* Constraints for a magnitude of derivative of loss function in outer layer (idea taken from aforementioned “Generating Sequences With Recurrent Neural Networks” (2013) by Alex Graves)


## Compilers supported
Developed and tested on MSVC2015 on Windows 7. Other modern compilers will probably require some hacks to compile. Please, submit your patches.
