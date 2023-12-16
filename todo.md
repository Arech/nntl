# TODO

There are multiple issues in this project, most of which was developed about 5 years ago.

## Maintanance
- Refactor benchmarks using a proper GBenchmark based backend
- Rewrite UML generator as a Python script, bwahahaha

## Improvements
- Whole internal `SMath::_istor_alloc` buffer storage system is horrible, unsafe and should not exist. Refactor it to a proper independent allocator component.
- Math interface mustn't as a whole depend on `RealT` template argument. Instead, each function should have it's own base data type template argument.
- Doxygen based docs

## Performance
- Test whether having biases separated from weights doesn't degrade perf notably - decoupling it would allow to use normal input data (now it requires bias column)
- Test which weights matrix multiplication is more efficient with the used BLAS: when samples are in rows, or when samples are in rows? Current design puts samples in rows which makes batch fetching insanely slow for col-major matricies.
- More efficient RNG utilization with data pre-generation.
- Better solution for chosing single-threaded vs. multithreaded function implementation. Ideally - automatic.
