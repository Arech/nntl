** Code structure
/tests folder contains code for nntl unit-testing
VC "tests" project is used to develop nntl and testing it. It's the core project of solution.
nntl files located in /nntl folder

** Design principles
- don't use exception if there is faster workaround available
--- don't use exception - use noexept or throw() whenever possible and pray any external code would never throw. It's ok to terminate in case of exception.

