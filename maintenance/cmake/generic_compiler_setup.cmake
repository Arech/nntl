# Has to be called strictly AFTER at least one project() call, because otherwise MSVC definition
# will not be defined.

function(generic_compiler_setup)

if(MSVC)
    add_compile_options(
        -Wall

        -wd4514
        # - 'function': unreferenced inline function has been removed
        # This warning is triggered by inline functions declared in headers (not helpful).

        -wd4711
        # - warning C4711: function 'some function(here)' selected for automatic
        #   inline expansion
        # Diagnostics warning that is irrelevant to code quality.

        -wd4820
        # - 'bytes' bytes padding added after construct 'member_name'
        # Implicit padding is not a concern
    )

    # Treat all warnings as errors when not using a Visual Studio generator, since CMake only
    # supports `-external:I` include declarations when using the Makefile or Ninja generators.
    # This prevents from limiting strict warnings to only own code.
    if(NOT CMAKE_GENERATOR MATCHES "^Visual Studio ")
        add_compile_options(-WX)
    endif()

else(MSVC)
    add_compile_options(
        # warnings
        -Wall -Wextra -pedantic -Wconversion -Wsign-conversion -Wdangling-else -Wswitch-enum -Werror -pedantic-errors -Wformat=2 -Wformat-overflow=2 -Wformat-signedness -Wformat-truncation=2 -Wnrvo -Wanalyzer-infinite-recursion -Wimplicit-fallthrough -Wmissing-include-dirs -Wunused -Wuninitialized -Wunknown-pragmas -Wstrict-overflow=5 -Wstringop-overflow=4 -Wsuggest-attribute=const -Wsuggest-attribute=pure -Wduplicated-branches -Wduplicated-cond -Wshadow -Wunsafe-loop-optimizations -Wundef -Wcast-qual -Wlogical-op -Wnormalized -Wpadded -Winvalid-pch -Winvalid-utf8 -Wvector-operation-performance
        # behaviour
        -fmax-errors=10 -flto -fstrict-aliasing

    )
endif(MSVC)

endfunction()