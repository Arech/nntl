# CMake's option() allows to define only boolean variables.
# option_list() defines a variable with a name ${var_name} to have any value allowed by the list
# "${def_value};${rest_list}"
# option_list() is better than option() even for defining booleans because it checks whether the
# variable have an allowed value while option() doesn't do that allowing the var to have any value
# set by -D CLI argument

function(option_list var_name def_value description rest_list)
    if(DEFINED ${var_name})
        set(DO_CHECK ON)
    else()
        set(DO_CHECK OFF)
    endif()

    # create the variable
    set(${var_name} "${def_value}" CACHE STRING "${description}")

    # define list of values GUI will offer for the variable
    set(VALS_LIST "${def_value}" ${rest_list})
    set_property(CACHE ${var_name} PROPERTY STRINGS "${VALS_LIST}")

    if(DO_CHECK)
        # since the value exist, check if value is in the list
        set(TEST_VAL "${${var_name}}")
        foreach(val IN LISTS VALS_LIST)
            if("${val}" STREQUAL "${TEST_VAL}")
                set(DO_CHECK OFF)
                break()
            endif()
        endforeach()
        if(DO_CHECK)
            message(FATAL_ERROR "Value of ${var_name} must be one of ${VALS_LIST}. Got ${TEST_VAL}")
        endif()
    endif()

endfunction()
