# Tests Directory

Welcome to the tests for the Y3 Cluster Cosmology Pipeline. This directory should contain only source code, data files should be in the `../data` directory.

## Building and Running Tests

Each test lives in a file called `<name>.test.cc`, where `<name>` is the name of the tests. To run the tests, build according to the `README` in the parent of this directory, and run `ctest` from that directory. Each test is compiled to an executable named `<name>_test`, which may be run on its own to check only that test and none other.

If one of the tests is failing and you need to see the output of the failing test, run the following:

```
$ CTEST_OUTPUT_ON_FAILURE=1 ctest
```

And ctest will tell you what checks failed, where.

## Running Python Tests

Python tests live alongside the C++ ones as `<name>.test.py`, mirroring
the `.test.cc` convention. Run them with pytest **from this directory**
(`pytest.ini` lives here, not the repo root, so pytest only picks up its
`*.test.py` discovery pattern and `--import-mode=importlib` setting when
invoked from here or a path under it):

```
$ cd test
$ pytest
```

Some Python tests read a real-pipeline fixture dump and skip gracefully
if it hasn't been generated yet — see the `cosmosis` run instructions in
each test's own header comment.

## Adding Tests

Please do add more tests! To do so is straightforward. Create a file in this directory named `your_test.test.cc`, and fill it with the following:

```
// This header is needed for the test framework
#include "catch2/catch.hpp"
// Than, any other headers you want - e.g.:
#include <lc_lt_t.hh>
#include <primitives.hh>
#include <test/your_test_file.hh>
...

using ...;
TEST_CASE("Your test name")
{
    // Your code - any necessary initializations and setup

    ...

    // Add any number of checks you wish. Either exact:
    CHECK(something == expected);
    // Or approximate:
    CHECK(value = Approx(expected).epsilon(1e-2).margin(1e-2);
}
```

Next, add the commands to build your test to the end of `CMakeLists.txt` in this directory:

```
add_executable(<your_test>_test main.cc <your_test>.test.cc)
target_link_libraries(<your_test>_test PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} interp lc_lt)
target_include_directories(<your_test>_test
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/externals)
add_test(<your_test>_test <your_test>_test)
```

Where `<your_test>` has been replaced with a descriptive name. Now, simply fire up the docker image and run `make` in the parent directory, and your test should automatically be built! (No need to rerun `cmake`.) If you are unsure how to write your test or are having trouble, look in the other `*.test.cc` files, and don't hesitate to ask for help.

## Diagnostic Executables

In addition, there are several non-test executables, intended to ease examining the behavior of specific terms of the integral. These include:

*  `integrate_(lc_lt|lo_lc|mor|roffset)`
    * Integrate, respectively, the probability distributions $P(\lc|\lt)$, $P(\lo|\lt)$, $P(\lt|M,\zt)$, and $P_{mis}(R_{mis}$, over respective ranges. This is to check whether the distributions integrate properly to 1.0, within acceptable limits.
    * Each executable emits CSV-formatted results tables of the resulting integration value, as well as the values of extra parameters used.
    * Each executable takes a number of command-line arguments for customization. Obtain a list with `./integrate_<whatever> -h` or `--help`.
* `integrate_delta_sigma`
    * Compares `del_sig_cen` and `del_sig_mis`. Aims to have a similar interface to the above integrations.
