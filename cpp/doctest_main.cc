#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN 

#include "doctest/doctest.h"
#include "offset_computer.hpp"

// Tests verifying the validity of the compilation process
#include "hamiltonian.hpp"

// Tests verifying the validity of the state vector and QIFT
#include "state_vector.hpp"

// End-to-end tests over different terms
#include "time_evolution.hpp"
