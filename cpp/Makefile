
	DOCTEST=./doctest-master
    YAML=./yaml-cpp/build

	CXX = g++-9
	CXXFLAGS = -O3 -std=c++17 -funroll-loops -march=native -Wno-unused -Wall -I. -I./yaml-cpp/include -I$(DOCTEST) -I /usr/local/include -g
	LDFLAGS =  -L$(YAML) -lyaml-cpp -L /usr/local/lib -lfftw3_omp -lfftw3 -lm -fopenmp
	MACFLAGS =  ${MKLROOT}/lib/libmkl_intel_ilp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a -fopenmp -lpthread -lm -ldl -I"${MKLROOT}/include"

default: qpesim

SRCS = \
hamiltonian.cc


HEADERS =           \
gate.hpp            \
offset_computer.hpp \
time_evolution.hpp  \
util.hpp            \
state_vector.hpp    \
types.hpp           \
hamiltonian.hpp     \
time_evolution_over_terms.hpp


TESTS =             \
tests

default: all

all: qpesim tests

qpesim: main.cc $(SRCS) $(HEADERS)
	$(CXX) -DDOCTEST_CONFIG_DISABLE $(CXXFLAGS) -o qpesim main.cc $(SRCS) $(LDFLAGS)

tests: doctest_main.cc $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o tests doctest_main.cc $(SRCS) $(LDFLAGS)

qift: large_qift.cc $(SRCS) $(HEADERS)
	$(CXX) -DDOCTEST_CONFIG_DISABLE $(CXXFLAGS) -o qift_large large_qift.cc $(SRCS) $(LDFLAGS)

fftw: fftw_test.cc $(SRCS) $(HEADERS)
	$(CXX) -DDOCTEST_CONFIG_DISABLE $(CXXFLAGS) -o fftw_test fftw_test.cc $(SRCS) $(LDFLAGS)

mkl: mkl_test.cc
	$(CXX) -DDOCTEST_CONFIG_DISABLE $(CXXFLAGS) -m64 -I"${MKLROOT}/include" -o mkl_test mkl_test.cc -L /opt/intel/oneapi/compiler/2021.3.0/mac/compiler/lib -liomp5 -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl

.PHONY: all clean

clean:
	rm -f *.o *~
	rm -f tests
	rm -f qpesim
