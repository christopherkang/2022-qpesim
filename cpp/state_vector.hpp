#ifndef STATE_VECTOR_HPP_
#define STATE_VECTOR_HPP_

#include <cassert>
#include <chrono>
#include <complex.h>
#include <functional> 
#include <numeric>
#include <vector>
// #include <fftw3.h>
#include <mkl.h>

#include "gate.hpp"
#include "util.hpp"

class StateVector : public std::vector<std::complex<double>> {
 public:
  // variable reserved for the implicit fftw plan
  // fftw_plan master_plan;
  StateVector(unsigned precision, unsigned num_orbitals,
              unsigned num_occupied)
      : precision{precision},
        num_orbitals_{num_orbitals},
        num_occupied_{num_occupied} {

    // TO - DO - RENAME ALL num_eigen_qubits_ TO precision
    assert(num_occupied <= num_orbitals);
    assert(precision > 0);
    n_choose_o_ = nChoosek(num_orbitals, num_occupied);
    resize((1ul << precision) * n_choose_o_ * n_choose_o_, 0);
    // std::complex<double>* arr = new std::complex<double>[(1ul << precision) * n_choose_o_ * n_choose_o_];
    
    // #pragma omp parallel for
    // for (size_t i = 0; i < (1ul << precision) * n_choose_o_ * n_choose_o_; i++) {
    //   arr[i] = 0;
    // }
    // assert(false);

    std::cout << "nCo: " << n_choose_o_ << std::endl;
    assert((*this).max_size() > (1ul << precision) * n_choose_o_ * n_choose_o_);
    std::cout << "Initialized StateVector" << std::endl;
  }

  /**
   * @brief apply a gate on an eigen qubit. Cannot yet apply gates on state
   * qubits.
   *
   * @param gate Gate to be applied
   * @param qubit eigen qubit to be applied to
   */
  void apply_gate(const Gate1D& gate, unsigned qubit) {
    StateVector& state_vector = *this;
    assert(qubit < precision);
    auto num_qubits = precision;
    auto ld = n_choose_o_ * n_choose_o_;
    unsigned target = num_qubits - qubit - 1;
    for (unsigned left = 0; left < (1ul << num_qubits);
         left += (1 << (target + 1))) {
      for (unsigned right = 0; right < (1ul << target); ++right) {
        for (size_t i = 0; i < ld; i++) {
          size_t b0 = (left + 0 + right) * ld + i;
          size_t b1 = (left + (1 << target) + right) * ld + i;
          auto coeff0 =
              gate(0, 0) * state_vector[b0] + gate(0, 1) * state_vector[b1];
          auto coeff1 =
              gate(1, 0) * state_vector[b0] + gate(1, 1) * state_vector[b1];
          state_vector[b0] = coeff0;
          state_vector[b1] = coeff1;
        }
      }
    }
  }

  void apply_controlled_gate(const Gate1D& gate, unsigned control,
                             unsigned qubit) {
    StateVector& state_vector = *this;
    auto num_qubits = precision;
    uint64_t ld = n_choose_o_ * n_choose_o_;
    assert(qubit < num_qubits);
    assert(control < num_qubits);
    assert(control != qubit);
    assert(num_qubits >= 2);

    unsigned new_control = num_qubits - control - 1;
    unsigned new_target = num_qubits - qubit - 1;
    unsigned p0 = std::min(new_control, new_target);
    unsigned p1 = std::max(new_control, new_target);

    // unsigned p0 = std::min(control, qubit);
    // unsigned p1 = std::max(control, qubit);

    // we would like to get all of the nCe**2 within a specific prefix
    // we would like to get all prefixes with 1 in the control bit and anything for the mid bits 
    // and anything for the targets
    for (uint64_t up = 0; up < (1ul << num_qubits); up += 1 << (p1 + 1)) {
      for (uint64_t mid = 0; mid < (1ul << p1); mid += 1 << (p0 + 1)) {
        for (uint64_t lo = 0; lo < (1ul << p0); lo += 1) {
          for (size_t i = 0; i < ld; i++) {
            size_t b0 =
                (up + mid + lo + (1ul << new_control) + (0ul << new_target)) * ld + i;
            size_t b1 =
                (up + mid + lo + (1ul << new_control) + (1ul << new_target)) * ld + i;
            auto coeff0 =
                gate(0, 0) * state_vector[b0] + gate(0, 1) * state_vector[b1];
            auto coeff1 =
                gate(1, 0) * state_vector[b0] + gate(1, 1) * state_vector[b1];
            state_vector[b0] = coeff0;
            state_vector[b1] = coeff1;
          }
        }
      }
    }
  }

  void qift() 
  {
    StateVector &state_vector = *this;
    uint64_t ld = n_choose_o_ * n_choose_o_;
    std::cout << "LD: " << (MKL_LONG) ld << std::endl;
    uint64_t num_trials = 1 << precision;

    // we're going to add a simple check to validate the probability distribution
    std::cout << "----- VALIDATION ------" << std::endl;
    double temp = 0;
    #pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < ld * num_trials; i++) {
      temp = temp + std::pow(std::abs(state_vector[i]), 2);
    }
    std::cout << "TOTAL PROBABILITY DISTRIBUTION SUM: " << temp << std::endl;
    std::cout << "----- END VALIDATION --";
    // end validation

    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    MKL_LONG sizes[2] = {ld, num_trials};
    MKL_LONG offsets[2] = {0, ld};

    status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, num_trials);
    std::cout << "MKL DFT Plan Step 0 Initialized" << std::endl;
    // set the computation to be in-place
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);

    // parallelization
    status = DftiSetValue(descriptor, DFTI_THREAD_LIMIT, 60);

    // complex -> complex transform
    // status = DftiSetValue(descriptor, DFTI_FORWARD_DOMAIN, DFTI_COMPLEX);
    // std::cout << DftiErrorMessage(status) << std::endl;
    // assert (status == DFTI_NO_ERROR);
    status = DftiSetValue(descriptor, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);

    // number of transforms and input distances
    status = DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, ld);
    status = DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, 1);
    status = DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, 1);
    status = DftiSetValue(descriptor, DFTI_INPUT_STRIDES, offsets);
    status = DftiSetValue(descriptor, DFTI_OUTPUT_STRIDES, offsets);

    // commiting the descriptor
    status = DftiCommitDescriptor(descriptor);
    std::cout << "MKL DFT Plan Initialized" << std::endl;

    status = DftiComputeForward(descriptor, state_vector.data());
    assert (status == DFTI_NO_ERROR);
    std::cout << "MKL DFT Plan Executed" << std::endl;

    status = DftiFreeDescriptor(&descriptor);
    std::cout << "Data freed" << std::endl;

    // plan the Fourier Transform
    double NORMALIZATION_FACTOR = sqrt(num_trials);

#pragma omp parallel for
    for (uint64_t i = 0; i < ld * num_trials; i++)
    {
        state_vector[i] /= NORMALIZATION_FACTOR;
    }
  }

  /**
   *
   * @brief QIFT for @param state_vector. @param c bits. The c bits are the MSB
   * bits in the address. For different values of the c bits (0..2^c-1) there
   are ld elements. Essentially, the storage is:
   * ld elements for 000..0 (c times), ld elements for 000..1, etc.
   * Follows the implementation here: https://www.cl.cam.ac.uk/teaching/1920/QuantComp/Quantum_Computing_Lecture_9.pdf

   * @param state_vector
   * @param c
   * @param ld
   */
  // void qift() {
  //   // for (int q = precision - 1; q >= 0; --q) {
  //   //   for (int t = precision - 1; t > q; --t) {
  //   //     apply_controlled_gate(InvRotate(t - q + 1), t, q);
  //   //     // apply_controlled_gate(Rotate(q - precision + 1), q, t);
  //   //   }
  //   //   apply_gate(Hadamard(), q);
  //   // }

  //   // for (unsigned q = 0; q <= precision - 1; q++) {
  //   //   for (unsigned t = 0; t < q; t++) {
  //   //     apply_controlled_gate(InvRotate(q - t + 1), t, q);
  //   //   }
  //   //   apply_gate(Hadamard(), q);
  //   // }

  //   StateVector& state_vector = *this;

  //   // FFTW - PLANNING
  //   // methods to make this multi-threaded
  //   int NUMBER_OF_THREADS = 40;
  //   fftw_init_threads();
  //   fftw_plan_with_nthreads(NUMBER_OF_THREADS);
  //   // std::cout << "NUM THREADS: " << fftw_planner_nthreads() << std::endl;

  //   uint64_t ld = n_choose_o_ * n_choose_o_;
  //   uint64_t num_trials = 1 << precision;

  //   fftw_iodim64 arr_dims[1];
  //   arr_dims[0].n = num_trials;
  //   arr_dims[0].is = ld;
  //   arr_dims[0].os = ld;

  //   fftw_iodim64 arr_howmany_dims[1];
  //   arr_howmany_dims[0].n = ld;
  //   arr_howmany_dims[0].is = 1;
  //   arr_howmany_dims[0].os = 1;

  //   // look at this:
  //   // https://stackoverflow.com/questions/58513592/confusion-about-fftw3-guru-interface-3-simultaneous-complex-ffts

  //   // g++ -DDOCTEST_CONFIG_DISABLE -O3 -std=c++17 -funroll-loops -march=native -Wno-unused -Wall -I. -I./yaml-cpp/include -I./doctest-master -I /usr/local/include -I/hpc/software/spack/opt/spack/linux-ubuntu18.04-cascadelake/gcc-9.4.0/fftw-3.3.9-sae5lnfifboo2puuigq3uy7vtq7s32vm/include/ -g -o qpesim main.cc hamiltonian.cc -L./yaml-cpp/build -lyaml-cpp -L /usr/local/lib -L/hpc/software/spack/opt/spack/linux-ubuntu18.04-cascadelake/gcc-9.4.0/fftw-3.3.9-sae5lnfifboo2puuigq3uy7vtq7s32vm/lib/ -lfftw3_omp -lfftw3 -lm -fopenmp
  //   auto io = reinterpret_cast<fftw_complex*>(state_vector.data());

  //   fftw_plan master_plan = fftw_plan_guru64_dft(
  //     1, arr_dims, 1, arr_howmany_dims,
  //     io, io, -1, FFTW_ESTIMATE
  //   );

  //   assert(master_plan != NULL);

  //   std::cout << "DFT Plan Initialized" << std::endl;

  //   // plan the Fourier Transform
  //   double NORMALIZATION_FACTOR = sqrt(num_trials);

  //   fftw_execute(master_plan);

  //   std::cout << "DFT Plan Executed" << std::endl;

  //   #pragma omp parallel for
  //   for (uint64_t i = 0; i < ld * num_trials; i++){
  //     state_vector[i] /= NORMALIZATION_FACTOR;
  //   }

  //   fftw_destroy_plan(master_plan);
  //   // fftw_cleanup_threads();

  //   // old code
  //   // assert(precision >= 2);
  //   // std::vector<std::complex<double>> buf1(num_trials);
  //   // std::vector<std::complex<double>> buf2(num_trials);
  //   // std::vector<std::complex<double>> buf3(num_trials);
  //   // std::vector<std::complex<double>> buf4(num_trials);

  //   // fftw_plan p1 = fftw_plan_dft_1d(num_trials, reinterpret_cast<fftw_complex*>(buf1.data()), reinterpret_cast<fftw_complex*>(buf1.data()), -1, FFTW_MEASURE);
  //   // fftw_plan p2 = fftw_plan_dft_1d(num_trials, reinterpret_cast<fftw_complex*>(buf2.data()), reinterpret_cast<fftw_complex*>(buf2.data()), -1, FFTW_MEASURE);
  //   // fftw_plan p3 = fftw_plan_dft_1d(num_trials, reinterpret_cast<fftw_complex*>(buf3.data()), reinterpret_cast<fftw_complex*>(buf3.data()), -1, FFTW_MEASURE);
  //   // fftw_plan p4 = fftw_plan_dft_1d(num_trials, reinterpret_cast<fftw_complex*>(buf4.data()), reinterpret_cast<fftw_complex*>(buf4.data()), -1, FFTW_MEASURE);

  //   // //timing
  //   // std::chrono::duration<double> gather_diff;
  //   // std::chrono::duration<double> compute_diff;
  //   // std::chrono::duration<double> scatter_diff;

  //   // // pragma can go here
  //   // // Unrolled loop
  //   // for (uint64_t ab = 0; ab < ld; ab+= 4) {
  //   //   // Inverse FFT of state_vector[b to b+2^p step n_choose_e * n_choose_e]
  //   //   // recognize that when QIFT is applied, we are applying QIFT|0> |\psi(0)> + QIFT|1> |\psi(1)> + ...
  //   //   // by linearity. Then, we can accumulate all of the rows of |\psi>, because QIFT|y> -> \sum c(x) |x> |\psi(y)>
  //   //   // so, let's just group by |\psi(y)> rows, knowing that we'd like to identify all the new \sum \sum c(x) |x> final coeffs
  //   //   // #pragma omp parallel for

  //   //   // see where memory cost is coming from:
  //   //   // fftw running or gather/scatter data

  //   //   auto start = std::chrono::high_resolution_clock::now();

  //   //   // array of buffers: omp_num_threads * unroll factor
  //   //   for(uint64_t i = 0; i < num_trials; i++) {
  //   //     buf1[i] = state_vector[i * ld + ab];
  //   //     buf2[i] = state_vector[i * ld + ab + 1];
  //   //     buf3[i] = state_vector[i * ld + ab + 2];
  //   //     buf4[i] = state_vector[i * ld + ab + 3];
  //   //   }

  //   //   auto gather = std::chrono::high_resolution_clock::now();

  //   //   fftw_execute(p1);
  //   //   fftw_execute(p2);
  //   //   fftw_execute(p3);
  //   //   fftw_execute(p4);

  //   //   auto compute = std::chrono::high_resolution_clock::now();

  //   //   // NOTE - this is not normalized, so buf1 needs to be normalized
  //   //   // #pragma omp parallel for
  //   //   for(uint64_t i = 0; i < num_trials; i++) {
  //   //     state_vector[i*ld + ab] = buf1[i] / NORMALIZATION_FACTOR;
  //   //     state_vector[i*ld + ab + 1] = buf2[i] / NORMALIZATION_FACTOR;
  //   //     state_vector[i*ld + ab + 2] = buf3[i] / NORMALIZATION_FACTOR;
  //   //     state_vector[i*ld + ab + 3] = buf4[i] / NORMALIZATION_FACTOR;
  //   //   }

  //   //   auto scatter = std::chrono::high_resolution_clock::now();
  //   //   gather_diff += gather - start;
  //   //   compute_diff += compute - gather;
  //   //   scatter_diff += scatter - compute;
  //   // }
  //   // std::cout << "Gather Time: " << gather_diff.count() << std::endl;
  //   // std::cout << "Compute time: " << compute_diff.count() << std::endl;
  //   // std::cout << "Scatter time: " << scatter_diff.count() << std::endl;

  //   // fftw_destroy_plan(p1);
  //   // fftw_destroy_plan(p2);
  //   // fftw_destroy_plan(p3);
  //   // fftw_destroy_plan(p4);
  // }

  /**
   * @brief Converts a StateVector with some precision into a vector of probabilities of
   * each of the potential phases. This should be called after all QPE parts (including QIFT)
   * have been executed.
   * 
   * @return std::vector<double> 
   * @note does not modify the StateVector
   */
  std::vector<double> IdentifyPhaseAmplitudes() {
    unsigned max_num_vals = std::pow(2, precision);
    std::vector<double> probs(max_num_vals, 0.0);

    // for each potential amplitude, we are going to add the probability to the phase
    unsigned width = pow(n_choose_o_, 2);
    #pragma omp parallel for
    for (unsigned idx = 0; idx < max_num_vals; idx++) {
      // we want to take each idx of the probs output and adjust it accordingly
      double temp_sum = 0;
      for (unsigned off = 0; off < width; off++) {
        temp_sum += pow(std::abs((*this)[idx * width + off]), 2);
      }
      probs[idx] = temp_sum;
    }


    // for (unsigned amp_idx = 0; amp_idx < (*this).size(); amp_idx++) {
    //   unsigned phase = amp_idx / std::pow(n_choose_o_, 2);
    //   std::cout << phase << std::endl;
    //   // unfortunately, each of these phases is now LittleEndian and needs to be read as such
    //   // So, let's reverse the digits of each of these phases
    //   // phase = reverse_bits(phase, precision);

    //   std::complex<double> amp = (*this)[amp_idx];
    //   // std::cout << phase << " | ";
    //   double new_prob = pow(std::abs(amp), 2);
    //   probs[phase] += new_prob;
    // }

    return probs;
  }

 protected:
  unsigned precision;
  unsigned num_orbitals_;
  unsigned num_occupied_;
  uint64_t n_choose_o_;
}; // class StateVector

#include "doctest/doctest.h"

TEST_CASE("Simple 1D Gate Application") {
  auto s = StateVector(1, 2, 1);
  // our state should have 8 combinations
  // 2 from the first qubit (precision), and 2**2 from nCe (2C1)
  // let's start by putting the StateVector as 0|01|01
  s[0] = std::complex(1.0);
  auto X = Gate1D(0, 1.0, 1.0, 0);
  s.apply_gate(X, 0);
  
  // now, we should have converted this state from 0|01|01 -> 1|01|01
  REQUIRE_EQ(s[4], 1.0);

  // Now, let's try a slightly bigger StateVector
  s = StateVector(2, 2, 1);
  s[0] = std::complex(1.0);
  s.apply_gate(X, 0);

  // this should leave us with 10|01|01
  REQUIRE_EQ(s[8], 1.0);

  // the State Vector is indexed as follows:
  // The first nCe**2 idxs are reserved for the 00...0 string
  // Within this 00...0 string, each block of nCe represent a shared second block
  // So, 00...0 | 0001 | (0001, 0010, ...1000)
}

TEST_CASE("Hadamard") {
  auto s = StateVector(1, 2, 1);
  s[0] = std::complex(1.0);
  s.apply_gate(Hadamard(), 0);
  // 0|01|01 / 1|01|01
  REQUIRE_EQ(s[0], 1.0 / std::pow(2, 0.5));
  REQUIRE_EQ(s[4], 1.0 / std::pow(2, 0.5));

  s = StateVector(2, 2, 1);
  s[0] = std::complex(1.0);
  s.apply_gate(Hadamard(), 0);
  // 00|01|01 / 10|01|01
  REQUIRE_EQ(s[0], 1.0 / std::pow(2, 0.5));
  REQUIRE_EQ(s[8], 1.0 / std::pow(2, 0.5));
}

TEST_CASE("Simple Controlled 1D Gate Application") {
  // again, we'll take one of the simplest potential StateVectors
  auto s = StateVector(2, 2, 1);
  // this should allow us to have a starting position of 00|01|01.... 11|10|10
  s[0] = std::complex(1.0);
  auto X = Gate1D(0, 1.0, 1.0, 0);
  s.apply_controlled_gate(X, 1, 0);
  // there should be no impact
  REQUIRE_EQ(s[0], 1.0);
  // now, let's instead start with 01|01|01
  s[0] = std::complex(0.0);
  s[4] = std::complex(1.0);
  s.apply_controlled_gate(X, 1, 0);
  // this should put us at 11|01|01
  REQUIRE_EQ(s[12], 1.0);

  s.apply_controlled_gate(X, 0, 1);
  // this should put us at 10|01|01
  REQUIRE_EQ(s[8], 1.0);

  s = StateVector(3, 2, 1);
  s[4] = std::complex(1.0);
  // we are now at 001|01|01

  s.apply_controlled_gate(X, 2, 0);
  // this should put us at 101|01|01
  REQUIRE_EQ(s[20], 1.0);

  s.apply_controlled_gate(X, 0, 2);
  // this should put us at 100|01|01
  REQUIRE_EQ(s[16], 1.0);
}

TEST_CASE("QIFT") {
  using namespace std::complex_literals;
  const unsigned MAX_QIFT_SIZE = 5;
  const double PI = 3.14159265358979323846;
  for (unsigned size = 1; size < MAX_QIFT_SIZE; size++) {
    StateVector s = StateVector(size, 1, 1);
    // so, our state will now be permanently as [ancilla] | 1 | 1
    // we'd like to initialize it with values so that it collapses nicely
    // Recall that QFT has the following action (ref: https://en.wikipedia.org/wiki/Quantum_Fourier_transform)
    // |x> -> 1/\sqrt{N} \sum_{k=0}^{N-1} w^{xk}_N |k>
    // So, because QIFT is QFT conjugate transpose, we can verify with the opposite coeficients
    double N = std::pow(2, size);
    for (unsigned col_idx = 0; col_idx < N; col_idx++) {
      // begin by prepping the StateVector in the right state
      for (unsigned amp_idx = 0; amp_idx < N; amp_idx++) {
        s[amp_idx] = std::exp(2 * PI * 1i * (col_idx * amp_idx / N)) / std::sqrt(N);
      }

      s.qift();

      // now, verify the QIFT works
      std::vector<double> probs;

      for (auto amp : s) {
        probs.push_back(std::abs(amp));
      }
      double max_idx = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
      CAPTURE(max_idx);
      CAPTURE(s[max_idx]);
      CAPTURE(col_idx);
      CAPTURE(size);

      // we also need to reference a different index
      // our array is now LittleEndian, and must consider it appropriately
      // so, flip col_idx from its BigEndian representation to its LittleEndian rep

      std::vector<bool> bitrep(size, false);

      unsigned tmp = col_idx;

      for (unsigned idx = 0; idx < size; idx++) {
        bitrep[idx] = tmp & 1;
        tmp >>= 1;
      }

      unsigned new_col_idx = 0;
      for (unsigned idx = 0; idx < bitrep.size(); idx++) {
        new_col_idx <<= 1;
        new_col_idx += (unsigned)bitrep.at(idx);
      }

      CAPTURE(new_col_idx);
      CHECK(std::abs(std::complex(1.0) - s[new_col_idx]) < 1e-5);
    }
  }
}

#endif // STATE_VECTOR_HPP_
