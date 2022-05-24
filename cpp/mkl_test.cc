#include <cstdint>
#include <cassert>
#include <iostream>
#include <complex.h>
#include <vector>
// #include <mkl_service.h>
// #include <mkl_dfti.h>
// #include <mkl_cdft.h>
// #include <omp.h>
#include <mkl.h>

uint64_t nChoosek(int n, int k)
{
    if (k > n)
        return 0;
    if (k * 2 > n)
        k = n - k;
    if (k == 0)
        return 1;

    uint64_t result = n;
    for (int i = 2; i <= k; ++i)
    {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

class StateVector : public std::vector<std::complex<double>>
{
public:
    StateVector(unsigned precision, unsigned num_orbitals,
                unsigned num_occupied)
        : precision{precision}, num_orbitals_{num_orbitals}, num_occupied_{num_occupied}
    {

        assert(num_occupied <= num_orbitals);
        assert(precision > 0);
        n_choose_o_ = nChoosek(num_orbitals, num_occupied);
        resize((1ul << precision) * n_choose_o_ * n_choose_o_, 0);
        assert((*this).max_size() > (1ul << precision) * n_choose_o_ * n_choose_o_);
    }

    void qift_mkl()
    {
        StateVector &state_vector = *this;
        uint64_t ld = n_choose_o_ * n_choose_o_;
        std::cout << "LD: " << (MKL_LONG) ld << std::endl;
        uint64_t num_trials = 1 << precision;

        DFTI_DESCRIPTOR_HANDLE descriptor;
        MKL_LONG status;

        MKL_LONG sizes[2] = {ld, num_trials};
        MKL_LONG offsets[2] = {0, ld};

        status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, num_trials);
        std::cout << "MKL DFT Plan Step 0 Initialized" << std::endl;
        // set the computation to be in-place
        status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);

        // parallelization
        status = DftiSetValue(descriptor, DFTI_THREAD_LIMIT, 40);

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

protected:
    unsigned precision;
    unsigned num_orbitals_;
    unsigned num_occupied_;
    uint64_t n_choose_o_;
};

int main(int argc, char *argv[])
{
    StateVector sve = StateVector(std::stoi(argv[1]), 15, 5);
    sve[0] = 1.0;

    sve.qift_mkl();

    return 0;
}