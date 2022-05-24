#include <cstdint>
#include <cassert>
#include <iostream>
#include <complex.h>
#include <vector>
#include <fftw3.h>

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

    void qift()
    {
        StateVector &state_vector = *this;

        int NUMBER_OF_THREADS = 40;
        fftw_init_threads();
        fftw_plan_with_nthreads(NUMBER_OF_THREADS);

        uint64_t ld = n_choose_o_ * n_choose_o_;
        uint64_t num_trials = 1 << precision;

        fftw_iodim64 arr_dims[1];
        arr_dims[0].n = num_trials;
        arr_dims[0].is = ld;
        arr_dims[0].os = ld;

        fftw_iodim64 arr_howmany_dims[1];
        arr_howmany_dims[0].n = ld;
        arr_howmany_dims[0].is = 1;
        arr_howmany_dims[0].os = 1;

        auto io = reinterpret_cast<fftw_complex *>(state_vector.data());

        fftw_plan master_plan = fftw_plan_guru64_dft(
            1, arr_dims, 1, arr_howmany_dims,
            io, io, -1, FFTW_ESTIMATE);

        assert(master_plan != NULL);

        std::cout << "DFT Plan Initialized" << std::endl;

        // plan the Fourier Transform
        double NORMALIZATION_FACTOR = sqrt(num_trials);

        fftw_execute(master_plan);

        std::cout << "DFT Plan Executed" << std::endl;

#pragma omp parallel for
        for (uint64_t i = 0; i < ld * num_trials; i++)
        {
            state_vector[i] /= NORMALIZATION_FACTOR;
        }

        fftw_destroy_plan(master_plan);
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

    sve.qift();

    return 0;
}