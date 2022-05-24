#ifndef TIME_EVOLUTION_MANUAL_HPP
#define TIME_EVOLUTION_MANUAL_HPP

#include <algorithm>
#include <random>

#include <complex>
#include <chrono>
#include <cstdint>
#include <vector>
#include <omp.h>

#include "hamiltonian.hpp"
#include "offset_computer.hpp"
#include "state_vector.hpp"
#include "types.hpp"
#include "util.hpp"

struct MixedPQRSLoopInfo
{
    uint64_t niter;
    uint64_t off01;
    uint64_t off10;
    unsigned mid_parity;
};

// global variables
unsigned e;
unsigned n;
Offset e_of_n;
Offset e1_of_n1;
Offset e1_of_n2;
Offset e2_of_n2;
Offset e2_of_n3;
Offset e2_of_n4;
Offset expbase;
std::function<unsigned(unsigned, unsigned, unsigned)> OFFSET;

// utility method
inline void updateStateVector(StateVector &sv, std::complex<double> &left, std::complex<double> &right, unsigned left_idx, unsigned right_idx);

// update methods
inline void updatePQRS_AAAA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s);
inline void updatePQRS_BBBB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s);
inline void updatePQRS_ABBA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s);

inline void updatePP_AA(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p);
inline void updatePP_BB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p);

inline void updatePQ_AA(StateVector &state_vector,
                        std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        uint64_t exp, unsigned p, unsigned q);
inline void updatePQ_BB(StateVector &state_vector,
                        std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        uint64_t exp, unsigned p, unsigned q);

inline void updatePQQP_AA(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q);
inline void updatePQQP_BB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q);
inline void updatePQQP_AB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q);

inline void updatePQQR_AAAA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r);
inline void updatePQQR_BBBB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r);
inline void updatePQQR_ABBA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r);
inline void updatePQQR_BAAB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r);

// discovery helper methods
inline unsigned discoverCCAA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned q, unsigned r, unsigned s, const unsigned &e);
inline unsigned discoverCNA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned q, unsigned r, const unsigned &e);
inline unsigned discoverCA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned r, const unsigned &e);
inline unsigned discoverNN_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned r, const unsigned &e);
inline unsigned discoverN_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned q, const unsigned &e);
inline void parallelPrefixSum(std::vector<unsigned> &out);
inline void applyUnitary(StateVector &state_vector, Hamiltonian &hamiltonian, IdxSingleVec &pp, IdxSingleVec &pq, IdxDoubleVec &pqqp, IdxDoubleVec &pqqr, IdxDoubleVec &pqrs, unsigned exp);
inline void applyUnitary_reordered(StateVector &state_vector, Hamiltonian &hamiltonian, IdxSingleVec &pp, IdxSingleVec &pq, IdxDoubleVec &pqqp, IdxDoubleVec &pqqr, IdxDoubleVec &pqrs, unsigned exp);
inline void setupGlobals(Hamiltonian &h);

extern double rdouble;
inline std::complex<double> time_evolve_over_terms(StateVector &state_vector, Hamiltonian &hamiltonian, unsigned precision)
{
    // ( ? )
    // omp_set_num_threads(40);
    setupGlobals(hamiltonian);
    std::cout << "nCe: " << e_of_n << "\n";
    std::complex<double> result;

    // grab the indexed terms
    auto full_term_list = hamiltonian.PrintIndexedSegmentedHamiltonianTerms();

    IdxSingleVec pp, pq;
    IdxDoubleVec pqqp, pqqr, pqrs;

    std::tie(pp, pq, pqqp, pqqr, pqrs) = full_term_list;
    // hamiltonian.PrintHamiltonianTerms(true);

    // auto rng = std::default_random_engine{5};
    // std::shuffle(std::begin(pq), std::end(pq), rng);
    // std::shuffle(std::begin(pqqr), std::end(pqqr), rng);
    // std::shuffle(std::begin(pqrs), std::end(pqrs), rng);

    int rand_count = 0;

    for (uint64_t exp = 1; exp < (1ul << precision); exp++)
    {
        // std::chrono::duration<double> diff;

        // if (exp >> rand_count == 1)
        // {
        //     rand_count++;
        //     auto rng = std::default_random_engine{5};
        //     std::shuffle(std::begin(pq), std::end(pq), rng);
        //     std::shuffle(std::begin(pqqr), std::end(pqqr), rng);
        //     std::shuffle(std::begin(pqrs), std::end(pqrs), rng);
        // }

        // we should first copy the iteration over from the previous round
        unsigned prev = (exp - 1) * expbase;
        unsigned curr = prev + expbase;
        std::copy_n(state_vector.begin() + prev, expbase, state_vector.begin() + curr);
        // - potential for parallel copy

        // applyUnitary(state_vector, hamiltonian, pp, pq, pqqp, pqqr, pqrs, exp);
        applyUnitary_reordered(state_vector, hamiltonian, pp, pq, pqqp, pqqr, pqrs, exp);
    }
    return result;
}

inline void time_evolve_iqpe(StateVector &state_vector, Hamiltonian &hamiltonian, unsigned num_of_applications)
{
    setupGlobals(hamiltonian);
    auto full_term_list = hamiltonian.PrintIndexedSegmentedHamiltonianTerms();

    IdxSingleVec pp, pq;
    IdxDoubleVec pqqp, pqqr, pqrs;

    std::tie(pp, pq, pqqp, pqqr, pqrs) = full_term_list;

    for (unsigned i = 0; i < num_of_applications; i++)
    {
        applyUnitary(state_vector, hamiltonian, pp, pq, pqqp, pqqr, pqrs, 1);
    }
}

inline void setupGlobals(Hamiltonian &h)
{
    e = h.num_occupied();
    n = h.num_orbitals();
    e_of_n = nChoosek(n, e);
    e1_of_n1 = nChoosek(n - 1, e - 1);
    e1_of_n2 = nChoosek(n - 2, e - 1);
    e2_of_n2 = nChoosek(n - 2, e - 2);
    e2_of_n3 = nChoosek(n - 3, e - 2);
    e2_of_n4 = nChoosek(n - 4, e - 2);
    expbase = powint(e_of_n, 2);

    OFFSET = [](unsigned e_fac, unsigned pos_alpha, unsigned pos_beta) {
        return e_fac * expbase + pos_alpha * e_of_n + pos_beta;
    };
}

inline void applyUnitary(StateVector &state_vector, Hamiltonian &hamiltonian, IdxSingleVec &pp, IdxSingleVec &pq, IdxDoubleVec &pqqp, IdxDoubleVec &pqqr, IdxDoubleVec &pqrs, unsigned exp)
{
    for (auto term : pp)
    {
        // execute the term
        unsigned p, q;
        std::pair<int, int> targets = term.first;
        std::tie(p, q) = targets;

        auto coeff = hamiltonian.pp_term(p);

        if (p < n)
        {
            // AA
            updatePP_AA(state_vector, coeff, exp, p);
        }
        else
        {
            // BB
            updatePP_BB(state_vector, coeff, exp, p);
        }
    }

    auto pp_time = std::chrono::high_resolution_clock::now();

    for (auto term : pq)
    {
        unsigned p, q;
        std::pair<unsigned, unsigned> targets;
        targets = term.first;
        std::tie(p, q) = targets;

        auto even_coeffs = hamiltonian.pq_term(p, q, 0);
        auto odd_coeffs = hamiltonian.pq_term(p, q, 1);

        assert(p > q);

        if (p < n && q < n)
        {
            // alphas
            updatePQ_AA(state_vector, even_coeffs, odd_coeffs, exp, p, q);
        }
        else
        {
            // betas
            updatePQ_BB(state_vector, even_coeffs, odd_coeffs, exp, p, q);
        }
    }

    auto pq_time = std::chrono::high_resolution_clock::now();

    for (auto term : pqqp)
    {
        unsigned p, q, garb1, garb2;

        std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term.first;
        std::tie(p, q, garb1, garb2) = targets;
        auto coeff = hamiltonian.pqqp_term(p, q);

        assert(p < q);

        // there are three potential combos
        // -> AAAA | ABBA | BBBB

        if (p < n && q < n)
        {
            updatePQQP_AA(state_vector, coeff, exp, p, q);
        }
        else if (p >= n && q >= n)
        {
            updatePQQP_BB(state_vector, coeff, exp, p, q);
        }
        else
        {
            updatePQQP_AB(state_vector, coeff, exp, p, q);
        }
    }

    auto pqqp_time = std::chrono::high_resolution_clock::now();

    for (auto term : pqqr)
    {
        // execute the term
        unsigned p, q, r, garb;

        std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term.first;
        std::tie(p, q, garb, r) = targets;
        assert(r > p);
        assert(q == garb);

        // these should be the same spins
        assert(r - p < n);

        auto even_coeffs = hamiltonian.pqqr_term(p, q, r, 0);
        auto odd_coeffs = hamiltonian.pqqr_term(p, q, r, 1);

        // we have a few potential combinations with spins
        // first, know that the q terms must have the same spin
        // so, that yields:
        // 1. pa qa qa ra -> all < n
        // 2. pb qa qa rb -> p, r >= n, q < n
        // 3. pa qb qb ra -> p, r < n, q >= n
        // 4. pb qb qb rb -> all >= n

        if (p < n && q < n)
        {
            // all alphas
            // so, the beta BV can be whatever
            // let's create the list of appropriate PQQR alpha bvs
            // auto start = std::chrono::high_resolution_clock::now();
            updatePQQR_AAAA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r);
            // auto end = std::chrono::high_resolution_clock::now();
            // diff += end - start;
        }
        else if (p >= n && q >= n)
        {
            // all betas
            updatePQQR_BBBB(state_vector, even_coeffs, odd_coeffs, exp, p, q, r);
        }
        else
        {
            if (q >= n)
            {
                // q in beta, pr in alpha
                // p,r / alpha / -----------------------
                updatePQQR_ABBA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r);
            }
            else
            {
                updatePQQR_BAAB(state_vector, even_coeffs, odd_coeffs, exp, p, q, r);
            }
        }
    }

    auto pqqr_time = std::chrono::high_resolution_clock::now();
    unsigned num_aaaa = 0;
    unsigned num_bbbb = 0;
    unsigned num_abba = 0;

    for (auto term : pqrs)
    {
        // execute the term
        unsigned p, q, r, s;

        std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term.first;
        std::tie(p, q, r, s) = targets;

        auto even_coeffs = hamiltonian.pqrs_term(p, q, r, s, 0);
        auto odd_coeffs = hamiltonian.pqrs_term(p, q, r, s, 1);

        unsigned idx0011, idx1100;

        // first, let's do all the alphas
        if (p < n && q < n && r < n && s < n)
        {
            num_aaaa++;
            // all alphas
            // so, we just need to iterate over the alphas until we find one that works
            // let's first create the list of alphas that will work

            updatePQRS_AAAA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
        else if (p >= n && q >= n && r >= n && s >= n)
        {
            num_bbbb++;
            // all alphas
            // so, we just need to iterate over the alphas until we find one that works
            // let's first create the list of alphas that will work
            updatePQRS_BBBB(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
        else
        {
            // mixed case
            num_abba++;

            assert(p < n && s < n);
            assert(q >= n && r >= n);

            updatePQRS_ABBA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
    }
    std::cout << "a/b: " << num_aaaa << " " << num_abba << " " << num_bbbb << std::endl;

    auto pqrs_time = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> diff = pp_time - unit_start;
    // std::cout << "PP: " << diff.count() << std::endl;
    // diff = pq_time - pp_time;
    // std::cout << "PQ: " << diff.count() << std::endl;
    // diff = pqqp_time - pq_time;
    // std::cout << "PQQP: " << diff.count() << std::endl;
    // diff = pqqr_time - pqqp_time;
    // std::cout << "PQQR: " << diff.count() << std::endl;
    std::chrono::duration<double> diff2 = pqrs_time - pqqr_time;
    std::cout << "PQRS: " << diff2.count() << std::endl;

    std::chrono::duration<double> diff = pqqr_time - pqqp_time;
    std::cout << "pqqr: " << diff.count() << std::endl;
}

inline void applyUnitary_reordered(StateVector &state_vector, Hamiltonian &hamiltonian, IdxSingleVec &pp, IdxSingleVec &pq, IdxDoubleVec &pqqp, IdxDoubleVec &pqqr, IdxDoubleVec &pqrs, unsigned exp)
{
    for (auto term : pp)
    {
        // execute the term
        unsigned p, q;
        std::pair<int, int> targets = term.first;
        std::tie(p, q) = targets;

        auto coeff = hamiltonian.pp_term(p);

        if (p < n)
        {
            // AA
            updatePP_AA(state_vector, coeff, exp, p);
        }
        else
        {
            // BB
            updatePP_BB(state_vector, coeff, exp, p);
        }
    }

    auto pp_time = std::chrono::high_resolution_clock::now();

    for (auto term : pqqp)
    {
        unsigned p, q, garb1, garb2;

        std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term.first;
        std::tie(p, q, garb1, garb2) = targets;
        auto coeff = hamiltonian.pqqp_term(p, q);

        assert(p < q);

        // there are three potential combos
        // -> AAAA | ABBA | BBBB

        if (p < n && q < n)
        {
            updatePQQP_AA(state_vector, coeff, exp, p, q);
        }
        else if (p >= n && q >= n)
        {
            updatePQQP_BB(state_vector, coeff, exp, p, q);
        }
        else
        {
            updatePQQP_AB(state_vector, coeff, exp, p, q);
        }
    }

    auto pqqp_time = std::chrono::high_resolution_clock::now();

    for (auto term : pq)
    {
        unsigned p, q;
        std::pair<unsigned, unsigned> targets;
        targets = term.first;
        std::tie(p, q) = targets;

        auto even_coeffs = hamiltonian.pq_term(p, q, 0);
        auto odd_coeffs = hamiltonian.pq_term(p, q, 1);

        assert(p > q);

        if (p < n && q < n)
        {
            // alphas
            updatePQ_AA(state_vector, even_coeffs, odd_coeffs, exp, p, q);
        }
        else
        {
            // betas
            updatePQ_BB(state_vector, even_coeffs, odd_coeffs, exp, p, q);
        }

        // std::cout << "HIGHLIGHT: " << p << " " << q << std::endl;

        for (auto term_pqqr : pqqr)
        {
            // execute the term
            unsigned p_, q_, r_, garb;
            std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term_pqqr.first;
            std::tie(p_, q_, garb, r_) = targets;  
            assert(r_ > p_);

            // p rr q terms
            // p_ q_q_ r_
            // std::cout << p_ << " " << q_ << " " << r_ << " " << std::endl;

            if (p_ == q && r_ == p){
                assert(q_ == garb);
                // std::cout<<"Happened!!" << std::endl;

                // these should be the same spins
                assert(r_ - p_ < n);

                auto even_coeffs = hamiltonian.pqqr_term(p_, q_, r_, 0);
                auto odd_coeffs = hamiltonian.pqqr_term(p_, q_, r_, 1);

                // we have a few potential combinations with spins
                // first, know that the q terms must have the same spin
                // so, that yields:
                // 1. pa qa qa ra -> all < n
                // 2. pb qa qa rb -> p, r >= n, q < n
                // 3. pa qb qb ra -> p, r < n, q >= n
                // 4. pb qb qb rb -> all >= n

                if (p_ < n && q_ < n)
                {
                    // all alphas
                    // so, the beta BV can be whatever
                    // let's create the list of appropriate PQQR alpha bvs
                    // auto start = std::chrono::high_resolution_clock::now();
                    updatePQQR_AAAA(state_vector, even_coeffs, odd_coeffs, exp, p_, q_, r_);
                    // auto end = std::chrono::high_resolution_clock::now();
                    // diff += end - start;
                }
                else if (p_ >= n && q_ >= n)
                {
                    // all betas
                    updatePQQR_BBBB(state_vector, even_coeffs, odd_coeffs, exp, p_, q_, r_);
                }
                else
                {
                    if (q_ >= n)
                    {
                        // q in beta, pr in alpha
                        // p,r / alpha / -----------------------
                        updatePQQR_ABBA(state_vector, even_coeffs, odd_coeffs, exp, p_, q_, r_);
                    }
                    else
                    {
                        updatePQQR_BAAB(state_vector, even_coeffs, odd_coeffs, exp, p_, q_, r_);
                    }
                }
            }
        }
    }
    
    unsigned num_aaaa = 0;
    unsigned num_bbbb = 0;
    unsigned num_abba = 0;

    for (auto term : pqrs)
    {
        // execute the term
        unsigned p, q, r, s;

        std::tuple<unsigned, unsigned, unsigned, unsigned> targets = term.first;
        std::tie(p, q, r, s) = targets;

        auto even_coeffs = hamiltonian.pqrs_term(p, q, r, s, 0);
        auto odd_coeffs = hamiltonian.pqrs_term(p, q, r, s, 1);

        unsigned idx0011, idx1100;

        // first, let's do all the alphas
        if (p < n && q < n && r < n && s < n)
        {
            num_aaaa++;
            // all alphas
            // so, we just need to iterate over the alphas until we find one that works
            // let's first create the list of alphas that will work

            updatePQRS_AAAA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
        else if (p >= n && q >= n && r >= n && s >= n)
        {
            num_bbbb++;
            // all alphas
            // so, we just need to iterate over the alphas until we find one that works
            // let's first create the list of alphas that will work
            updatePQRS_BBBB(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
        else
        {
            // mixed case
            num_abba++;

            assert(p < n && s < n);
            assert(q >= n && r >= n);

            updatePQRS_ABBA(state_vector, even_coeffs, odd_coeffs, exp, p, q, r, s);
        }
    }
    std::cout << "a/b: " << num_aaaa << " " << num_abba << " " << num_bbbb << std::endl;
}

// // #pragma omp declare simd
inline void updateStateVector(StateVector &sv, std::array<std::complex<double>, 4> &coeffs, unsigned leftIdx, unsigned rightIdx)
{
    std::complex<double> v_l, v_r;
    v_l = sv[leftIdx];
    v_r = sv[rightIdx];

    // std::cout << "IDX: " << leftIdx % 16 << ", " << rightIdx % 16 << std::endl;

    // nv_l = v_l * left + v_r * right;
    // nv_r = v_l * right + v_r * left;

    // sv[leftIdx] = nv_l;
    // sv[rightIdx] = nv_r;

    sv[leftIdx] = v_l * coeffs[0] + v_r * coeffs[1];
    sv[rightIdx] = v_l * coeffs[1] + v_r * coeffs[0];

    // potential alternative:
    // std::complex<double> v_l;
    // v_l = sv[leftIdx];
    // sv[leftIdx] = v_l * left + sv[rightIdx] * right;
    // sv[rightIdx] = v_l * right + sv[rightIdx] * left;

    // return 0;
}

inline void _updateAAAA(StateVector &sv, std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        std::vector<MixedPQRSLoopInfo> &alpha_pairs, uint64_t num_parts_a, uint64_t exp)
{
    std::vector<unsigned> goalposts(num_parts_a + 1);
    goalposts[0] = 0;
    for (uint64_t i = 1; i <= num_parts_a; i++)
    {
        goalposts[i] = alpha_pairs[i - 1].niter;
    }

    parallelPrefixSum(goalposts);

    uint64_t total_number_iter = goalposts[num_parts_a];

#pragma omp parallel for schedule(guided)
    for (uint64_t pos = 0; pos < total_number_iter; pos++)
    {
        auto pos_alpha_it = std::upper_bound(goalposts.begin(), goalposts.end(), pos);
        unsigned pos_alpha = (pos_alpha_it - goalposts.begin()) - 1;
        unsigned pos_na = pos - goalposts[pos_alpha];

        uint64_t offa01 = alpha_pairs[pos_alpha].off01;
        uint64_t offa10 = alpha_pairs[pos_alpha].off10;

        unsigned start_0011 = OFFSET(exp, offa01 + pos_na, 0);
        unsigned start_1100 = OFFSET(exp, offa10 + pos_na, 0);
        unsigned parity = alpha_pairs[pos_alpha].mid_parity;

        std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

        // unroll loop ?
        // this definitely needs to be vectorized

        // #pragma GCC unroll 0
        for (Offset pos_beta = 0; pos_beta < e_of_n; pos_beta++)
        {
            updateStateVector(sv, coeffs, start_0011 + pos_beta, start_1100 + pos_beta);
        }
    }

    // the niter_max values can vary greatly within a term
    // so even though the manhattan loop collapse introducces overall greater work
    // it should enable greater parallelism

    // #pragma omp parallel for
    //     for (uint64_t pos_alpha = 0; pos_alpha < num_parts; pos_alpha++)
    //     {
    //         unsigned niter_max = alpha_pairs[pos_alpha].niter;
    //         unsigned parity = alpha_pairs[pos_alpha].mid_parity;
    //         unsigned start_01 = alpha_pairs[pos_alpha].off01;
    //         unsigned start_10 = alpha_pairs[pos_alpha].off10;
    //         std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

    //         for (unsigned na = 0; na < niter_max; na++)
    //         {
    //             unsigned start_011 = OFFSET(exp, start_01 + na, 0);
    //             unsigned start_110 = OFFSET(exp, start_10 + na, 0);
    //             for (Offset pos_beta = 0; pos_beta < e_of_n; pos_beta++)
    //             {
    //                 unsigned idx011 = start_011 + pos_beta;
    //                 unsigned idx110 = start_110 + pos_beta;
    //                 // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx011, idx110);
    //                 // std::cout << idx011 << ", " << idx110 << std::endl;
    //                 updateStateVector(state_vector, coeffs[0], coeffs[1], idx011, idx110);
    //             }
    //         }
    //     }
}

inline void _updateBBBB(StateVector &sv, std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        std::vector<MixedPQRSLoopInfo> &beta_pairs, uint64_t num_parts_b, uint64_t exp)
{
    std::vector<unsigned> goalposts(num_parts_b + 1);
    goalposts[0] = 0;
    for (uint64_t i = 1; i <= num_parts_b; i++)
    {
        goalposts[i] = beta_pairs[i - 1].niter;
    }

    parallelPrefixSum(goalposts);

    uint64_t total_number_iter = goalposts[num_parts_b];

#pragma omp parallel for collapse(2)
    for (Offset pos_alpha = 0; pos_alpha < e_of_n; pos_alpha++)
    {
        // // #pragma omp for nowait
        for (uint64_t pos = 0; pos < total_number_iter; pos++)
        {
            unsigned a_start_0011 = OFFSET(exp, pos_alpha, 0);
            auto pos_beta_it = std::upper_bound(goalposts.begin(), goalposts.end(), pos);
            unsigned pos_beta = (pos_beta_it - goalposts.begin()) - 1;

            // find the remainder
            unsigned pos_nb = pos - goalposts[pos_beta];

            unsigned parity = beta_pairs[pos_beta].mid_parity;
            unsigned off01 = beta_pairs[pos_beta].off01;
            unsigned off10 = beta_pairs[pos_beta].off10;
            std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

            updateStateVector(sv, coeffs, a_start_0011 + off01 + pos_nb, a_start_0011 + off10 + pos_nb);
        }
    }

    // for / guided -> 1.2s
    // for / guided // collapse(2) -> 1.0s
    // for / dynamic(1) // collapse(2) -> 2.6s
    // for / dynamic(4) // collapse(2) -> 1.2s
    // for / dynamic(16) // collapse(2) -> 1.0s
    // for / static // collapse(2) -> 1.1s

    // // #pragma omp parallel for collapse(2) schedule(guided)
    // for (Offset pos_alpha = 0; pos_alpha < e_of_n; pos_alpha++)
    // {
    //     //// #pragma omp for nowait
    //     for (Offset pos_beta = 0; pos_beta < num_parts_b; pos_beta++)
    //     {
    //         unsigned a_start_0011 = OFFSET(exp, pos_alpha, 0);

    //         unsigned niter_max = beta_pairs[pos_beta].niter;
    //         unsigned parity = beta_pairs[pos_beta].mid_parity;
    //         unsigned off01 = beta_pairs[pos_beta].off01;
    //         unsigned off10 = beta_pairs[pos_beta].off10;
    //         std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

    //         unsigned start_0011 = a_start_0011 + off01;
    //         unsigned start_1100 = a_start_0011 + off10;

    //         for (unsigned nb = 0; nb < niter_max; nb++)
    //         {
    //             unsigned idx0011 = start_0011 + nb;
    //             unsigned idx1100 = start_1100 + nb;
    //             // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx0011, idx1100);
    //             updateStateVector(state_vector, coeffs[0], coeffs[1], idx0011, idx1100);
    //         }
    //     }
    // }
}

inline void _updateABBA(StateVector &sv, std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        std::vector<MixedPQRSLoopInfo> &alpha_pairs, std::vector<MixedPQRSLoopInfo> &beta_pairs, uint64_t num_parts_a, uint64_t num_parts_b, uint64_t exp)
{
    // manhattan loop collapse
    // 1. create array with all the goalposts
    // 2. simulate execution over the goalposts

    std::vector<unsigned> goalposts(num_parts_a + 1);
    goalposts[0] = 0;
    for (uint64_t i = 1; i <= num_parts_a; i++)
    {
        goalposts[i] = alpha_pairs[i - 1].niter;
    }

    parallelPrefixSum(goalposts);

    // now, simulate execution
    uint64_t total_number_iter = goalposts[num_parts_a];

#pragma omp parallel for
    for (uint64_t pos = 0; pos < total_number_iter; pos++)
    {
        // find the highest goalpost < pos
        auto pos_alpha_it = std::upper_bound(goalposts.begin(), goalposts.end(), pos);
        unsigned pos_alpha = (pos_alpha_it - goalposts.begin()) - 1;

        // find the remainder
        unsigned pos_na = pos - goalposts[pos_alpha];
        // std::cout << "A:" << pos_alpha << " " << pos_na << std::endl;

        uint64_t offa01 = alpha_pairs[pos_alpha].off01;
        uint64_t offa10 = alpha_pairs[pos_alpha].off10;

        unsigned start_0011 = OFFSET(exp, offa01 + pos_na, 0);
        unsigned start_1100 = OFFSET(exp, offa10 + pos_na, 0);
        unsigned a_parity = alpha_pairs[pos_alpha].mid_parity;

        // now, we should iterate through the nb info
        for (uint64_t pos_beta = 0; pos_beta < num_parts_b; ++pos_beta)
        {
            uint64_t offb01 = beta_pairs[pos_beta].off01;
            uint64_t offb10 = beta_pairs[pos_beta].off10;

            unsigned parity = 1 & (a_parity + beta_pairs[pos_beta].mid_parity);
            std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

            unsigned niter_max_b = beta_pairs[pos_beta].niter;

            Offset idx0011 = start_0011 + offb01;
            Offset idx1100 = start_1100 + offb10;

            // #pragma GCC unroll 0
            for (uint64_t nb = 0; nb < niter_max_b; ++nb)
            {
                // extern int nupdates;
                // nupdates += 1;
                // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx0011 + nb, idx1100 + nb);
                updateStateVector(sv, coeffs, idx0011 + nb, idx1100 + nb);
            }
        }
    }

    // // #pragma omp parallel for
    // for (pos_alpha = 0; pos_alpha < num_parts_a; ++pos_alpha)
    // {
    //     uint64_t offa01 = alpha_pairs[pos_alpha].off01;
    //     uint64_t offa10 = alpha_pairs[pos_alpha].off10;
    //     unsigned niter_max_a = alpha_pairs[pos_alpha].niter;

    //     for (uint64_t na = 0; na < niter_max_a; ++na)
    //     {
    //         unsigned start_0011 = OFFSET(exp, offa01 + na, 0);
    //         unsigned start_1100 = OFFSET(exp, offa10 + na, 0);

    //         for (pos_beta = 0; pos_beta < num_parts_b; ++pos_beta)
    //         {
    //             // grab their indices
    //             // flag - loop collapse (?)

    //             uint64_t offb01 = beta_pairs[pos_beta].off01;
    //             uint64_t offb10 = beta_pairs[pos_beta].off10;

    //             unsigned parity = 1 & (alpha_pairs[pos_alpha].mid_parity + beta_pairs[pos_beta].mid_parity);
    //             std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

    //             unsigned niter_max_b = beta_pairs[pos_beta].niter;

    //             // Note - invert when beta / alpha
    //             // Offset idx0011 = OFFSET(exp, offa01 + na, offb01);
    //             // Offset idx1100 = OFFSET(exp, offa10 + na, offb10);
    //             Offset idx0011 = start_0011 + offb01;
    //             Offset idx1100 = start_1100 + offb10;

    //             for (uint64_t nb = 0; nb < niter_max_b; ++nb)
    //             {
    //                 // extern int nupdates;
    //                 // nupdates += 1;
    //                 // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx0011 + nb, idx1100 + nb);
    //                 updateStateVector(state_vector, coeffs[0], coeffs[1], idx0011 + nb, idx1100 + nb);
    //             }
    //         }
    //     }
    // }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // extern double pqrs_time;
    // pqrs_time += diff.count();
}

inline void updatePQRS_AAAA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e2_of_n4);
    unsigned num_parts_a = discoverCCAA_BVs(alpha_pairs, p, q, r, s, e);

    auto start = std::chrono::high_resolution_clock::now();

    _updateAAAA(state_vector, even_coeffs, odd_coeffs, alpha_pairs, num_parts_a, exp);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
}

inline void updatePQRS_BBBB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> beta_pairs(e2_of_n4);
    unsigned num_parts_b = discoverCCAA_BVs(beta_pairs, p - n, q - n, r - n, s - n, e);
    _updateBBBB(state_vector, even_coeffs, odd_coeffs, beta_pairs, num_parts_b, exp);
}

inline void updatePQRS_ABBA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r, unsigned s)
{
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n2);
    unsigned num_parts_a = discoverCA_BVs(alpha_pairs, p, s, e);

    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n2);
    unsigned num_parts_b = discoverCA_BVs(beta_pairs, q - n, r - n, e);

    _updateABBA(state_vector, even_coeffs, odd_coeffs, alpha_pairs, beta_pairs, num_parts_a, num_parts_b, exp);
};

inline void updatePP_AA(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p)
{

    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n1);
    unsigned num_parts_a = discoverN_BVs(alpha_pairs, p, e);

#pragma omp parallel for
    for (uint64_t pos_alpha = 0; pos_alpha < num_parts_a; pos_alpha++)
    {
        unsigned niter = alpha_pairs[pos_alpha].niter;
        unsigned start_01 = alpha_pairs[pos_alpha].off01;
        for (unsigned na = 0; na < niter; na++)
        {
            unsigned idx01 = OFFSET(exp, start_01 + na, 0);
            for (Offset pos_beta = 0; pos_beta < e_of_n; pos_beta++)
            {
                state_vector[idx01 + pos_beta] *= coeff;
            }
        }
    }
}
inline void updatePP_BB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p)
{
    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n1);
    unsigned num_parts_b = discoverN_BVs(beta_pairs, p - n, e);

#pragma omp parallel for collapse(2)
    for (Offset pos_alpha = 0; pos_alpha < e_of_n; pos_alpha++)
    {
        for (uint64_t pos_beta = 0; pos_beta < num_parts_b; pos_beta++)
        {
            unsigned niter = beta_pairs[pos_beta].niter;
            unsigned idx01 = OFFSET(exp, pos_alpha, beta_pairs[pos_beta].off01);
            for (unsigned nb = 0; nb < niter; nb++)
            {
                state_vector[idx01 + nb] *= coeff;
            }
        }
    }
}

inline void updatePQ_AA(StateVector &state_vector,
                        std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        uint64_t exp, unsigned p, unsigned q)
{
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n2);
    unsigned num_parts_a = discoverCA_BVs(alpha_pairs, p, q, e);

// 0, 3 -> ?? 0__1 -> 0011, 0101, ... 

    for (uint64_t pos_alpha = 0; pos_alpha < num_parts_a; pos_alpha++)
    {
        unsigned niter = alpha_pairs[pos_alpha].niter;
        unsigned start_01 = alpha_pairs[pos_alpha].off01;
        unsigned start_10 = alpha_pairs[pos_alpha].off10;
        unsigned parity = alpha_pairs[pos_alpha].mid_parity;
        std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

        for (unsigned na = 0; na < niter; na++)
        {
            unsigned idx01 = OFFSET(exp, start_01 + na, 0);
            unsigned idx10 = OFFSET(exp, start_10 + na, 0);

            for (Offset pos_beta = 0; pos_beta < e_of_n; pos_beta++)
            {
                // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx01 + pos_beta, idx10 + pos_beta);
                updateStateVector(state_vector, coeffs, idx01 + pos_beta, idx10 + pos_beta);
            }
        }
    }
}
inline void updatePQ_BB(StateVector &state_vector,
                        std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                        uint64_t exp, unsigned p, unsigned q)
{
    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n2);
    unsigned num_parts_b = discoverCA_BVs(beta_pairs, p - n, q - n, e);

    for (Offset pos_alpha = 0; pos_alpha < e_of_n; pos_alpha++)
    {
        unsigned idx01 = OFFSET(exp, pos_alpha, 0);

        for (Offset pos_beta = 0; pos_beta < e1_of_n2; pos_beta++)
        {
            unsigned niter = beta_pairs[pos_beta].niter;
            unsigned start_01 = beta_pairs[pos_beta].off01;
            unsigned start_10 = beta_pairs[pos_beta].off10;
            unsigned parity = beta_pairs[pos_beta].mid_parity;
            std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

            unsigned idx01_b = idx01 + start_01;
            unsigned idx10_b = idx01 + start_10;

            for (unsigned nb = 0; nb < niter; nb++)
            {
                // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx01_b + nb, idx10_b + nb);
                updateStateVector(state_vector, coeffs, idx01_b + nb, idx10_b + nb);
            }
        }
    }
}

inline void updatePQQP_AA(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e2_of_n2);
    unsigned num_parts_a = discoverNN_BVs(alpha_pairs, p, q, e);

#pragma omp parallel for
    for (uint64_t pos_alpha = 0; pos_alpha < num_parts_a; pos_alpha++)
    {
        // std::cout << "NUMBER OF THREADS " << omp_get_num_threads() << std::endl;
        unsigned start_011 = alpha_pairs[pos_alpha].off01;
        unsigned niter = alpha_pairs[pos_alpha].niter;

        for (unsigned na = 0; na < niter; na++)
        {
            unsigned idx1111 = OFFSET(exp, start_011 + na, 0);

            for (Offset pos_beta = 0; pos_beta < e_of_n; pos_beta++)
            {
                state_vector[idx1111 + pos_beta] *= coeff;
            }
        }
    }
}
inline void updatePQQP_BB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> beta_pairs(e2_of_n2);
    unsigned num_parts_b = discoverNN_BVs(beta_pairs, p - n, q - n, e);

#pragma omp parallel for
    for (Offset pos_alpha = 0; pos_alpha < e_of_n; pos_alpha++)
    {
        unsigned idx1111 = OFFSET(exp, pos_alpha, 0);

        for (uint64_t pos_beta = 0; pos_beta < num_parts_b; pos_beta++)
        {
            unsigned niter = beta_pairs[pos_beta].niter;
            unsigned start_011 = beta_pairs[pos_beta].off01;

            for (unsigned nb = 0; nb < niter; nb++)
            {
                state_vector[idx1111 + start_011 + nb] *= coeff;
            }
        }
    }
}
inline void updatePQQP_AB(StateVector &state_vector, std::complex<double> &coeff, uint64_t exp, unsigned p, unsigned q)
{
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n1);
    unsigned num_parts_a = discoverN_BVs(alpha_pairs, p, e);

    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n1);
    unsigned num_parts_b = discoverN_BVs(beta_pairs, q - n, e);

#pragma omp parallel for
    for (uint64_t pos_alpha = 0; pos_alpha < num_parts_a; pos_alpha++)
    {
        unsigned start_011_a = alpha_pairs[pos_alpha].off01;
        unsigned niter_a = alpha_pairs[pos_alpha].niter;

        for (unsigned na = 0; na < niter_a; na++)
        {
            unsigned idx1111 = OFFSET(exp, start_011_a + na, 0);

            for (uint64_t pos_beta = 0; pos_beta < num_parts_b; pos_beta++)
            {
                unsigned start_011_b = beta_pairs[pos_beta].off01;
                unsigned niter_b = beta_pairs[pos_beta].niter;

                for (unsigned nb = 0; nb < niter_b; nb++)
                {
                    state_vector[idx1111 + start_011_b + nb] *= coeff;
                }
            }
        }
    }
}

inline void updatePQQR_AAAA(StateVector &sv,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e2_of_n3);
    unsigned num_parts = discoverCNA_BVs(alpha_pairs, p, q, r, e);

    _updateAAAA(sv, even_coeffs, odd_coeffs, alpha_pairs, num_parts, exp);
}
inline void updatePQQR_BBBB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r)
{
    if (e < 2)
    {
        return;
    }
    std::vector<MixedPQRSLoopInfo> beta_pairs(e2_of_n3);
    unsigned num_parts_b = discoverCNA_BVs(beta_pairs, p - n, q - n, r - n, e);

    _updateBBBB(state_vector, even_coeffs, odd_coeffs, beta_pairs, num_parts_b, exp);
}
inline void updatePQQR_ABBA(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r)
{
    // q in beta, pr in alpha
    // p,r / alpha / -----------------------
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n2);
    unsigned num_parts_a = discoverCA_BVs(alpha_pairs, p, r, e);

    // q / beta -----------
    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n1);
    unsigned num_parts_b = discoverN_BVs(beta_pairs, q - n, e);

    _updateABBA(state_vector, even_coeffs, odd_coeffs, alpha_pairs, beta_pairs, num_parts_a, num_parts_b, exp);
}

inline void updatePQQR_BAAB(StateVector &state_vector,
                            std::array<std::complex<double>, 4> &even_coeffs, std::array<std::complex<double>, 4> &odd_coeffs,
                            uint64_t exp, unsigned p, unsigned q, unsigned r)
{
    // q alpha
    std::vector<MixedPQRSLoopInfo> alpha_pairs(e1_of_n1);
    unsigned num_parts_a = discoverN_BVs(alpha_pairs, q, e);

    // pr beta -----------
    std::vector<MixedPQRSLoopInfo> beta_pairs(e1_of_n2);
    unsigned num_parts_b = discoverCA_BVs(beta_pairs, p - n, r - n, e);

#pragma omp parallel for
    for (uint64_t pos_alpha = 0; pos_alpha < num_parts_a; pos_alpha++)
    {
        unsigned niter_max_a = alpha_pairs[pos_alpha].niter;
        unsigned idx_a = alpha_pairs[pos_alpha].off01;

        for (unsigned na = 0; na < niter_max_a; na++)
        {
            unsigned a_start_011 = OFFSET(exp, idx_a + na, 0);

            for (uint64_t pos_beta = 0; pos_beta < num_parts_b; pos_beta++)
            {
                unsigned niter_max_b = beta_pairs[pos_beta].niter;
                unsigned parity = beta_pairs[pos_beta].mid_parity;
                unsigned start_01_b = beta_pairs[pos_beta].off01;
                unsigned start_10_b = beta_pairs[pos_beta].off10;
                std::array<std::complex<double>, 4> &coeffs = (parity & 1) == 0 ? even_coeffs : odd_coeffs;

                unsigned start_011 = start_01_b + a_start_011;
                unsigned start_110 = start_10_b + a_start_011;

                for (unsigned nb = 0; nb < niter_max_b; nb++)
                {
                    unsigned idx011 = start_011 + nb;
                    unsigned idx110 = start_110 + nb;
                    // result += updateStateVector(state_vector, coeffs[0], coeffs[1], idx011, idx110);
                    updateStateVector(state_vector, coeffs, idx011, idx110);
                }
            }
        }
    }
}

inline unsigned discoverCCAA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned q, unsigned r, unsigned s, const unsigned &e)
{
    unsigned idxs[4] = {p, q, r, s};
    std::sort(idxs, idxs + 4);

    // Prepare all of the alpha Bvs
    unsigned ctr;
    Offset num_parts = 0;
    BitVector bv;
    for (ctr = 0, bv = ((1ul << (e - 2)) - 1); ctr < alpha_pairs.size(); ++num_parts)
    {
        // we actually need to divide the number into five segments
        // | s | r | q | p | (hi, midhi, midmid, midlo, lo)
        unsigned lo = bv & ((1ul << idxs[0]) - 1);
        unsigned midlo = (bv & ((1ul << (idxs[1] - 1)) - 1)) >> idxs[0];
        unsigned midmid = (bv & ((1ul << (idxs[2] - 2)) - 1)) >> (idxs[1] - 1);
        unsigned midhi = (bv & ((1ul << (idxs[3] - 3)) - 1)) >> (idxs[2] - 2);
        unsigned hi = bv >> (idxs[3] - 3);

        // now, let's add our custom forced bits
        uint64_t bvb = lo | (midlo << (idxs[0] + 1)) | (midmid << (idxs[1] + 1)) | (midhi << (idxs[2] + 1)) | (hi << (idxs[3] + 1));

        Offset alp01 = OffsetComputer::rank(bvb | (1ul << r) | (1ul << s));
        Offset alp10 = OffsetComputer::rank(bvb | (1ul << q) | (1ul << p));

        alpha_pairs[num_parts].off01 = alp01;
        alpha_pairs[num_parts].off10 = alp10;
        alpha_pairs[num_parts].mid_parity = parity(midlo) + parity(midhi);

        // now, skip the appropriate number of bits
        unsigned niter = 1;
        if (lo != 0)
        {
            assert(is_power_of_2(lo + 1));
            unsigned lo_e = int_pop_count(lo);
            niter = nChoosek(idxs[0], lo_e);
        }
        assert(niter > 0);
        alpha_pairs[num_parts].niter = niter;

        // update trip count
        ctr += niter;
        bv = bv == 0 ? 1 : bv;
        for (uint64_t i = 0; i < niter; ++i)
        {
            bv = next_comb(bv);
        }
    }

    return num_parts;
}

inline unsigned discoverCNA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned q, unsigned r, const unsigned &e)
{
    assert(r > p);
    // Prepare all of the alpha Bvs
    unsigned idxs[3] = {p, q, r};
    std::sort(idxs, idxs + 3);

    unsigned ctr;
    Offset num_parts = 0;
    BitVector bv;
    for (ctr = 0, bv = ((1ul << (e - 2)) - 1); ctr < alpha_pairs.size(); ++num_parts)
    {
        // we actually need to divide the number into four segments
        // | r | q | p | (hi, midhi, midlo, lo)
        unsigned lo = bv & ((1ul << idxs[0]) - 1);
        unsigned midlo = (bv & ((1ul << (idxs[1] - 1)) - 1)) >> idxs[0];
        unsigned midhi = (bv & ((1ul << (idxs[2] - 2)) - 1)) >> (idxs[1] - 1);
        unsigned hi = bv >> (idxs[2] - 2);

        // now, let's add our custom forced bits
        uint64_t bvb = lo | (midlo << (idxs[0] + 1)) | (midhi << (idxs[1] + 1)) | (hi << (idxs[2] + 1));

        Offset alp011 = OffsetComputer::rank(bvb | (1ul << q) | (1ul << r));
        Offset alp110 = OffsetComputer::rank(bvb | (1ul << q) | (1ul << p));

        alpha_pairs[num_parts].off01 = alp011;
        alpha_pairs[num_parts].off10 = alp110;

        if (q > r)
        {
            alpha_pairs[num_parts].mid_parity = parity(midlo);
        }
        else if (q < p)
        {
            alpha_pairs[num_parts].mid_parity = parity(midhi);
        }
        else
        {
            alpha_pairs[num_parts].mid_parity = parity(midlo) + parity(midhi) + 1;
        }

        // now, skip the appropriate number of bits
        unsigned niter = 1;
        if (lo != 0)
        {
            assert(is_power_of_2(lo + 1));
            unsigned lo_e = int_pop_count(lo);
            niter = nChoosek(idxs[0], lo_e);
        }
        assert(niter > 0);
        alpha_pairs[num_parts].niter = niter;

        // update trip count
        ctr += niter;
        bv = bv == 0 ? 1 : bv;
        for (uint64_t i = 0; i < niter; ++i)
        {
            bv = next_comb(bv);
        }
    }

    return num_parts;
}

inline unsigned discoverCA_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned r, const unsigned &e)
{
    unsigned ctr;
    Offset num_parts = 0;
    BitVector bv;

    unsigned new_p = std::min(p, r);
    unsigned new_r = std::max(p, r);

    for (ctr = 0, bv = ((1ul << (e - 1)) - 1); ctr < alpha_pairs.size(); ++num_parts)
    {
        // we actually need to divide the number into four segments
        // | r | q | p | (hi, midhi, midlo, lo)
        unsigned lo = bv & ((1ul << new_p) - 1);
        unsigned mid = (bv & ((1ul << (new_r - 1)) - 1)) >> new_p;
        unsigned hi = bv >> (new_r - 1);

        // now, let's add our custom forced bits
        uint64_t bvb = lo | (mid << (new_p + 1)) | (hi << (new_r + 1));

        Offset alp011 = OffsetComputer::rank(bvb | (1ul << r));
        Offset alp110 = OffsetComputer::rank(bvb | (1ul << p));

        alpha_pairs[num_parts].off01 = alp011;
        alpha_pairs[num_parts].off10 = alp110;
        alpha_pairs[num_parts].mid_parity = parity(mid);

        // now, skip the appropriate number of bits
        unsigned niter = 1;
        if (lo != 0)
        {
            assert(is_power_of_2(lo + 1));
            unsigned lo_e = int_pop_count(lo);
            niter = nChoosek(new_p, lo_e);
        }
        assert(niter > 0);
        alpha_pairs[num_parts].niter = niter;

        // update trip count
        ctr += niter;
        bv = bv == 0 ? 1 : bv;
        for (uint64_t i = 0; i < niter; ++i)
        {
            bv = next_comb(bv);
        }
    }

    return num_parts;
}

inline unsigned discoverNN_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned p, unsigned r, const unsigned &e)
{

    assert(p < r);
    unsigned ctr;
    Offset num_parts = 0;
    BitVector bv;
    for (ctr = 0, bv = ((1ul << (e - 2)) - 1); ctr < alpha_pairs.size(); ++num_parts)
    {
        // we actually need to divide the number into four segments
        // | r | q | p | (hi, midhi, midlo, lo)
        unsigned lo = bv & ((1ul << p) - 1);
        unsigned mid = (bv & ((1ul << (r - 1)) - 1)) >> p;
        unsigned hi = bv >> (r - 1);

        // now, let's add our custom forced bits
        uint64_t bvb = lo | (mid << (p + 1)) | (hi << (r + 1));

        Offset alp011 = OffsetComputer::rank(bvb | (1ul << r) | (1ul << p));

        alpha_pairs[num_parts].off01 = alp011;

        // now, skip the appropriate number of bits
        unsigned niter = 1;
        if (lo != 0)
        {
            assert(is_power_of_2(lo + 1));
            unsigned lo_e = int_pop_count(lo);
            niter = nChoosek(p, lo_e);
        }
        assert(niter > 0);
        alpha_pairs[num_parts].niter = niter;

        // update trip count
        ctr += niter;
        bv = bv == 0 ? 1 : bv;
        for (uint64_t i = 0; i < niter; ++i)
        {
            bv = next_comb(bv);
        }
    }

    return num_parts;
}

inline unsigned discoverN_BVs(std::vector<MixedPQRSLoopInfo> &alpha_pairs, unsigned q, const unsigned &e)
{
    unsigned ctr;
    BitVector bv;
    Offset num_parts = 0;
    for (ctr = 0, bv = ((1ul << (e - 1)) - 1); ctr < alpha_pairs.size(); ++num_parts)
    {
        // we actually need to divide the number into 2 segments
        // | q |
        unsigned lo = bv & ((1ul << q) - 1);
        unsigned hi = bv >> q;

        // now, let's add our custom forced bits
        uint64_t bvb = lo | (hi << (q + 1));

        Offset alp011 = OffsetComputer::rank(bvb | (1ul << q));
        alpha_pairs[num_parts].off01 = alp011;
        alpha_pairs[num_parts].off10 = alp011;

        // now, skip the appropriate number of bits
        unsigned niter = 1;
        if (lo != 0)
        {
            assert(is_power_of_2(lo + 1));
            unsigned lo_e = int_pop_count(lo);
            niter = nChoosek(q, lo_e);
        }
        assert(niter > 0);
        alpha_pairs[num_parts].niter = niter;

        // update trip count
        ctr += niter;
        bv = bv == 0 ? 1 : bv;
        for (uint64_t i = 0; i < niter; ++i)
        {
            bv = next_comb(bv);
        }
    }

    return num_parts;
}

void inline parallelPrefixSum(std::vector<unsigned> &out)
{
    // Hille and steele algorithm
    // taken from here: https://www.cs.fsu.edu/~engelen/courses/HPC/Synchronous.pdf
    // there might be an alternative: https://software.intel.com/content/www/us/en/develop/articles/openmp-simd-for-inclusiveexclusive-scans.html
    int n = out.size();

    std::vector<unsigned> t(n);

    for (int j = 0; j < std::log2(n); j++)
    {
#pragma omp parallel for
        for (int i = 1 << j; i < n; i++)
        {
            t[i] = out[i] + out[i - (1 << j)];
        }

#pragma omp parallel for
        for (int i = 1 << j; i < n; i++)
        {
            out[i] = t[i];
        }
    }
}

#endif // TIME_EVOLUTION_MANUAL_HPP
