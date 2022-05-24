#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP

#include <yaml-cpp/yaml.h>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#include "state_vector.hpp"
#include "types.hpp"
#include "util.hpp"

// Relevant datatypes
typedef std::vector<std::pair<OneBodyTerm, double>> SingleBodyTermVec;
typedef std::vector<std::pair<TwoBodyTerm, double>> DoubleBodyTermVec;

typedef std::vector<IndexedSingleBodyTerm> IdxSingleVec;
typedef std::vector<IndexedDoubleBodyTerm> IdxDoubleVec;

typedef std::tuple<IdxSingleVec, IdxSingleVec, IdxDoubleVec, IdxDoubleVec, IdxDoubleVec> IdxSegmentedHamiltonian;

class Hamiltonian
{
public:
    // --------------------------------------------------------
    // COMPILER METHODS
    Hamiltonian() = default;

    // Add terms
    void AddOneBodyTerm(std::pair<int, int> targets, double coeff);
    void AddTwoBodyTerm(std::tuple<int, int, int, int> targets, double coeff);

    /**
     * @brief Ingest a Broombridge file into the Compiler
     * 
     * @param filepath
     */
    StateVector IngestBroombridge(std::string filepath, unsigned precision = 5);

    /**
     * @brief Prep the Compiler to be used by indexing
     * 
     * @param convention 
     */
    void IndexTerms(SpinIndex convention);

    /**
     * @brief Prints the Hamiltonian to std::cout
     * 
     * @param indexed - whether the terms should use indices or SpinOrbitals
     */
    void PrintHamiltonianTerms(bool indexed);

    double GetIdentityTerm() { return identity; }

    IdxSegmentedHamiltonian PrintIndexedSegmentedHamiltonianTerms();
    IdxSegmentedHamiltonian FilterIndexedSegmentedHamiltonianTerms();

    // NEW SIGNATURES WILL GO HERE

    std::vector<std::pair<unsigned, unsigned>> PrintSingleBodyTerms();
    std::vector<std::tuple<unsigned, unsigned, unsigned, unsigned>> PrintDoubleBodyTerms();

    // --------------------------------------------------------
    // HAMILTONIAN METHODS
    unsigned int num_orbitals() const { return max_number_of_orbitals; }
    unsigned int num_occupied() const { return num_occupied_; }
    void SetTimestep(double timestep) { t_ = timestep; }
    void SetNumOccupied(unsigned occupied) { num_occupied_ = occupied; }

    // TERMS
    /**
     * @brief return the exponentiated p+.p term. Called when the bit p is |1>.
     *
     * @param p orbital (0<= p < norbitals)
     * @return the exponentiated pp term
     */
    std::complex<double> pp_term(unsigned p) const;

    /**
     * @brief return the exponentiated q+.p and p+.q terms. Called for:
     *   - spin(p) == spin(q),
     *   - pq == 10 (for q+.p) and 01 (for p+.q) in that order
     *   - p != q
     *
     * @param p orbital
     * @param q orbital
     * @param parity parity of addressing bits p-1 ... q+1
     * @return 2x2 matrix that produces the updated values for pq=10 and 01
     */
    std::array<std::complex<double>, 4> pq_term(unsigned p, unsigned q, unsigned parity) const;

    /**
     * @brief return the exponentiated p+.q+.q.p term. Called for:
     *   - pq == 11
     *   - p != q
     *   - if spin(p) == spin(q), p>q
     *
     * @param p orbital
     * @param q orbital
     * @return std::complex<double> exponential p+.q+.q.p term
     */
    std::complex<double> pqqp_term(unsigned p, unsigned q) const;

    /**
     * @brief Get the exponentiated p+.q+.q.r and r.q+.q.p terms. Called for:
     *   - pqr = 110 (r+.q+.q.p) and 011 (p+.q+.q.r) terms, in that order
     *   - spin(p) == spin(r)
     *   - p > r
     *
     * @param p orbital
     * @param q orbital
     * @param r orbital
     * @param parity Parity of bits p-1...r+1, excluding q (if q in [p,q])
     * @return std::array<std::complex<double>,4> the 2x2 matrix for producing
     * updated values of pqr = 110 and 011.
     */
    std::array<std::complex<double>, 4> pqqr_term(unsigned p, unsigned q, unsigned r, unsigned parity) const;

    /**
     * @brief Exponentiated r+.s+.p.q and p+.q+.r.s terms. Called for:
     *   - pqrs = 1100 and 0011, in that order
     *   - spin(p) == spin(s)
     *   - spin(q) == spin(r)
     *   - p > q
     *   - r > s
     *   - p > s
     *
     * @param p orbital
     * @param q orbital
     * @param r orbital
     * @param s orbital
     * @param parity sum of parity for p-1..q+1 and r-1..s+1
     * @return 2x2 matrix of coefficients to produce new values of pqrs=1100 and
     * 0011, respectively.
     */
    std::array<std::complex<double>, 4> pqrs_term(unsigned p, unsigned q, unsigned r, unsigned s, unsigned parity) const;

private:
    // COMPILER METHODS / VARIABLES
    // Single values that are important for computation
    double identity;

    // number of orbitals before spin indexing
    // will be half of the number of qubits necessary
    // for example, 1^ 1_ 2^ 2_ will have max_number_of_orbitals = 2
    unsigned max_number_of_orbitals;
    unsigned int num_occupied_;

    // does this need OneBodyTermComp
    // FLAG- this needs to be looked into bc this will automatically order the terms by the comparator
    std::map<OneBodyTerm, double, OneBodyTermComp> one_body_terms;
    std::map<TwoBodyTerm, double, TwoBodyTermComp> two_body_terms;

    // Methods to update the internal data structures
    void _UpdateDataModel_OneBody(std::pair<OneBodyTerm, double> info);
    void _UpdateDataModel_TwoBody(std::pair<TwoBodyTerm, double> info);

    RaisingLoweringOperator ParsePolishNotation(std::string targets);

    // Methods used to convert the Broombridge format into spun terms
    std::vector<OneBodyTerm> _EnumerateSpinOrbitals_OneBody(std::pair<int, int> targets);
    std::vector<TwoBodyTerm> _EnumerateSpinOrbitals_TwoBody(std::tuple<int, int, int, int> targets);
    std::set<std::tuple<int, int, int, int>> _EnumerateTwoBodyPermutations(std::tuple<int, int, int, int> targets);

    // Reordering methods
    std::pair<TwoBodyTerm, double> _ReorderPQQPTerm(TwoBodyTerm targets, double coeff);
    std::pair<TwoBodyTerm, double> _ReorderPQQRTerm(TwoBodyTerm targets, double coeff);
    std::pair<TwoBodyTerm, double> _ReorderPQRSTerm(TwoBodyTerm targets, double coeff);

    // HAMILTONIAN VARIABLES / METHODS
    std::map<std::pair<unsigned, unsigned>, double> indexed_one_body_terms;
    std::map<std::tuple<unsigned, unsigned, unsigned, unsigned>, double> indexed_two_body_terms;

    // timestep
    double t_;

    // METHODS
    Spin spin(unsigned orbital) const { return orbital < max_number_of_orbitals ? Spin::Up : Spin::Down; }

    std::string broombridge_input_file_;
};

// UNIT TESTS---------------------

#include "doctest/doctest.h"

#include <iostream>
using namespace std::complex_literals;
TEST_CASE("PP term verification")
{
    // Ensure the Hamiltonian is working properly
    Hamiltonian h = Hamiltonian();
    h.SetTimestep(1.0);
    std::string filepath = "../qsharp-verify/broombridge/PP.yml";
    h.IngestBroombridge(filepath);
    h.IndexTerms(SpinIndex::HalfUp);

    auto coeff_00 = h.pp_term(0);
    auto coeff_11 = h.pp_term(1);
    auto truth = std::exp(-1i * 1.0);
    REQUIRE_EQ(coeff_00, truth);
    REQUIRE_EQ(coeff_11, truth);

    // Ensure the Hamiltonian handles empty terms correctly
    h = Hamiltonian();
    h.SetTimestep(1.0);
    h.AddOneBodyTerm(std::make_pair(2, 2), 1.0);
    h.IndexTerms(SpinIndex::HalfUp);

    coeff_00 = h.pp_term(0);
    truth = std::complex<double>(1.0);
    REQUIRE_EQ(coeff_00, truth);

    // Missing test cases:
    // - Does not verify that the Hamiltonian is ONLY (0, 0) and (1, 1)
    // - Relies upon SpinIndexing accuracy
}

TEST_CASE("PQ term verification")
{
    Hamiltonian h = Hamiltonian();
    std::string filepath = "../qsharp-verify/broombridge/PQ.yml";
    h.SetTimestep(1.0);
    h.IngestBroombridge(filepath);
    h.IndexTerms(SpinIndex::HalfUp);

    auto coeff_10 = h.pq_term(1, 0, 0);
    auto coeff_32 = h.pq_term(3, 2, 0);
    auto truth00 = std::cos(1.0);
    auto truth01 = std::sin(1.0) * (-1.0 * 1i);
    std::array<std::complex<double>, 4> truth{truth00, truth01, truth01, truth00};
    REQUIRE_EQ(coeff_10, truth);
    REQUIRE_EQ(coeff_32, truth);
}

TEST_CASE("PQQP term verification")
{
    Hamiltonian h = Hamiltonian();
    std::string filepath = "../qsharp-verify/broombridge/PQQP.yml";
    h.SetTimestep(1.0);
    h.IngestBroombridge(filepath);
    h.IndexTerms(SpinIndex::HalfUp);

    auto coeff10 = h.pqqp_term(0, 1);
    auto coeff21 = h.pqqp_term(1, 2);
    auto coeff30 = h.pqqp_term(0, 3);
    auto coeff32 = h.pqqp_term(2, 3);
    auto truth = std::exp(-1i * 1.0);
    REQUIRE_EQ(coeff10, truth);
    REQUIRE_EQ(coeff21, truth);
    REQUIRE_EQ(coeff30, truth);
    REQUIRE_EQ(coeff32, truth);
}

TEST_CASE("PQQR term verification")
{
    Hamiltonian h = Hamiltonian();
    std::string filepath = "../qsharp-verify/broombridge/PQQR.yml";
    h.SetTimestep(1.0);
    h.IngestBroombridge(filepath);
    h.IndexTerms(SpinIndex::HalfUp);

    auto coeff012 = h.pqqr_term(2, 1, 0, 0);
    auto coeff315 = h.pqqr_term(5, 1, 3, 0);
    auto coeff024 = h.pqqr_term(2, 4, 0, 0);
    auto coeff543 = h.pqqr_term(5, 4, 3, 0);
    auto truth00 = std::cos(1.0);
    auto truth01 = std::sin(1.0) * (-1.0 * 1i);
    std::array<std::complex<double>, 4> truth{truth00, truth01, truth01, truth00};
    REQUIRE_EQ(coeff012, truth);
    REQUIRE_EQ(coeff315, truth);
    REQUIRE_EQ(coeff024, truth);
    REQUIRE_EQ(coeff543, truth);
}

TEST_CASE("PQRS term verification")
{
    Hamiltonian h = Hamiltonian();
    std::string filepath = "../qsharp-verify/broombridge/PQRS.yml";
    h.SetTimestep(1.0);
    h.IngestBroombridge(filepath);
    h.IndexTerms(SpinIndex::HalfUp);

    // one potential iteration strategy is to execute the pre-spun terms:
    // for all p < q < r < s
    // execute pa qa sa ra | pa ra sa qa | pb qb sb rb | pb rb sb qb
    // mixed, too: pa qb sb ra | pa rb sb qa | pb qa sa rb | pb ra sa qb

    auto coeff0132 = h.pqrs_term(0, 1, 3, 2, 0); // 1^ 2^ 4^ 3^
    auto coeff0231 = h.pqrs_term(0, 2, 3, 1, 0); // 1^ 3^ 4^ 2^
    auto coeff0536 = h.pqrs_term(0, 5, 3, 6, 0); // 1^ 2_ 4^ 3_
    auto coeff0635 = h.pqrs_term(0, 6, 3, 5, 0); // 1^ 3_ 4^ 2_
    auto coeff4172 = h.pqrs_term(4, 1, 7, 2, 0); // 1_ 2^ 4_ 3^
    auto coeff4271 = h.pqrs_term(4, 2, 7, 1, 0); // 1_ 3^ 4_ 2^
    auto coeff4576 = h.pqrs_term(4, 5, 7, 6, 0); // 1_ 2_ 4_ 3_
    auto coeff4675 = h.pqrs_term(4, 6, 7, 5, 0); // 1_ 3_ 4_ 2_

    auto truth00 = std::cos(-1.0);
    auto truth01 = std::sin(-1.0) * (-1.0 * 1i);
    std::array<std::complex<double>, 4> truth{truth00, truth01, truth01, truth00};

    REQUIRE_EQ(coeff0132, truth);
    REQUIRE_EQ(coeff0231, truth);
    REQUIRE_EQ(coeff0536, truth);
    REQUIRE_EQ(coeff0635, truth);
    REQUIRE_EQ(coeff4172, truth);
    REQUIRE_EQ(coeff4271, truth);
    REQUIRE_EQ(coeff4576, truth);
    REQUIRE_EQ(coeff4675, truth);
}

#endif // HAMILTONIAN_HPP
