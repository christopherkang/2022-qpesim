#ifndef TIME_EVOLUTION_HPP_
#define TIME_EVOLUTION_HPP_

#include <complex>
#include <cstdint>
#include <vector>

#include "hamiltonian.hpp"
#include "offset_computer.hpp"
#include "state_vector.hpp"
#include "types.hpp"
#include "util.hpp"

uint64_t result = 0;
extern double rdouble;
inline void time_evolve(StateVector& state_vector, Hamiltonian& hamiltonian,
                      const unsigned precision) {
  const unsigned e = hamiltonian.num_occupied();
  const unsigned n = hamiltonian.num_orbitals();
  Offset e_of_n = nChoosek(n, e);
  size_t i = 0;

  auto OFFSET = [e_of_n](unsigned e_fac, unsigned pos_alpha, unsigned pos_beta) {
    return e_fac * e_of_n * e_of_n + pos_alpha * e_of_n + pos_beta;
  };

  std::vector<unsigned> ones_alpha, zeroes_alpha;
  std::vector<unsigned> ones_beta, zeroes_beta;
  BitVector bv_alpha, bv_beta;
  Offset ebase, pos_alpha, pos_beta;
  for (uint64_t exp = 0, ebase = 0; exp < (1ul << precision);
       exp++, ebase += e_of_n * e_of_n) {
    
    // consider making this constant
    hamiltonian.SetTimestep(exp);
    for (pos_alpha = 0, bv_alpha = ((1ul << e) - 1); pos_alpha < e_of_n;
         pos_alpha++, bv_alpha = next_comb(bv_alpha)) {
      compute_zeroes_and_ones(bv_alpha, n, zeroes_alpha, ones_alpha);
      OffsetComputer oc_alpha{n, bv_alpha, pos_alpha};
      for (pos_beta = 0, bv_beta = ((1ul << e) - 1); pos_beta < e_of_n;
           pos_beta++, bv_beta = next_comb(bv_beta)) {
        compute_zeroes_and_ones(bv_beta, n, zeroes_beta, ones_beta);
        OffsetComputer oc_beta{n, bv_beta, pos_beta};

        uint64_t pos = ebase + pos_alpha * e_of_n + pos_beta;
        uint64_t bv = (bv_alpha << n) | bv_beta;
        // std::cout << "bv=" << exp << " | " << std::bitset<4>{bv} << "\n";
        // std::cout << "bv=" << std::bitset<32>{bv} << "\n";
        // std::cout << std::bitset<8>(bv_alpha) << " | " << bv_beta << "\n";

        // //----------------------------pp terms--------------------------------
        // {
        //   std::complex<double> coeff{1};
        //   // pp-alpha term
        //   for (unsigned p_alpha : ones_alpha) {
        //     coeff *= hamiltonian.pp_term(p_alpha);
        //   }  // for p_alpha
        //   // pp-beta term
        //   for (unsigned p_beta : ones_beta) {
        //     coeff *= hamiltonian.pp_term(p_beta + n);
        //   }  // for p_beta
        //   state_vector[OFFSET(exp, pos_alpha, pos_beta)] *= coeff;
        // }  // pp terms

        //----------------------------pqqp terms------------------------------
        {
          std::complex<double> coeff{1};
          // pa-qa-qa-pa terms
          for (size_t ppos = 0, p_alpha = ones_alpha[ppos];
               ppos < ones_alpha.size(); ++ppos, p_alpha = ones_alpha[ppos]) {
            for (size_t qpos = ppos + 1, q_alpha = ones_alpha[qpos];
                 qpos < ones_alpha.size(); ++qpos, q_alpha = ones_alpha[qpos]) {
              coeff *= hamiltonian.pqqp_term(ppos, qpos);
            }  // for qpos
          }    // for ppos
          // pb-qb-qb-pb terms
          for (size_t ppos = 0, p_beta = ones_beta[ppos];
               ppos < ones_beta.size(); ++ppos, p_beta = ones_beta[ppos]) {
            for (size_t qpos = ppos + 1, q_beta = ones_beta[qpos];
                 qpos < ones_beta.size(); ++qpos, q_beta = ones_beta[qpos]) {
              coeff *= hamiltonian.pqqp_term(ppos + n, qpos + n);
            }  // for qpos
          }    // for ppos
          state_vector[OFFSET(exp, pos_alpha, pos_beta)] *= coeff;
        }  // pqqp terms

        auto find_lub = [](const std::vector<unsigned>& vec, unsigned val) {
            size_t i;
            for (i = 0; i < vec.size() && vec[i] <= val; i++) {
              // no-op
            }  // for i
            return i;
          };

        //----------------------------pq terms----------------------------
        {
          // pa-qa terms
          for (size_t ppos = 0, p_alpha = ones_alpha[ppos];
               ppos < ones_alpha.size(); ++ppos, p_alpha = ones_alpha[ppos]) {
            for (size_t qpos = ppos + 1, parity = 0, q_alpha = (bv_alpha >> qpos) & 1;
                 qpos < n; ++qpos, q_alpha = ((bv_alpha >> qpos) & 1)) {
              if (q_alpha == 1) {
                // OPTIMIZATION: COULD THIS BE A BOOLEAN?
                parity += 1;
              } else {
                auto coeffs = hamiltonian.pq_term(ppos, qpos, parity);
                auto v10 = state_vector[pos];
                uint64_t off01 = OFFSET(
                    exp, oc_alpha.rank_with_swap(ppos, qpos), pos_beta);
                std::cout << "a" << pos << " | " << off01 << "\n";
                std::cout << "mat: " << coeffs[0] << " // " << coeffs[1] << "\n"; 
                  
                std::cout << ppos << " " << qpos << "\n";
                auto v01 = state_vector[off01];
                std::complex<double> nv10, nv01;
                nv01 = coeffs[0] * v01 + coeffs[1] * v10;
                nv10 = coeffs[2] * v01 + coeffs[3] * v10;
                std::cout << "orig: " << v01 << ", " << v10 << "\n";
                std::cout << nv01 << ", " << nv10 << "\n";
                state_vector[pos] = nv10;
                state_vector[off01] = nv01;
                print_precise_array(state_vector);
              }  // if q_alpha
            }    // for qpos
          }      // for ppos

          // pb-qb terms
          for (size_t ppos = 0, p_beta = ones_beta[ppos];
               ppos < ones_beta.size(); ++ppos, p_beta = ones_beta[ppos]) {
            for (size_t qpos = ppos + 1, parity = 0, q_beta = (bv >> qpos) & 1;
                 qpos < n; ++qpos, q_beta = ((bv >> qpos) & 1)) {
              if (q_beta == 1) {
                parity += 1;
              } else {
                auto coeffs = hamiltonian.pq_term(ppos + n, qpos + n, parity);
                auto v10 = state_vector[pos];
                uint64_t off01 =
                    OFFSET(exp, pos_alpha, oc_beta.rank_with_swap(ppos, qpos));
                auto v01 = state_vector[off01];
                std::complex<double> nv10, nv01;
                nv01 = coeffs[0] * v01 + coeffs[1] * v10;
                nv10 = coeffs[2] * v01 + coeffs[3] * v10;
                std::cout << pos << " | " << off01 << "\n";
                std::cout << "mat: " << coeffs[0] << " // " << coeffs[1] << "\n"; 
                std::cout << "orig: " << v01 << ", " << v10 << "\n";
                std::cout << nv01 << ", " << nv10 << "\n";
                state_vector[pos] = nv10;
                state_vector[off01] = nv01;
                print_precise_array(state_vector);
              }  // if q_beta
            }    // for qpos
          }      // for ppos
        }        // pq terms

        // //----------------------------pqqr terms----------------------------
        // {
        //   auto find_lub = [](const std::vector<unsigned>& vec, unsigned val) {
        //     size_t i;
        //     for (i = 0; i < vec.size() && vec[i] <= val; i++) {
        //       // no-op
        //     }  // for i
        //     return i;
        //   };
        //   // pa-*-*-ra terms
        //   for (size_t ppos = 0, p_alpha = ones_alpha[ppos];
        //        ppos < ones_alpha.size(); ++ppos, p_alpha = ones_alpha[ppos]) {
        //     for (size_t rpos = ppos + 1, parity = 0, r_alpha = (bv >> rpos) & 1;
        //          rpos < n; ++rpos, r_alpha = ((bv >> rpos) & 1)) {
        //       if (r_alpha == 1) {
        //         parity += 1;
        //       } else {
        //         auto v110 = state_vector[pos];
        //         uint64_t off011 = OFFSET(
        //             e, oc_alpha.rank_with_swap(p_alpha, r_alpha), pos_beta);
        //         auto v011 = state_vector[off011];
        //         std::complex<double> nv110 = v110, nv011 = v011;
        //         for (auto q_alpha : ones_alpha) {
        //           auto coeffs =
        //               hamiltonian.pqqr_term(p_alpha, q_alpha, r_alpha, parity);
        //           nv110 = coeffs[0] * nv110 + coeffs[1] * nv011;
        //           nv011 = coeffs[2] * nv110 + coeffs[3] * nv011;
        //         }  // for q_alpha
        //         for (auto q_beta : ones_beta) {
        //           auto coeffs =
        //               hamiltonian.pqqr_term(p_alpha, q_beta, r_alpha, parity);
        //           nv110 = coeffs[0] * nv110 + coeffs[1] * nv011;
        //           nv011 = coeffs[2] * nv110 + coeffs[3] * nv011;
        //         }  // for q_beta
        //         state_vector[pos] = nv110;
        //         state_vector[off011] = nv011;
        //       }  // if r_alpha
        //     }    // for rpos
        //   }      // for ppos

        //   // pb-*-*-rb terms
        //   for (size_t ppos = 0, p_beta = ones_beta[ppos];
        //        ppos < ones_beta.size(); ++ppos, p_beta = ones_beta[ppos]) {
        //     for (size_t rpos = ppos + 1, parity = 0, r_beta = (bv >> rpos) & 1;
        //          rpos < n; ++rpos, r_beta = ((bv >> rpos) & 1)) {
        //       if (r_beta == 1) {
        //         parity += 1;
        //       } else {
        //         auto v110 = state_vector[pos];
        //         uint64_t off011 = OFFSET(
        //             e, pos_alpha, oc_beta.rank_with_swap(p_beta, r_beta));
        //         auto v011 = state_vector[off011];
        //         std::complex<double> nv110 = v110, nv011 = v011;
        //         for (auto q_alpha : ones_alpha) {
        //           auto coeffs =
        //               hamiltonian.pqqr_term(p_beta, q_alpha, r_beta, parity);
        //           nv110 = coeffs[0] * nv110 + coeffs[1] * nv011;
        //           nv011 = coeffs[2] * nv110 + coeffs[3] * nv011;
        //         }  // for q_alpha
        //         for (auto q_beta : ones_beta) {
        //           auto coeffs =
        //               hamiltonian.pqqr_term(p_beta, q_beta, r_beta, parity);
        //           nv110 = coeffs[0] * nv110 + coeffs[1] * nv011;
        //           nv011 = coeffs[2] * nv110 + coeffs[3] * nv011;
        //         }  // for q_beta
        //         state_vector[pos] = nv110;
        //         state_vector[off011] = nv011;
        //       }  // if r_alpha
        //     }    // for rpos
        //   }      // for ppos

        // }  // pqqr terms

        //----------------------------pqrs terms----------------------------
        // pa q* r* sa  x p* qa ra s*
        // let's start with just the pa qa ra sa case
        // where p$ q$ r s (but the hamiltonian will take it as p q s r)
        // for (size_t ppos = 0, p_alpha = ones_alpha[ppos]; ppos < ones_alpha.size(); ++ppos, p_alpha = ones_alpha[ppos]) {
        //   for (size_t qpos = ppos, q_alpha = ones_alpha[qpos]; qpos < ones_alpha.size(); ++qpos, q_alpha = ones_alpha[qpos]) {
        //     // we the only constraint on r is that it exceeds
        //     for (size_t rpos = 0, r_alpha = zeroes_alpha[rpos]; zeroes_alpha[rpos] > q_alpha; ++rpos, r_alpha = zeroes_alpha[rpos]) {
        //       //
        //     }
        //   }
        // }
#if 0
        // in one-body case: 1-2 form a mulliken pair
        // in two-body case: 1-4 and 2-3 form mulliken pais
        // spin is preserved in a mulliken pair

        // pa-pa (all p) -- single term
        // pb-pb (all p) -- single term

        // pa-qa term (p>q) -- handle pa-qa and qa-pa pair
        // pb-qb term (p>q) -- handle pb-qb and qb-pb pair

        // pa-qa-qa-pa term (all p > q) single terms
        // pb-qb-qb-pb term (all p > q) single terms
        // ??? - pa-qb-qb-pa term (all p, q) single terms

        // pa-qa-qa-ra term (p>r, forall q!=p and q!=r) single terms 
        // pa-qb-qb-ra term (p>r, forall q!=p and q!=r) single terms
        // pb-qa-qa-rb term (p>r, forall q!=p and q!=r) single terms
        // pb-qb-qb-rb term (p>r, forall q!=p and q!=r) single terms

        // pa-qa-ra-sa term (p>q, r>s, p>s) handle pqrs and rspq pairs 
        // pb-qb-rb-sb term (p>q, r>s, p>s) handle pqrs and rspq pairs

        for (unsigned ppos = 0; ppos < ones.size(); ppos++) {
          auto p = ones[ppos];
          // apply here -- pp term (pbit = 1)
          uint64_t off_1 = pos;
          rdouble += state_vector[ebase + off_1] * state_vector[ebase + off_1];
          unsigned qpos = 0;
          for (qpos = 0; qpos < zeroes.size() && zeroes[qpos] < ones[ppos];
               qpos++) {
            // no-op
          }
          for (; qpos < zeroes.size(); qpos++) {
            auto q = zeroes[qpos];
            // apply here -- pq term (pq = 10, 01)
            uint64_t off_10 = pos;
            uint64_t off_01 = oc.shift_rank_ij(p, q);
            rdouble +=
                state_vector[ebase + off_10] * state_vector[ebase + off_01];
            result += off_10 ^ off_01;
          }
        }
        uint64_t off_1100 = pos;
        for (unsigned ppos = 0; ppos < ones.size(); ppos++) {
          auto p = ones[ppos];
          for (unsigned qpos = ppos + 1; qpos < ones.size(); qpos++) {
            auto q = ones[qpos];
            // apply here -- pqqp term (pq=11)
            unsigned rpos = 0;
            for (rpos = 0; rpos < zeroes.size() && zeroes[rpos] < ones[ppos];
                 rpos++) {
              // no-op
            }
            for (; rpos < zeroes.size(); rpos++) {
              auto r = zeroes[rpos];
              // apply here -- pqqr term (pqr = 011, 110, 101)
              uint64_t off_110 = pos;
              //   uint64_t off_011 = shift_rank_ij(off_110, p, r);
              //   uint64_t off_101 = shift_rank_ij(off_110, q, r);
              uint64_t off_011 = find_rank(bv ^ (1ul << p) ^ (1ul << r));
              uint64_t off_101 = find_rank(bv ^ (1ul << q) ^ (1ul << r));
              rdouble += state_vector[ebase + off_110] *
                         state_vector[ebase + off_011] *
                         state_vector[ebase + off_101];
              result += off_110 ^ off_011 - off_101;
              for (unsigned spos = rpos + 1; spos < zeroes.size(); spos++) {
                auto s = zeroes[spos];
                uint64_t off_0011 = oc.shift_rank_ijkl(p, q, r, s);
                rdouble += state_vector.at(ebase + off_1100) *
                           state_vector.at(ebase + off_0011);
                result += off_1100 ^ off_0011;
              }
            }
          }
        }
#endif
      }
    }
  }
}

inline void qpe(Hamiltonian& hamiltonian, unsigned precision,
                unsigned num_time_steps) {
  StateVector state_vector{precision, hamiltonian.num_orbitals(),
                           hamiltonian.num_occupied()};
  // random initialization to disable optimizations
  unsigned i = 0;
  for (auto& v : state_vector) {
    v = -i + i * i;
    i++;
  }
  time_evolve(state_vector, hamiltonian, precision);
  // QIFT
  state_vector.qift();
}

#include "doctest/doctest.h"

TEST_CASE("PP Term") {
  const unsigned precision = 7;
  StateVector s = StateVector(precision, 1, 1);
  Hamiltonian h = Hamiltonian();
  h.IngestBroombridge("../qsharp-verify/broombridge/PP.yml");
  h.IndexTerms(SpinIndex::HalfUp);
  h.SetNumOccupied(1);
  // h.PrintHamiltonianTerms(true);

  // start the state in 0|1|1
  s[0] = 1.0;

  // Prep the ancilla for QPE
  for (unsigned idx = 0; idx < precision; idx++) {
    s.apply_gate(Hadamard(), idx);
  }

  time_evolve(s, h, precision);
  s.qift();
  auto probs = s.IdentifyPhaseAmplitudes();

  // for the PP term, we already know the phase that should be achieved is ~87/128
  // The Q# output is 0.15625 * 2pi for the phase of e^{it -h_pp/2 Z_0 Z_1}
  // We also need the phase of e^{-it h_pp / 2} | p=0, 1
  // Or e^{-it 1/2} e^{-it 1/2} = e^{-it}
  // Thus, the new phase should be -Qsharp - 1 -> -0.15625 * 2pi - 1 = -1.987 -> 4.30
  // 4.30 = 2pi (0.684595058) -> 0.684595058 ~ 87/128
  REQUIRE_GT(probs[87], 0.5);
}

TEST_CASE("PQ Term") {
  const unsigned precision = 7;
  StateVector s = StateVector(precision, 2, 1);
  Hamiltonian h = Hamiltonian();
  h.IngestBroombridge("../qsharp-verify/broombridge/PP.yml");
  h.IndexTerms(SpinIndex::HalfUp);
  h.SetNumOccupied(1);
  // h.PrintHamiltonianTerms(true);

  // start the state in 0|10|01
  // we'd like (2a)+ and (1b)+, so the SECOND alpha and the FIRST beta
  s[2] = 1.0;

  // Prep the ancilla for QPE
  for (unsigned idx = 0; idx < precision; idx++) {
    s.apply_gate(Hadamard(), idx);
  }

  time_evolve(s, h, precision);
  s.qift();
  auto probs = s.IdentifyPhaseAmplitudes();

  // for the PQ term, we can simply reference the Q# output
  REQUIRE_GT(probs[87], 0.5);
}

#endif  // TIME_EVOLUTION_HPP_
