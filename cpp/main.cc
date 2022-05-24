#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "hamiltonian.hpp"
#include "state_vector.hpp"
#include "time_evolution.hpp"
#include "time_evolution_over_terms.hpp"
#include "gate.hpp"

double pqrs_time = 0;
uint64_t nupdates = 0;

inline double iterativeEstimation(Hamiltonian &h, StateVector &sv, int precision);
inline int iterativeEstimationHelper(Hamiltonian &h, StateVector sv, std::vector<bool> &prev);

double rdouble;
int main(int argc, char *argv[])
{
#if 0
  Hamiltonian h = Hamiltonian();

  for (int PRECISION = 1; PRECISION < 14; PRECISION ++) {
    StateVector sv = h.IngestBroombridge("../qsharp-verify/broombridge/Nick/H2O/H2O-15orb.yaml", PRECISION);
    h.IndexTerms(SpinIndex::HalfUp);
    h.SetTimestep(140);
    // h.PrintHamiltonianTerms(true);

    unsigned nchoosee = nChoosek(h.num_orbitals(), h.num_occupied());

    // Prep the ancilla for QPE
    std::cout.precision(17);
    std::cout << std::scientific;
    double scaling_factor = 1 / sqrt(powint(2, PRECISION));
    for (int idx = 0; idx < powint(nchoosee, 2); idx++)
    {
      sv[idx] *= scaling_factor;
    }

    std::cout << "Beginning evolution" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    time_evolve_over_terms(sv, h, PRECISION);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << std::endl;
    std::cout << "PRECISION: " << PRECISION << std::endl;
    std::cout << " | Time to run evolution: " << diff.count() << "\n";

    sv.qift();
    std::cout << "QIFT Finished" << std::endl;
    auto after = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> delt = after - end;
    std::cout << "Time to run QIFT: " << delt.count() << "\n";
    std::cout << "\n";
  }
#endif
#if 1
  // Three key steps with the Hamiltonian
  // Ingestion of the Broombridge File
  // Indexing the terms as either HalfUp or UpDown
  // and Setting the timestep

  assert(argc > 1);
  std::string FILENAME(argv[1]);

  unsigned PRECISION;
  unsigned STEPS;
  if (argc > 3)
  {
    PRECISION = std::stoi(argv[2]);
    STEPS = std::stoi(argv[3]);
  }
  else
  {
    PRECISION = 1;
    STEPS = 10;
  }

  // double STEP_SIZE = 1.0/10.0;
  // double STEP_SIZE = 1.0/20.0;
  double STEP_SIZE = 1.0/static_cast<double>(STEPS);

  std::cout.setf(std::ios::unitbuf);
  std::cout << "FILE: " << FILENAME << "| Precision: " << PRECISION << "\n";

  // Hamiltonian h = Hamiltonian();
  // StateVector sv = h.IngestBroombridge("../qsharp-verify/broombridge/" + FILENAME, 1);
  // h.IndexTerms(SpinIndex::HalfUp);
  // h.SetTimestep(STEP_SIZE);

  // double out = iterativeEstimation(h, sv, PRECISION);
  // std::cout << "PHASE: " << out << std::endl;

  Hamiltonian h = Hamiltonian();
  std::cout << "Building Hamiltonian" << std::endl;
  StateVector sv = h.IngestBroombridge("../qsharp-verify/broombridge/" + FILENAME, PRECISION);
  h.IndexTerms(SpinIndex::HalfUp);
  h.SetTimestep(STEP_SIZE);
  // h.PrintHamiltonianTerms(true);

  unsigned nchoosee = nChoosek(h.num_orbitals(), h.num_occupied());

  // Prep the ancilla for QPE
  std::cout.precision(17);
  std::cout << std::scientific;
  double scaling_factor = 1 / sqrt(powint(2, PRECISION));
  for (int idx = 0; idx < powint(nchoosee, 2); idx++)
  {
    sv[idx] *= scaling_factor;
  }

  std::cout << "Beginning evolution" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  time_evolve_over_terms(sv, h, PRECISION);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << std::endl;
  std::cout << "Time to run evolution: " << diff.count() << "\n";

  sv.qift();
  std::cout << "QIFT Finished" << std::endl;
  auto after = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> delt = after - end;
  std::cout << "Time to run QIFT: " << delt.count() << "\n";

  std::cout << "Running Phase Identification" << std::endl;
  auto probs = sv.IdentifyPhaseAmplitudes();
  auto last = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> delt2 = last - after;
  std::cout << "Time to run IPA: " << delt2.count() << std::endl;

  std::cout << "PQRS TIME:" << pqrs_time << "\n";
  std::cout << "nupdates:" << nupdates << "\n";

  std::cout << "----AFTER QIFT----\n";

  unsigned idx = 0;
  for (auto amp : probs)
  {
    std::cout.precision(17);
    std::cout << idx << "\t" << amp << "\n";
    idx++;
  }

  // now, perform a weighted average
  idx = 0;
  double total = 0.0;
  for (auto amp : probs)
  {
    total += idx * amp;
    idx++;
  }

  double phase_est = total / (1 << PRECISION);

  // now, we've measured the negative value.
  // We'd actually like to invert this phase and add the identity
  double energy = -1.0 * 2 * 2 * acos(0.0) * phase_est + h.GetIdentityTerm();
  std::cout << "FINAL ENERGY LEVEL ESTIMATE: " << energy << "\n";
  std::cout << "MOE: " << 2 * 2 * acos(0.0) / (STEP_SIZE * (1 << PRECISION)) << std::endl;

  return 0;
  YAML::Node config = YAML::LoadFile("PP.yml");
  std::cout << config << "\n";
  std::cout << "-------------------\n";
  std::cout << config["integral_sets"]["basis_set"]
            << "\n";
  return 0;
  //   std::ofstream fout("config.yaml");
  //   fout << config;
  assert(argc == 4);
  int n = atoi(argv[1]);
  int e = atoi(argv[2]);
  int c = atoi(argv[3]);
  uint64_t result = 0;

  std::cout << "n=" << n << " e=" << e << " c=" << c << "\n";

  // enumerate_power_set_minus_empty_set(n,e);

  //   unsigned long comb = ((1ul << e) - 1)<<(n-e);
  unsigned long comb = ((1ul << e) - 1);
  //   std::cout << "choose(n,e):" << nChoosek(n, e) << "\n";
#endif
#if 0
  uint64_t e_of_n = nChoosek(n, e);
  for (uint64_t i = 0; i < e_of_n; i++) {
    // std::cout << std::hex << std::bitset<64>{comb} << "\n";
    // comb = snoob(comb);
    // if(!next_combination(comb)) {
    //     std::cout<<"overflow of combination\n";
    // }
    // next_combination(comb);
    comb = next_comb(comb);
    result += comb;
  }
#endif
  //   std::cout<<"result="<<result<<"\n";
  //apply_gates(c, n, e);
  // Hamiltonian hamiltonian;
  // qpe(hamiltonian, c, 1);
  // return result + static_cast<int>(rdouble);
  return 0;
}