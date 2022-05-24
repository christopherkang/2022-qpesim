#include "hamiltonian.hpp"
#include <cassert>
#include <tuple>
#include <regex>
#include "state_vector.hpp"
#include "offset_computer.hpp"

StateVector Hamiltonian::IngestBroombridge(std::string filepath, unsigned precision)
{
    broombridge_input_file_ = filepath;
    YAML::Node broombridge = YAML::LoadFile(filepath);

    const auto &integrals = broombridge["integral_sets"][0];
    const auto &hamiltonian = integrals["hamiltonian"];
    identity = integrals["coulomb_repulsion"]["value"].as<double>() + integrals["energy_offset"]["value"].as<double>();

    // this relies upon metadata, which may not be provided
    // max_number_of_orbitals = integrals["n_orbitals"].as<unsigned>();
    // num_occupied_ = integrals["n_electrons"].as<unsigned>() / 2;

    // One-Body Terms
    const auto &one_body_terms = hamiltonian["one_electron_integrals"];

    for (const auto &term : one_body_terms["values"])
    {
        unsigned p = term[0].as<unsigned>();
        unsigned q = term[1].as<unsigned>();
        double coeff = term[2].as<double>();
        if (q > p)
        {
            std::swap(p, q);
        }
        AddOneBodyTerm(std::make_pair(p, q), coeff);

        unsigned max_num = std::max(p, q);
        max_number_of_orbitals = std::max(max_num, max_number_of_orbitals);
    }

    // Two-Body Terms
    const auto &two_body_terms = hamiltonian["two_electron_integrals"];
    for (const auto &term : two_body_terms["values"])
    {
        unsigned i = term[0].as<unsigned>();
        unsigned l = term[1].as<unsigned>();
        unsigned j = term[2].as<unsigned>();
        unsigned k = term[3].as<unsigned>();
        double coeff = term[4].as<double>();
        AddTwoBodyTerm(std::make_tuple(i, j, k, l), coeff);

        unsigned max_num = std::max(std::max(i, j), std::max(k, l));
        max_number_of_orbitals = std::max(max_number_of_orbitals, max_num);
    }

    // STATE PREPARATION ROUTINE
    const auto initial_state_data = integrals["initial_state_suggestions"];

    const auto first_state = initial_state_data[0]["state"];
    const auto data = first_state["superposition"];

    std::vector<std::pair<unsigned, double>> state_prep_data;
    for (auto basis : data)
    {
        unsigned num_of_electrons = 0;
        // we want to add each basis to our state vector
        std::vector<RaisingLoweringOperator> operators;

        // now, lets ingest each of the RaisingLowering operators
        for (auto idx = std::next(basis.begin()); idx != basis.end(); idx++)
        {
            if (std::next(idx) != basis.end())
            {
                operators.push_back(ParsePolishNotation(idx->as<std::string>()));
            }
        }

        // we iterate in reverse to represent matrix multiplication
        unsigned bv = 0;
        double coeff = basis[0].as<double>();
        while (!operators.empty())
        {
            RaisingLoweringOperator op = operators.back();
            unsigned orb = op.orbital;
            Spin s = op.spin;
            bool creation = op.raising;

            unsigned idx = s == Spin::Up ? orb : orb + max_number_of_orbitals;
            if (creation)
            {
                // we are adding. ensure the electron isn't already present
                assert(((bv >> idx) % 2) == 0);
                bv += (1 << idx);
                num_of_electrons += 1;
            }
            else
            {
                // we are deleting an electron. ensure there is an electron to delete:
                assert(((bv >> idx) & 1) == 1);
                bv -= (1 << idx);
                num_of_electrons -= 1;
            }
            // regardless, we need to add the parity factor
            unsigned tmp = bv;
            for (unsigned pos = 0; pos < idx; pos++, tmp >>= 1)
            {
                if ((tmp & 1) == 1)
                {
                    coeff *= -1.0;
                }
            }
            operators.pop_back();
        }

        // there is an equal number of electrons in each alpha/beta class
        // so we need to divide the total number of electrons by 2
        num_occupied_ = num_of_electrons / 2;

        std::cout << std::bitset<8>(bv) << "\n";
        // now, we need to split this into alpha/beta
        unsigned bv_beta = bv >> max_number_of_orbitals;
        unsigned bv_alpha = bv & ((1 << max_number_of_orbitals) - 1);

        // and grab the idx of each of them
        Offset oc_b = OffsetComputer::rank(bv_beta);
        Offset oc_a = OffsetComputer::rank(bv_alpha);

        unsigned address = nChoosek(max_number_of_orbitals, num_occupied_) * oc_a + oc_b;
        state_prep_data.push_back(std::make_pair(address, coeff));
    }

    StateVector sv = StateVector(precision, max_number_of_orbitals, num_occupied_);

    #pragma omp parallel for
    for (auto comp_basis : state_prep_data)
    {
        unsigned address;
        double coeff;
        std::tie(address, coeff) = comp_basis;
        sv[address] = coeff;
    }

    print_precise_array(sv);

    return sv;
}

RaisingLoweringOperator Hamiltonian::ParsePolishNotation(std::string targets)
{
    std::regex pattern(R"(\((\d+)([ab])\)(\+*))");
    std::smatch m;
    std::regex_match(targets, m, pattern);

    // now, since the input was (1a)+, for instance, we are able to create the SpinOrbitals
    bool creating = m[3] == "+" ? true : false;
    Spin s = m[2] == "a" ? Spin::Up : Spin::Down;

    // recall: the broombridge format is 1-indexed, but we are 0-indexed
    return RaisingLoweringOperator{std::stoi(m[1]) - 1, s, creating};
}

void Hamiltonian::IndexTerms(SpinIndex convention = SpinIndex::HalfUp)
{
    // Convert data model into Indexed data model
    // One-Body Term conversion
    for (std::pair<OneBodyTerm, double> term : one_body_terms)
    {
        SpinOrbital p, q;
        OneBodyTerm targets;
        double coeff;
        std::tie(targets, coeff) = term;
        std::tie(p, q) = targets;

        int p_idx, q_idx;
        p_idx = p.ConvertToInt(convention, max_number_of_orbitals);
        q_idx = q.ConvertToInt(convention, max_number_of_orbitals);

        indexed_one_body_terms.insert(std::make_pair(std::make_pair(p_idx, q_idx), coeff));
    }

    // Two-Body Term conversion
    for (std::pair<TwoBodyTerm, double> term : two_body_terms)
    {
        SpinOrbital p, q, r, s;
        TwoBodyTerm targets;
        double coeff;

        std::tie(targets, coeff) = term;
        std::tie(p, q, r, s) = targets;

        int p_idx, q_idx, r_idx, s_idx;
        p_idx = p.ConvertToInt(convention, max_number_of_orbitals);
        q_idx = q.ConvertToInt(convention, max_number_of_orbitals);
        r_idx = r.ConvertToInt(convention, max_number_of_orbitals);
        s_idx = s.ConvertToInt(convention, max_number_of_orbitals);

        indexed_two_body_terms.insert(std::make_pair(std::make_tuple(p_idx, q_idx, r_idx, s_idx), coeff));
    }
}

// FLAG - THE PRINT HAMILTONIAN TERMS
// AND RETURN SEGMENTED HAMILTONIAN METHODS
// ARE SUPER SIMILAR AND COULD LIKELY BE COMPRESSED

void Hamiltonian::PrintHamiltonianTerms(bool indexed = false)
{
    // Print out the data model in different ways
    // Depending on whether the user wants indexing
    if (indexed)
    {
        for (auto it = indexed_one_body_terms.cbegin(); it != indexed_one_body_terms.cend(); it++)
        {
            std::pair<int, int> targets = it->first;
            double coeff = it->second;

            int p, q;
            std::tie(p, q) = targets;

            std::cout << p << " " << q << " " << coeff << "\n";
        }

        for (auto it = indexed_two_body_terms.cbegin(); it != indexed_two_body_terms.cend(); it++)
        {
            std::tuple<int, int, int, int> targets = it->first;
            double coeff = it->second;

            int p, q, r, s;
            std::tie(p, q, r, s) = targets;

            std::cout << p << " " << q << " " << r << " " << s << " " << coeff << "\n";
        }
    }
    else
    {
        for (auto it = one_body_terms.cbegin(); it != one_body_terms.cend(); it++)
        {
            OneBodyTerm targets = it->first;
            SpinOrbital p, q;
            std::tie(p, q) = targets;

            std::cout << p << " " << q << " " << it->second << "\n";
        }

        for (auto it = two_body_terms.cbegin(); it != two_body_terms.cend(); it++)
        {
            TwoBodyTerm targets = it->first;
            SpinOrbital p, q, r, s;
            std::tie(p, q, r, s) = targets;

            std::cout << p << " " << q << " " << r << " " << s << " " << it->second << "\n";
        }
    }
}

IdxSegmentedHamiltonian Hamiltonian::PrintIndexedSegmentedHamiltonianTerms()
{
    std::vector<IndexedSingleBodyTerm> pp, pq;
    std::vector<IndexedDoubleBodyTerm> pqqp, pqqr, pqrs;

    for (IndexedSingleBodyTerm term : indexed_one_body_terms)
    {
        std::pair<int, int> targets = term.first;
        int p, q;
        std::tie(p, q) = targets;

        if (p == q)
        {
            pp.push_back(term);
        }
        else
        {
            pq.push_back(term);
        }
    }

    for (IndexedDoubleBodyTerm term : indexed_two_body_terms)
    {
        std::tuple<int, int, int, int> targets = term.first;
        int p, q, r, s;
        std::tie(p, q, r, s) = targets;

        if (q == r)
        {
            // we may have a pqqp or pqqr term
            if (p == s)
            {
                pqqp.push_back(term);
            }
            else
            {
                pqqr.push_back(term);
            }
        }
        else
        {
            // we have a pqrs term
            pqrs.push_back(term);
        }
    }
    return std::make_tuple(pp, pq, pqqp, pqqr, pqrs);
}

void Hamiltonian::AddOneBodyTerm(std::pair<int, int> targets, double coeff)
{
    // Nothing should be executed if the coefficient is 0
    if (coeff == 0)
    {
        return;
    }

    unsigned int p, q;
    std::tie(p, q) = targets;

    assert(p >= q); // Broombridge assumes p >= q? Something to verify

    // max_number_of_orbitals = std::max(p, max_number_of_orbitals);

    // Iterate over the Spin orbitals
    std::vector<OneBodyTerm> orbitalPermutations = _EnumerateSpinOrbitals_OneBody(targets);

    for (OneBodyTerm pair : orbitalPermutations)
    {
        _UpdateDataModel_OneBody(std::make_pair(pair, coeff));
    }
}

void Hamiltonian::AddTwoBodyTerm(std::tuple<int, int, int, int> targets, double coeff)
{
    // We should not evaluate if the coefficient is 0
    if (coeff == 0)
    {
        return;
    }
    int p, q, r, s;
    std::tie(p, q, r, s) = targets;

    // unsigned int max_target_orbital = std::max(std::max(p, q), std::max(r, s));
    // max_number_of_orbitals = std::max(max_target_orbital, max_number_of_orbitals);

    if ((p == q) && (q == r) && (r == s))
    {
        // We are in the case of PPPP
        // Which we can shortcut:
        // As the only potential term is
        // P^ P_ P_ P^
        // After symmetries
        SpinOrbital p_up = SpinOrbital(p, Spin::Up);
        SpinOrbital p_down = SpinOrbital(p, Spin::Down);

        _UpdateDataModel_TwoBody(std::make_pair(TwoBodyTerm(std::make_tuple(p_up, p_down, p_down, p_up)), coeff));
        return;
    }

    std::set<std::tuple<int, int, int, int>> permutations = _EnumerateTwoBodyPermutations(targets);

    for (std::tuple<int, int, int, int> permuted : permutations)
    {
        std::vector<TwoBodyTerm> spunPermuted = _EnumerateSpinOrbitals_TwoBody(permuted);

        for (TwoBodyTerm combination : spunPermuted)
        {
            std::pair<TwoBodyTerm, double> final_term;

            // calculate the number of unique spinorbitals involved
            // to identify the relevant term
            SpinOrbital ps, qs, rs, ss;
            std::tie(ps, qs, rs, ss) = combination;

            std::set<SpinOrbital> uniqueOrbitals;
            uniqueOrbitals.insert(ps);
            uniqueOrbitals.insert(qs);
            uniqueOrbitals.insert(rs);
            uniqueOrbitals.insert(ss);

            int unique = uniqueOrbitals.size();

            if (unique == 2)
            {
                // PQQP term
                final_term = Hamiltonian::_ReorderPQQPTerm(combination, coeff);
            }
            else if (unique == 3)
            {
                // PQQR term
                final_term = Hamiltonian::_ReorderPQQRTerm(combination, coeff);
            }
            else if (unique == 4)
            {
                // PQRS term
                final_term = Hamiltonian::_ReorderPQRSTerm(combination, coeff);
            }
            else
            {
                throw "Unknown type not caught";
            }

            _UpdateDataModel_TwoBody(final_term);
        }
    }
}

std::set<std::tuple<int, int, int, int>> Hamiltonian::_EnumerateTwoBodyPermutations(std::tuple<int, int, int, int> targets)
{
    int i, j, k, l;
    std::tie(i, j, k, l) = targets;

    std::set<std::tuple<int, int, int, int>> tmp;

    tmp.insert(std::tuple<int, int, int, int>(i, j, k, l));
    tmp.insert(std::tuple<int, int, int, int>(l, j, k, i));
    tmp.insert(std::tuple<int, int, int, int>(i, k, j, l));
    tmp.insert(std::tuple<int, int, int, int>(l, k, j, i));
    tmp.insert(std::tuple<int, int, int, int>(j, i, l, k));
    tmp.insert(std::tuple<int, int, int, int>(k, i, l, j));
    tmp.insert(std::tuple<int, int, int, int>(j, l, i, k));
    tmp.insert(std::tuple<int, int, int, int>(k, l, i, j));

    return tmp;
}

// SPIN ORBITAL ENUMERATION------------------

std::vector<OneBodyTerm> Hamiltonian::_EnumerateSpinOrbitals_OneBody(std::pair<int, int> targets)
{
    std::vector<OneBodyTerm> out;

    int p, q;
    std::tie(p, q) = targets;

    std::vector<Spin> spins = {Spin::Up, Spin::Down};
    for (Spin direction : spins)
    {
        SpinOrbital spun_p = SpinOrbital(p, direction);
        SpinOrbital spun_q = SpinOrbital(q, direction);
        OneBodyTerm spunTerm = std::make_pair(spun_p, spun_q);
        out.push_back(spunTerm);
    }

    return out;
}

std::vector<TwoBodyTerm> Hamiltonian::_EnumerateSpinOrbitals_TwoBody(std::tuple<int, int, int, int> targets)
{
    std::vector<TwoBodyTerm> out;

    int p, q, r, s;
    std::tie(p, q, r, s) = targets;

    std::vector<Spin> spins = {Spin::Up, Spin::Down};
    for (Spin direction_inner : spins)
    {
        for (Spin direction_outer : spins)
        {
            SpinOrbital spun_p, spun_q, spun_r, spun_s;
            spun_p = SpinOrbital(p, direction_outer);
            spun_q = SpinOrbital(q, direction_inner);
            spun_r = SpinOrbital(r, direction_inner);
            spun_s = SpinOrbital(s, direction_outer);
            TwoBodyTerm spunTerm = std::make_tuple(spun_p, spun_q, spun_r, spun_s);
            out.push_back(spunTerm);
        }
    }

    return out;
}

// REORDERING TWO-BODY TERMS------------------

std::pair<TwoBodyTerm, double> Hamiltonian::_ReorderPQQPTerm(TwoBodyTerm targets, double coeff)
{
    SpinOrbital p, q, r, s;
    std::tie(p, q, r, s) = targets;

    // check for a PPQQ term, in which we may immediately exit
    if (r == s || p == q)
    {
        TwoBodyTerm trash;
        return std::make_pair(trash, 0.0);
    }

    const double PQQP_COEFF_ADJUSTMENT = 2.0;
    double new_coeff = coeff / PQQP_COEFF_ADJUSTMENT;

    // let's check for PQPQ
    if (p == r)
    {
        std::swap(r, s);
        new_coeff *= -1.0;
    }

    // now, we have PQQP, so let's verify P < Q
    if (!(p < q))
    {
        std::swap(p, q);
        std::swap(r, s);
    }

    assert((p == s) and (q == r));

    return std::make_pair(TwoBodyTerm(std::make_tuple(p, q, r, s)), new_coeff);
}

std::pair<TwoBodyTerm, double> Hamiltonian::_ReorderPQQRTerm(TwoBodyTerm targets, double coeff)
{
    SpinOrbital p, q, r, s;
    std::tie(p, q, r, s) = targets;

    // eliminate the PRQQ or QQPR terms
    if ((p == q) || (r == s))
    {
        TwoBodyTerm trash;
        return std::make_pair(trash, 0.0);
    }

    const double PQQR_COEFF_ADJUSTMENT = 4.0;
    double new_coeff = coeff / PQQR_COEFF_ADJUSTMENT;

    if (q == r)
    {
        // good to go
    }
    else if (q == s)
    {
        // PQRQ
        new_coeff *= -1.0;
        std::swap(r, s);
    }
    else if (p == r)
    {
        // QPQR
        new_coeff *= -1.0;
        std::swap(p, q);
    }
    else if (p == s)
    {
        // QPRQ
        std::swap(p, q);
        std::swap(r, s);
    }
    else
    {
        throw "Unknown PQQR term";
    }

    // verify P < S
    if (s < p)
    {
        std::swap(p, s);
    }

    return std::make_pair(TwoBodyTerm(std::make_tuple(p, q, r, s)), new_coeff);
}

std::pair<TwoBodyTerm, double> Hamiltonian::_ReorderPQRSTerm(TwoBodyTerm targets, double coeff)
{
    SpinOrbital p, q, r, s;
    std::tie(p, q, r, s) = targets;

    unsigned alphas = 0;

    alphas += p.spin == Spin::Up ? 1 : 0;
    alphas += q.spin == Spin::Up ? 1 : 0;
    alphas += r.spin == Spin::Up ? 1 : 0;
    alphas += s.spin == Spin::Up ? 1 : 0;

    assert((alphas % 2) == 0);

    const double PQRS_COEFF_ADJUSTMENT = 4.0;
    double new_coeff = coeff / PQRS_COEFF_ADJUSTMENT;

    if (std::min(p, q) < std::min(r, s))
    {
        // NoOp
    }
    else
    {
        // we need to rearrange pqrs -> rspq
        // this can be done without coefficient cost
        std::swap(p, r);
        std::swap(q, s);
    }

    // now, we'd like to reorder the terms to be in the canonical ordering
    // First, if there are any alphas, they should be on the outside
    // because they will be lower valued than potential betas

    if (alphas == 2)
    {
        // we have a mix of alphas/betas
        // so, we must split them
        if (q.spin == Spin::Up)
        {
            std::swap(p, q);
            new_coeff *= -1.0;
        }

        if (r.spin == Spin::Up)
        {
            std::swap(r, s);
            new_coeff *= -1.0;
        }
    }
    else
    {
        // everything is either all alphas or all betas
        // so we should just order by orbital
        if (q.orbital < p.orbital)
        {
            std::swap(p, q);
            new_coeff *= -1.0;
        }

        if (r.orbital < s.orbital)
        {
            std::swap(r, s);
            new_coeff *= -1.0;
        }
    }

    return std::make_pair(TwoBodyTerm(std::make_tuple(p, q, r, s)), new_coeff);
}

// UPDATE THE DATA MODEL

void Hamiltonian::_UpdateDataModel_OneBody(std::pair<OneBodyTerm, double> info)
{
    OneBodyTerm key;
    double delta;
    std::tie(key, delta) = info;

    // don't make modifications if the coefficient is zero
    if (delta == 0)
    {
        return;
    }

    double existing_coeff = one_body_terms[key];

    // reset with the new value
    one_body_terms[key] = existing_coeff + delta;
}

void Hamiltonian::_UpdateDataModel_TwoBody(std::pair<TwoBodyTerm, double> info)
{
    TwoBodyTerm key;
    double delta;
    std::tie(key, delta) = info;

    // don't make modifications if the coefficient is zero
    if (delta == 0)
    {
        return;
    }

    double existing_coeff = two_body_terms[key];

    // reset with the new value
    two_body_terms[key] = existing_coeff + delta;
}

// HAMILTONIAN METHODS -----------------------------

std::complex<double> Hamiltonian::pp_term(unsigned p) const
{
    using namespace std::complex_literals;
    assert(p >= 0 && p < 2 * max_number_of_orbitals);

    std::pair<int, int> key = std::make_pair(p, p);

    if (indexed_one_body_terms.count(key) == 0)
    {
        // the index does not have a term, so the coefficient should remain unchanged
        return std::complex<double>(1.0);
    }

    return std::exp(-1i * indexed_one_body_terms.at(key) * t_);
}

std::array<std::complex<double>, 4> Hamiltonian::pq_term(unsigned p, unsigned q, unsigned parity) const
{
    using namespace std::complex_literals;

    // verify that we do not double count / have valid idxs
    assert(p >= 0 && p < 2 * max_number_of_orbitals);
    assert(q >= 0 && q < 2 * max_number_of_orbitals);
    assert(p != q);
    assert(spin(p) == spin(q));
    assert(parity == 0 || parity == 1);
    assert(p > q);

    std::complex<double> m00, m01;
    std::pair<int, int> key = std::make_pair(p, q);

    if (indexed_one_body_terms.count(key) == 0)
    {
        // term is not found, so coefficient should be unchanged;
        m00 = std::complex<double>(1.0);
        m01 = std::complex<double>(0.0);
    }
    else
    {
        // grab the coefficient
        double coeff = indexed_one_body_terms.at(key);
        m00 = std::cos(coeff * t_);
        m01 = -1i * static_cast<std::complex<double>>(powint(-1, parity)) * std::sin(coeff * t_);
    }

    return {m00, m01, m01, m00};
}

std::complex<double> Hamiltonian::pqqp_term(unsigned p, unsigned q) const
{
    using namespace std::complex_literals;
    assert(p >= 0 && p < 2 * max_number_of_orbitals);
    assert(q >= 0 && q < 2 * max_number_of_orbitals);
    assert(p != q);
    // assert(spin(p) != spin(q) || p > q);
    assert(spin(p) != spin(q) || q > p);

    std::tuple<int, int, int, int> key = std::make_tuple(p, q, q, p);

    if (indexed_two_body_terms.count(key) == 0)
    {
        // Again, we should just return 1
        return std::complex<double>(1.0);
    }
    else
    {
        double coeff = indexed_two_body_terms.at(key);
        return std::exp(-1i * coeff * t_);
    }
}

std::array<std::complex<double>, 4> Hamiltonian::pqqr_term(unsigned p, unsigned q, unsigned r, unsigned parity) const
{
    using namespace std::complex_literals;
    assert(p >= 0 && p < 2 * max_number_of_orbitals);
    assert(r >= 0 && r < 2 * max_number_of_orbitals);
    assert(r > p);
    assert(parity == 0 || parity == 1);

    // Perform the computation
    auto key = std::make_tuple(p, q, q, r);
    std::complex<double> m00, m01;

    if (indexed_two_body_terms.count(key) == 0)
    {
        m00 = std::complex<double>(1.0);
        m01 = std::complex<double>(0.0);
    }
    else
    {
        double coeff = indexed_two_body_terms.at(key);
        m00 = std::cos(coeff * t_);
        m01 = -1i * static_cast<std::complex<double>>(powint(-1, parity)) * std::sin(coeff * t_);
    }
    return {m00, m01, m01, m00};
}

std::array<std::complex<double>, 4> Hamiltonian::pqrs_term(unsigned p, unsigned q, unsigned r, unsigned s, unsigned parity) const
{
    using namespace std::complex_literals;
    assert(p >= 0 && p < 2 * max_number_of_orbitals);
    assert(r >= 0 && r < 2 * max_number_of_orbitals);
    // assert(p > q);  // not sure what assertions we can actually make
    // assert(r > s);
    // assert(p > s);

    auto key = std::make_tuple(p, q, r, s);
    std::complex<double> m00, m01;

    if (indexed_two_body_terms.count(key) == 0)
    {
        m00 = std::complex<double>(1.0);
        m01 = std::complex<double>(0.0);
    }
    else
    {
        double coeff = indexed_two_body_terms.at(key);
        m00 = std::cos(coeff * t_);
        m01 = -1i * static_cast<std::complex<double>>(powint(-1, parity)) * std::sin(coeff * t_);
    }

    return {m00, m01, m01, m00};
}
