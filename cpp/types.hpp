#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <cstdint>
#include <tuple>
#include <iostream>

using BitPosition = unsigned;
using CombOrder = unsigned;
using BitVector = uint64_t;
using Offset = int64_t;

// Hamiltonian data structures-----------------

enum SpinIndex
{
    HalfUp,
    UpDown
};

enum Spin
{
    Up,
    Down
};

struct RaisingLoweringOperator {
    int orbital;
    Spin spin;
    bool raising;
};

class SpinOrbital
{
public:
    int orbital;
    Spin spin;
    SpinOrbital()
    {
        orbital = 0;
        spin = Spin::Up;
    };

    SpinOrbital(int orb, Spin sp)
    {
        orbital = orb;
        spin = sp;
    };

    int ConvertToInt(SpinIndex convention, int num_of_orbitals = 0)
    {
        if (convention == SpinIndex::HalfUp)
        {
            return (spin == Spin::Up ? 0 : 1) * num_of_orbitals + orbital - 1;
        }
        else if (convention == SpinIndex::UpDown)
        {
            return 2 * (orbital - 1) + spin == Spin::Up ? 0 : 1;
        }
        else
        {
            throw "Unknown Convention";
        }
    }

    bool operator<(const SpinOrbital &t) const
    {
        // // currently using the updown convention to order the operators
        // if (t.orbital != this->orbital)
        // {
        //     return this->orbital < t.orbital;
        // }
        // return this->spin < t.spin;

        if (t.spin != this->spin)
        {
            return this->spin < t.spin;
        }

        return this->orbital < t.orbital;
    };

    bool operator==(const SpinOrbital &t) const
    {
        if (t.orbital == this->orbital && t.spin == this->spin)
        {
            return true;
        }
        return false;
    };

    bool operator!=(const SpinOrbital &t) const
    {
        return !(*this == t);
    };
};

inline std::ostream &operator<<(std::ostream &strm, const SpinOrbital &t)
{
    return strm << t.orbital << (t.spin == Spin::Up ? "^" : "_");
};

// could these be union data types?
typedef std::pair<SpinOrbital, SpinOrbital> OneBodyTerm;
typedef std::tuple<SpinOrbital, SpinOrbital, SpinOrbital, SpinOrbital> TwoBodyTerm;

struct OneBodyTermComp
{
    bool operator()(const OneBodyTerm &lhs, const OneBodyTerm &rhs) const
    {
        return lhs < rhs;
    }
};

struct TwoBodyTermComp
{
    bool operator()(const TwoBodyTerm &lhs, const TwoBodyTerm &rhs) const
    {
        return lhs < rhs;
    }
};

typedef std::pair<std::pair<unsigned, unsigned>, double> IndexedSingleBodyTerm;
typedef std::pair<std::tuple<unsigned, unsigned, unsigned, unsigned>, double> IndexedDoubleBodyTerm;

inline std::ostream &operator<<(std::ostream &strm, const IndexedSingleBodyTerm data)
{
    std::pair<unsigned, unsigned> targets;
    double coeff;
    unsigned p, q;

    std::tie(targets, coeff) = data;
    std::tie(p, q) = targets;
    return strm << p << " " << q << " : " << coeff << "\n";
};

inline std::ostream &operator<<(std::ostream &strm, const IndexedDoubleBodyTerm data)
{
    std::tuple<unsigned, unsigned, unsigned, unsigned> targets;
    double coeff;
    unsigned p, q, r, s;

    std::tie(targets, coeff) = data;
    std::tie(p, q, r, s) = targets;
    return strm << p << " " << q << " " << r << " " << s << " : " << coeff << "\n";
};

#endif // TYPES_HPP_
