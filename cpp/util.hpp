#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <cstdint>
#include <complex>
#include <iostream>

using U64 = uint64_t;

/**
 * @brief integer version of pow(base, exponent) giving base**exponent
 *
 * @param base base to be exponentiated
 * @param exponent the exponent
 * @return constexpr int base**exponent
 */
inline constexpr int powint(int base, int exponent) {
  int result = 1;
  for (int i = 0; i < exponent; i++) {
    result *= base;
  }
  return result;
}

/**
 * @brief Get the next subset of @param set with the same number of bits set as
 * @param sub
 *
 * @param sub Subset of set
 * @param set Set being enumerated
 * @return U64 Lexicographically next subset of @param set with same number of
 * set bits as @param sub
 */
inline U64 snoob(U64 sub, U64 set) {
  U64 tmp = sub - 1;
  U64 rip = set & (tmp + (sub & (0 - sub)) - set);
  for (sub = (tmp & sub) ^ rip; sub &= sub - 1; rip ^= tmp, set ^= tmp)
    tmp = set & (0 - set);
  return rip;
}

/**
 * @brief Combination: n choose k
 *
 * @param n size of set chosen from
 * @param k number of elements chosen
 * @return constexpr uint64_t combination (n, k)
 */
inline constexpr uint64_t nChoosek(int n, int k) {
  if (k > n) return 0;
  if (k * 2 > n) k = n - k;
  if (k == 0) return 1;

  uint64_t result = n;
  for (int i = 2; i <= k; ++i) {
    result *= (n - i + 1);
    result /= i;
  }
  return result;
}

constexpr uint64_t next_comb(uint64_t x) {
  unsigned long u = x & -x;  // extract rightmost bit 1; u =  0'00^a10^b
  unsigned long v =
      u +
      x;  // set last non-trailing bit 0, and clear to the right; v=x'10^a00^b
  return v +
         (((v ^ x) / u) >>
          2);  // v^x = 0'11^a10^b, (v^x)/u = 0'0^b1^{a+2}, and x ← x'100^b1^a
}


constexpr unsigned parity(uint64_t x) {
  x = (x & 0xFFFFFFFFul) ^ (x >> 32);
  x = (x & 0x0000FFFFul) ^ (x >> 16);
  x = (x & 0x000000FFul) ^ (x >> 8);
  x = (x & 0x0000000Ful) ^ (x >> 4);
  x = (x & 0x00000003ul) ^ (x >> 2);
  x = (x & 0x00000001ul) ^ (x >> 1);
  return x & 1;
}

constexpr bool is_power_of_2(uint64_t x) { return (x & (x - 1)) == 0; }

//assume x != 0
constexpr unsigned int_log_2(uint64_t x) {
  static_assert(sizeof(unsigned long) == sizeof(uint64_t));
  return (63 - __builtin_clzl(x));
}

constexpr unsigned int_pop_count(uint64_t x) {
  static_assert(sizeof(unsigned long) == sizeof(uint64_t));
  return __builtin_popcountl(x);
}

inline void compute_zeroes_and_ones(uint64_t bv, unsigned n,
                             std::vector<unsigned>& zeroes,
                             std::vector<unsigned>& ones) {
  zeroes.clear();
  ones.clear();
  for (uint64_t i = 0, bv1 = bv; i < n; i++, bv1 >>= 1) {
    if (bv1 & 1) {
      ones.push_back(i);
    } else {
      zeroes.push_back(i);
    }
  }
}

template<typename T>
std::ostream&
operator << (std::ostream& os, const std::complex<T>& c) {
  os << "[" << c.real() << ", " << c.imag() << "]";
  return os;
}

template <typename T>
void print_array(std::ostream& os, const std::string& msg, T* ptr, unsigned n) {
  if (n) {
    os << msg;
    for (unsigned i = 0; i < n; i++) {
      os << ptr[i] << " ";
    }
    os << "\n";
  }
}

inline unsigned reverse_bits(unsigned val, unsigned num_of_bits) {
  std::vector<bool> bitrep(num_of_bits, false);

  // the Value can be, at most, the 111...1 vector
  assert((val >> num_of_bits) == 0);

  unsigned tmp = val;

  // create the bit representation
  for (unsigned idx = 0; idx < num_of_bits; idx++) {
    bitrep[idx] = tmp & 1;
    tmp >>= 1;
  }

  // read it in reverse
  unsigned rev_val = 0;
  for (unsigned idx = 0; idx < bitrep.size(); idx++) {
    rev_val <<= 1;
    rev_val += (unsigned)bitrep.at(idx);
  }

  return rev_val;
}

inline void print_precise_array(std::vector<std::complex<double>> arr)
{
  unsigned idx = 0;
  std::cout.precision(17);
  for (auto s : arr)
  {
    if (s != std::complex<double>(0))
    {

    std::cout << idx << " " << std::fixed << s << std::endl;
    }
    idx++;
  }
  std::cout << "--------------------------------"
            << "\n";
  std::cout << "\n";
}

inline unsigned ComputeParityPQRS(unsigned bv, unsigned p, unsigned q, unsigned r, unsigned s)
{
    unsigned tmp = bv;
    unsigned parity = 0;
    for (unsigned idx = 0; idx < s; idx++, tmp >>= 1)
    {
        if ((tmp & 1) == 1)
        {
            parity += 1;
        }
    }

    tmp = bv;
    for (unsigned idx = 0; idx < r; idx++, tmp >>= 1)
    {
        // essentially, we should account for when we have gone over the newly destroyed orbital
        if (idx == s)
        {
            // no op
        }
        else if (((tmp & 1) == 1))
        {
            parity += 1;
        }
    }

    tmp = bv;
    for (unsigned idx = 0; idx < q; idx++, tmp >>= 1)
    {
        if (idx == s || idx == r)
        {
            // No-op- orbitals are empty now
        }
        else if (((tmp & 1) == 1))
        {
            parity += 1;
        }
    }

    tmp = bv;
    for (unsigned idx = 0; idx < p; idx++, tmp >>= 1)
    {
        if (idx == s || idx == r)
        {
            // No-op; we this orbital is newly empty
        }
        else if (((tmp & 1) == 1) || idx == q)
        {
            // the q orbital is newly filled
            parity += 1;
        }
    }

    return parity;
}

#endif // UTIL_HPP_
