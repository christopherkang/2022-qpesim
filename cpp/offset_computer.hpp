#ifndef OFFSET_COMPUTER_HPP_
#define OFFSET_COMPUTER_HPP_

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <numeric>
#include <vector>

#include "types.hpp"
#include "util.hpp"

class OffsetComputer {
 public:
  static Offset rank(BitVector bv) {
    Offset ret = 0;
    BitVector bv1;
    unsigned i, k;
    for (i = 0, k = 1, bv1 = bv; bv1 != 0; ++i, bv1 >>= 1) {
      if (bv1 & 1) {
        ret += nChoosek(i, k);
        ++k;
      }
    }
    return ret;
  }

  static Offset rank_with_double_swap(BitVector bv, unsigned p, unsigned q, unsigned r, unsigned s, unsigned n, unsigned e) {
    // we need to prep the new bv
    // so, let's iterate through and change values

    BitVector tmp = bv;
    for (unsigned idx = 0; idx < n; idx++, tmp >>= 1) {
      if (idx == p || idx == q) {
        // we need to create an electron
        // verify that there is a 0
        if ((tmp & 1) == 1) {
          throw "Unexpected electron found";
        }

        bv += 1 << idx;
      } else if (idx == r || idx == s) {
        // we need to delete an electron
        // verify that there is a 1
        if ((tmp % 2) == 0) {
          throw "Unexpected electron lacking";
        }

        bv -= 1 << idx;
      }
    }

    return rank(bv);
  }

  static Offset rank_of_full_bv(BitVector bv, unsigned n, unsigned e) {
    // we need to split the bitvector into the alpha / beta portions
    // the assumption is that it is bv_beta | bv_alpha
    BitVector bv_beta = bv >> n;
    BitVector bv_alpha = bv & ((1 << e) - 1);

    // now, identify the rank of each
    Offset beta_rank = rank(bv_beta);
    Offset alpha_rank = rank(bv_alpha);

    // and then combine to identify the final rank of the bv
    return alpha_rank * nChoosek(n, e) + beta_rank;
  }

  OffsetComputer(unsigned n, BitVector bv, Offset bv_rank)
      : n_{n}, bv_{bv}, bv_rank_{bv_rank}, bits_(n, 0), order_(n, 0) {
    assert(rank(bv) == bv_rank);
    for (unsigned i = 0; i < n; ++i) {
      if ((bv_ >> i) & 1) {
        order_[i] = 1;
        bits_[i] = 1;
      }
    }
    std::partial_sum(bits_.begin(), bits_.end(), order_.begin());
    for (unsigned v = 0; v < 2; ++v) {
      GEP_[v].resize(n_);
      GEM_[v].resize(n_);
      GTP_[v].resize(n_);
      GTM_[v].resize(n_);
    }
    for (unsigned v = 0; v < 2; ++v) {
      for (unsigned i = 0; i < n_; i++) {
        if (bits_[i]) {
          GEP_[v][i] = -nChoosek(i, order(i)) + nChoosek(i, order(i) + (v + 1));
          GEM_[v][i] = -nChoosek(i, order(i)) + nChoosek(i, static_cast<int>(order(i)) - (v + 1));
        } else {
          GEP_[v][i] = 0;
          GEM_[v][i] = 0;
        }
      }
    }
    for (unsigned v = 0; v < 2; ++v) {
      for (int i = n_ - 2; i >= 0; --i) {
        GEP_[v][i] += GEP_[v][i + 1];
        GEM_[v][i] += GEM_[v][i + 1];
      }
    }
    for (unsigned v = 0; v < 2; ++v) {
      GTP_[v][n_ - 1] = 0;
      GTM_[v][n_ - 1] = 0;
      for (int i = n_ - 2; i >= 0; --i) {
        GTP_[v][i] = GEP_[v][i + 1];
        GTM_[v][i] = GEM_[v][i + 1];
      }
    }
  }

  unsigned n() const { return n_; }

  unsigned bit(BitPosition p) const {
    assert(p < n_);
    return bits_[p];
  }

  CombOrder order(BitPosition p) const {
    assert(p < n_);
    return order_[p];
  }

  Offset rank() const { return bv_rank_; }

  Offset rank_with_swap(BitPosition i, BitPosition j) const {
    assert(i < n_);
    assert(j < n_);
    if (bits_[i] == bits_[j]) {
      return bv_rank_;
    }
    unsigned lo = std::min(i, j);
    unsigned hi = std::max(i, j);

    if (bits_[lo] == 1) {
      return bv_rank_ + destroy_bit(lo) + create_bit(hi, order_[hi]) +
             minus<1>(lo, hi);
    } else {
      return bv_rank_ + destroy_bit(hi) + create_bit(lo, order_[lo] + 1) +
             plus<1>(lo, hi);
    }
  }

  Offset rank_with_double_swap(BitPosition i, BitPosition j, BitPosition k,
                               BitPosition l) {
    assert(i < n_);
    assert(j < n_);
    assert(k < n_);
    assert(l < n_);

    unsigned bits_sum = bits_[i] + bits_[j] + bits_[k] + bits_[l];
    if (bits_sum == 0 || bits_sum == 4) {
      return bv_rank_;
    }
    assert(bits_sum == 2);

    std::array<BitPosition, 4> positions{i, j, k, l};
    std::sort(positions.begin(), positions.end());
    unsigned config = bits_[positions[0]] + (bits_[positions[1]] << 1) + (bits_[positions[2]] << 2) +
                      (bits_[positions[3]] << 3);

    unsigned ret;
    switch (config) {
      case 0b0011:
        ret = bv_rank_ + minus<1>(positions[0], positions[1]) +
              minus<2>(positions[1], positions[2]) +
              minus<1>(positions[2], positions[3]) + destroy_bit(positions[0]) +
              destroy_bit(positions[1]) +
              create_bit(positions[2], order_[positions[2]] - 1) +
              create_bit(positions[3], order_[positions[3]]);
        break;
      case 0b0101:
      case 0b0110:
      case 0b1001:
      case 0b1010:
        ret = rank_with_swap(i, j) + rank_with_swap(k, l) - bv_rank_;
        break;
      case 0b1100:
        ret = bv_rank_ + plus<1>(positions[0], positions[1]) +
              plus<2>(positions[1], positions[2]) +
              plus<1>(positions[2], positions[3]) + destroy_bit(positions[2]) +
              destroy_bit(positions[3]) +
              create_bit(positions[0], order_[positions[0]] + 1) +
              create_bit(positions[1], order_[positions[1]] + 2);
        break;
      default:
        assert(0);  // unreachable
    }
    return ret;
  }

 private:
  template <int N>
  struct PLUS {
    Offset operator()(OffsetComputer& offs, BitPosition p) {
      assert(p < offs.n());
      return -nChoosek(p, offs.order(p)) + nChoosek(p, offs.order(p) + N);
    }
  };

  template <int N>
  struct MINUS {
    Offset operator()(OffsetComputer& offs, BitPosition p) {
      assert(p < offs.n());
      return -nChoosek(p, offs.order(p)) + nChoosek(p, offs.order(p) - N);
    }
  };

  Offset destroy_bit(BitPosition b) const {
    assert(b < n_);
    return -nChoosek(b, order_[b]);
  }

  Offset create_bit(BitPosition b, CombOrder order) const {
    assert(b < n_);
    return nChoosek(b, order);
  }

  /**
   * @brief shift 1 positions in (lo,hi) (excluding lo and hi in the interval)
   * by N position left in the rank ordering of the 1s chosen in this
   * combination
   *
   * @tparam N
   * @param lo
   * @param hi
   * @return int64_t
   */
  template <int N>
  Offset plus(BitPosition lo, BitPosition hi) const {
    static_assert(N == 1 || N == 2);
    return GTP_[N - 1][lo] - GEP_[N - 1][hi];
  }

  template <int N>
  Offset minus(BitPosition lo, BitPosition hi) const {
    static_assert(N == 1 || N == 2);
    return GTM_[N - 1][lo] - GEM_[N - 1][hi];
  }

  unsigned n_;
  BitVector bv_;
  Offset bv_rank_;
  std::vector<unsigned> bits_;
  std::vector<CombOrder> order_;
  std::vector<Offset> GTP_[2];
  std::vector<Offset> GEP_[2];
  std::vector<Offset> GTM_[2];
  std::vector<Offset> GEM_[2];
}; // class OffsetComputer

#include "doctest/doctest.h"
#include <bitset>
#include <cstdlib>
#include <iostream>

TEST_CASE("offset computing test: swap one zero-one pair") {
  unsigned n = 15, COUNT = 5;
  std::srand(0);
  while (COUNT--) {
    BitVector bv{0};
    for (unsigned i = 0; i < n; ++i) {
      if (std::rand() % 2 == 1) {
        bv |= 1ull << i;
      }
    }
    OffsetComputer oc{n, bv, OffsetComputer::rank(bv)};
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < n; j++) {
        if (((bv >> i) & 1) ^ ((bv >> j) & 1)) {
          REQUIRE_EQ(oc.rank_with_swap(i, j),
                     OffsetComputer::rank(bv ^ (1ull << i) ^ (1ull << j)));
        }
      }
    }
  }
}

TEST_CASE("offset computing test: swap one one-one or zero-zero pair") {
  unsigned n = 15, COUNT = 5;
  std::srand(0);
  while (COUNT--) {
    BitVector bv{0};
    for (unsigned i = 0; i < n; ++i) {
      if (std::rand() % 2 == 1) {
        bv |= 1ull << i;
      }
    }
    OffsetComputer oc{n, bv, OffsetComputer::rank(bv)};
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = 0; j < n; j++) {
        if (((bv >> i) & 1) == ((bv >> j) & 1)) {
          REQUIRE_EQ(oc.rank_with_swap(i, j), OffsetComputer::rank(bv));
        }
      }
    }
  }
}

TEST_CASE("offset computing test: double swaps") {
  unsigned n = 20, COUNT = 5;
  std::srand(0);
  while (COUNT--) {
    BitVector bv{0};
    for (unsigned i = 0; i < n; ++i) {
      if (std::rand() % 2 == 1) {
        bv |= 1ull << i;
      }
    }
    OffsetComputer oc{n, bv, OffsetComputer::rank(bv)};
    for (unsigned i = 0; i < n; i++) {
      for (unsigned j = i + 1; j < n; j++) {
        for (unsigned k = j + 1; k < n; k++) {
          for (unsigned l = k + 1; l < n; l++) {
            auto bit_sum = ((bv >> i) & 1) + ((bv >> j) & 1) + ((bv >> k) & 1) +
                           ((bv >> l) & 1);
            if (bit_sum == 0 || bit_sum == 4) {
              REQUIRE_EQ(oc.rank_with_double_swap(i, j, k, l),
                         OffsetComputer::rank(bv));
            } else if (bit_sum == 2) {
                REQUIRE_EQ(oc.rank_with_double_swap(i, j, k, l),
                           OffsetComputer::rank(bv ^ (1ull << i) ^ (1ull <<
                           j) ^
                                                (1ull << k) ^ (1ull << l)));
            }
          }
        }
      }
    }
  }
}

#endif  // OFFSET_COMPUTER_HPP_
