#ifndef GATE_HPP_
#define GATE_HPP_

#include <array>
#include <complex>

struct Gate1D {
  Gate1D() = default;
  constexpr Gate1D(std::complex<double> v0, std::complex<double> v1,
                   std::complex<double> v2, std::complex<double> v3)
      : Gate1D{std::array<std::complex<double>, 4>{v0, v1, v2, v3}} {}
  constexpr Gate1D(const std::array<std::complex<double>, 4>& values)
      : values_{values} {}
  std::complex<double> operator()(unsigned i, unsigned j) const {
    assert(i >= 0 && i < 2);
    assert(j >= 0 && j < 2);
    return values_[i * 2 + j];
  }
  std::array<std::complex<double>, 4> values_;
}; // Gate1D

inline constexpr Gate1D Hadamard() {
  return {1.0 / std::sqrt(2.0), 1.0 / sqrt(2.0), 1.0 / std::sqrt(2.0),
          -1.0 / std::sqrt(2.0)};
}

inline Gate1D Rotate(unsigned m) {
  const double pi = 3.14159265358979323846;
  using namespace std::complex_literals;
  return {1.0, 0.0i, 0.0i,
          std::exp(2 * pi * 1.0i / std::pow(2.0 + 0.0i, m))};
}

inline Gate1D InvRotate(unsigned m) {
  const double pi = 3.14159265358979323846;
  using namespace std::complex_literals;
  return {1.0, 0.0i, 0.0i,
          std::exp(-2 * pi * 1.0i / std::pow(2.0 + 0.0i, m))};
}

#endif // GATE_HPP_
